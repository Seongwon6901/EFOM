# efom_running_v14 — 모듈별 도식 및 상세 설명

이 문서는 `efom_running_v14.ipynb`의 코드를 모듈별로 도식화하고, 각 모듈(논리적 구획)에 포함된 함수와 변수, 동작 및 부작용을 상세하게 설명합니다.

목차
- 개요
- 전체 구조 도식(모듈 간 관계)
- 모듈별 상세 설명
  - Helpers
  - ML Cache Adapter
  - Fidelity (slope gate 및 관련 유틸)
  - Alpha 자동 튜닝 (auto_alpha_until_pass_for_ts)
  - GP 학습/캐시 관련
  - Optimization 및 RCOT 관련
  - Counterfactual 스케줄 및 시뮬레이션
  - Runner (run_production) 흐름
- 개선 제안 및 테스트 아이디어

---

개요
- 목적: 생산 최적화 파이프라인을 실행하는 런너. ML 예측(6개월 lookback), GP 잔차 학습/예측, slope 기반 fidelity gate, α(regularizer) 자동 튜닝, multi-knob RCOT 최적화(anchored objective), 결과 저장(곡선/감사/스냅샷), 카운터팩추얼 시뮬레이션을 수행합니다.
- 모드: `historical`, `closed_loop` (시뮬레이션된 state 적용)
- 주요 외부 의존: `src.gp_residuals` (gpmod), `src.optimizer` (opt), `src.ml_predictor` (MLPredictor)

전체 구조 도식 (ASCII)

  +----------------+      uses      +------------------+      uses      +-----------------+
  |   run_production|--------------->|  ML (MLPredictor) |<---------------|   ML cache CSV   |
  +----------------+                +------------------+                +-----------------+
        |  |
        |  +--------------------------------------------+
        v                                               v
  +----------------+    calls    +-------------------+    calls    +-----------------+
  |  GP training   |-----------> | gpmod.GPResiduals |-----------> | anchored_curve   |
  | (build table)  |             +-------------------+             | rc0_guess_for_geom|
  +----------------+                                              +-----------------+
        |                                                          ^
        |                                                          |
        v                                                          |
  +----------------+                +-----------------+           |
  | Fidelity gates |<---------------| pipeline (SRTO) |-----------+
  +----------------+ predict spot  +-----------------+
        |
        v
  +----------------------+
  | optimize_rcot_for_ts_multi |
  +----------------------+
        |
        v
  +----------------+   saves   +--------------+
  | snapshots/audits|<---------|  outputs CSV  |
  +----------------+          +--------------+

모듈별 상세 설명

1) Helpers
- 주요 함수: `_ts_per_day`, `_safe_merge_csv`, `_safe_merge_pickle`, `realized_margin_from_Y`, `_geometry_label_for_row`, `_norm_geom`, `_bounds_for_geoms`, `_rc_grid_for`
- 역할: 날짜/타임스탬프 처리, 캐시 병합(CSV/피클), geometry 판정, rc grid 생성 등의 공통 유틸 제공
- 부작용: `_safe_merge_csv`와 `_safe_merge_pickle`은 로컬 파일 시스템에 쓰기(to_csv, to_pickle)
- 주의: 파일 I/O는 race condition 가능(동시 실행 시), 대형 파일 병합 시 메모리 사용량 큼

2) ML Cache Adapter
- 클래스: `MLCacheAdapter`
  - 목적: ML 예측 캐시(DataFrame)로부터 `predict_row()` API를 제공하여 기존 ML 인터페이스와 호환시킴
  - 메소드: `transform(X_12h, Y_12h)` (no-op), `predict_row(row_like, **kwargs)`
- 부작용: 없음
- 사용: GP 학습 테이블 구성 시 `ml` 인자로 전달되어 잔차 계산에 사용됨

3) Fidelity 관련 유틸
- `_robust_slope_metrics(curve, prod)`
  - 목적: SRTO(Anchor) slope와 CORR slope의 finite-difference 상관계수 계산 및 sign coverage 비율 판단
  - 반환: `(slope_corr, sign_cov)` — slope_corr은 상관계수(nan 가능), sign_cov은 변화량이 유효한 샘플 비율(0..1)
  - 특이: 작은 변화(95% 백분위 기반) 이하의 ds는 noise로 간주하여 mask 처리
- `_rcot_setter_single_knob`, `_local_rc_grid`, `_knob_from_leg` 등은 knob setter 및 지역 grid 생성용 도우미

4) Knob-level fidelity gate: `build_knob_fidelity_gate(...)`
- 목적: pipeline 상태로부터 활성 leg들을 가져와 각 knob(챔버)별로 slope 기반 fidelity를 평가하고 신뢰할 수 있는 RCOT search 범위를 반환
- 반환: `(bounds_by_knob: dict, knob_fid_df: DataFrame)`
- 동작: anchored_curve_at_ts 호출하여 KEY products(예: Ethylene, Propylene, RPG)에 대해 `_robust_slope_metrics` 실행.
  - 모두 통과 → ±halfspan_ok 범위 신뢰(geombounds로 클립)
  - 실패 → SRTO central-difference fallback(rc0±5C)로 유효성 검사 → 통과 시 ±halfspan_fallback, 실패 시 freeze(rc0,rc0)
- 부작용: pipeline.predict_spot_plant 사용(외부 인프라 의존), anchored_curve_at_ts 호출로 계산 비용 큼

5) Alpha 자동 튜닝: `auto_alpha_until_pass_for_ts(...)`
- 목적: GP의 alpha_overrides를 탐색하여 slope fidelity 조건을 만족시키도록 자동으로 세팅
- 반환: `(overrides, fid, summ)` — overrides: 튜닝된 α 매핑, fid: per-product 상세, summ: 요약(제품×geometry별 성공률)
- 동작 요약:
  1. geoms(활성 geometry) 판정
  2. 초기 패스: α=0 상태에서 각 geometry별 curve 생성, slope metrics 수집 → 실패 목록 산출
  3. 실패 쌍에 대해 alpha_grid를 순회하여 조건 만족 시 해당 α를 overrides에 기록
  4. 최종 overrides 적용 후 fid/summ 재계산 반환
- 부작용: `gp.set_alpha_overrides()` 호출로 GP 내부 상태 변경
- 성능: 실패 수 × alpha_grid 길이 × anchored_curve 비용 → 고비용

6) GP 학습 및 캐시
- 내부 도우미: `ensure_gp_train_for_window(train_start, train_end, Xs, Ys, merged_lims, pipeline, ml_cached)` (run_production 내부 정의)
  - 목적: gp 학습용 훈련 테이블을 `gp_train_cache{cache_tag}.pkl`에 부분 병합하여 보관
  - 동작: 부족한 타임스탬프 범위(need)가 있으면 `gpmod.GPResiduals.build_training_table(...)` 호출 후 `_safe_merge_pickle`으로 병합
  - 반환: 해당 윈도우 범위에 해당하는 훈련 DataFrame
- GP fitting: `gp = gpmod.GPResiduals(...).fit_parallel(df_train_win, n_jobs=GP_JOBS)` → 병렬 모델 학습
- 산출: `gps_dict`(product->sklearn GPR 모델), `feature_cols_gp` 등

7) Optimization 및 RCOT 관련 (opt 모듈 의존)
- 핵심 호출: `opt.optimize_rcot_for_ts_multi(...)` — anchored objective 최적화 수행
- 입력: 현재 row, gps, feature_cols_gp, ml_cached, bounds_by_knob, gp, pipeline, 가격 공급자(pp) 등 다수
- 출력: `res` dict — keys: status, row_opt, yields_opt, margin_current_per_h, margin_opt_per_h, improvement_per_h, rcot_opt, 등
- 부작용: 내부에서 여러 시뮬레이션(Spyro 예측 via pipeline/total_spyro_yield_for_now) 호출 가능

8) Counterfactual 스케줄 및 시뮬레이션
- `build_rcot_schedule_from_recs(rec_df)` — rec_df에서 `rcot_opt_` 접두사 컬럼을 모아 (timestamp, setpoints) 리스트 생성
- `apply_schedule_to_X(X_12h, schedule, start, end, hold)` — schedule을 X 데이터프레임에 적용(구간별로 체크포인트 적용, 기본 동작은 hold_until_next)
- `simulate_path_corrected(...)` — X_sim의 각 타임스탬프에 대해 `opt.corrected_yields_for_row`를 호출하여 Y_sim, M_sim 생성 및 반환

9) Runner: `run_production(...)` 전체 흐름
- 주요 입력: X_12h, Y_12h, merged_lims, pipeline, prices_df, total_spyro_yield_for_now, start, end, mode, closed_loop_opts
- 초기화: OUT_DIR/서브디렉토리 생성, PriceProvider(pp) 생성, gpmod.set_rcot_groups_from_columns(X_12h.columns), fg_consts 생성, X_state 및 Y_sim_state 초기화(closed_loop 전용)
- 일일 루프: `for day in pd.date_range(start, end)`
  1. stamps(그 날의 타임스탬프) 결정 → te, ts_train_start 계산
  2. ML 예측 캐시 확보: `ensure_ml_preds_for(train_stamps, ...)` → ml_cached_train
  3. GP 훈련 윈도우 확보: `ensure_gp_train_for_window(...)` → df_train_win
  4. GP 학습: `gp = GPResiduals(...).fit_parallel(...)` → gps_dict, feature_cols_gp
  5. 각 타임스탬프별 처리(스탬프 루프):
     - 현재 row_current 결정(X_state 또는 X_12h)
     - spot ML baseline (ensure_ml_preds_for single ts) → y0, m0(마진)
     - auto_alpha_until_pass_for_ts → overrides, fid_detail/summary 저장
     - build_knob_fidelity_gate → bounds_by_knob, knob_fid 저장
     - opt.optimize_rcot_for_ts_multi → res (최적화 결과), 출력/로깅, rcot_moves 저장, multi_snapshots 저장
     - price snapshot 및 수익/비용/ΔFG 계산(내부 helper들) → audit items 및 summary 저장
     - anchored_curve 저장(try/except) 및 curve audit 저장
     - rec_rows에 flat 추천 행 누적
  6. (생략된 뒷부분) rec_rows를 종합하여 CSV/pickle로 저장하고, counterfactual 스케줄 빌드 및 `simulate_path_corrected` 호출 후 결과 저장/요약

개선 제안 요약
- 캐시 파일 I/O��� 대한 원자적 쓰기/잠금 추가 권장(동시 실행 대비)
- ML 피팅 비용 최적화(rolling fit, incremental, fit 주기 관리 또는 병렬화)
- 하드코딩된 하이퍼파라미터를 외부 설정 파일로 분리(임계값, alpha_grid 등)
- 예외 처리 강화(외부 모듈 실패 시 per-stamp 실패로 전체 중단 방지)
- 로깅(levelled) 추가 및 모듈별 단위 테스트 보강

테스트 아이디어(단위 테스트 대상)
- `_robust_slope_metrics` 경계 케이스(빈/작은/constant/NaN 포함)
- `_safe_merge_csv`/`_safe_merge_pickle`의 인덱스 중복 처리 및 파일 생성 동작
- `MLCacheAdapter.predict_row`의 존재/비존재 timestamp 처리
- `build_knob_fidelity_gate` 및 `auto_alpha_until_pass_for_ts`는 gpmod/pipeline을 mocking하여 조건별 동작 확인

---

본문은 `efom_running_v14.ipynb`의 코드와 주석을 바탕으로 재구성되었습니다. PDF 버전(동일 내용)은 같은 커밋에 `efom_running_v14_modules_diagram.pdf`로 추가합니다.