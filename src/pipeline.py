# -*- coding: utf-8 -*-
"""
Download PI tags listed in the FIRST ROW of an Excel file into a single CSV/Parquet.
If, there are existing output file, then update by concatenating new data and re-saving.
If no existing output file, then create a new one.

- Check if there are existing output files (CSV/Parquet)
- If exist, read and determine the last timestamp
- Use that timestamp as the new START_DATE (plus one interval)
- If not exist, use the configured START_DATE and END_DATE.
- Reads every non-empty cell in row 1 as a tag name
- Pulls INTERVAL interpolated data (or recorded if RECORDING=True)
- Works in chunks to avoid overloading the PI server
- Saves a wide table: index = timestamp, columns = tags
"""


from dataclasses import dataclass
import os
from typing import Any, List, Dict, Tuple, Optional
from pathlib import Path
import pandas as pd
import PIconnect as PI
from PIconnect.PIConsts import AuthenticationMode

# ← We assume you keep using the existing global settings/functions from your script:
# - TZ, INTERVAL, CHUNK_DAYS, RECORDING, TAGS_EXCEL, OUT_CSV, OUT_PARQUET
# - fetch_tags_firstrow(...)
# - PI.PIConfig.DEFAULT_TIMEZONE = TZ (already set somewhere in your script)

# ─────────────────────────────────────────────────────────────────────────────
# Config & Pipeline
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DownloadConfig:
    # --- PI server / auth ---
    pi_server: str = "172.17.21.117"
    auth_mode: AuthenticationMode = AuthenticationMode.WINDOWS_AUTHENTICATION
    pi_username: str = ""         # blank → current Windows user
    pi_password: str = ""     # blank → current Windows user
    pi_domain: Optional[str] = None  # e.g. "MYDOMAIN" when passing explicit Windows creds

    # --- Time window & cadence ---
    tz: str = "Asia/Seoul"
    start_date: str = "2025-07-01 00:00"   # inclusive
    end_date: str   = "2025-08-12 00:00"   # exclusive
    interval: str   = "1m"                 # ignored if recorded=True
    chunk_days: int = 7
    recorded: bool  = False                # True → recorded_values, False → interpolated

    sheet_name: str = "python_import"  # Excel sheet name to read tags from Excel
    column_name: str = "tags"          # Excel column name to read tags from Excel

    sheet_name_config: str = "pipeline_config"  # Excel sheet name to read config from Excel
    column_name_tag_config: str = "tags"        # Excel column name to read config tags from Excel
    column_name_value_config: str = "values"    # Excel column name to read config values from Excel

    # --- Sources / outputs ---
    input_dir: str = Path.cwd() / "input"               # ensure file exists
    tags_excel: str = "EFOM_input_data_tag_list.xlsx"   # first row contains tag names
    out_csv: str = Path.cwd() / "intermediate" / "pi_firstrow.csv"
    out_parquet: str = r""                              # "" to skip parquet

    # --- Behavior ---
    incremental: bool = True            # True → incremental update, False → full re-download
    # NEW
    override_end_with_now: bool = True     # ignore Excel end_date; use current time
    safety_lag_minutes: int = 2            # avoid querying data that may not have landed yet
    align_end_to_interval: bool = True     # snap end to interval boundary when interpolated


class Pipeline:
    def __init__(self, cfg: DownloadConfig):
        self.cfg = cfg
        # Ensure timezone is aligned (no-op if already the same)
        PI.PIConfig.DEFAULT_TIMEZONE = cfg.tz

    # # ── Internal helpers ─────────────────────────────────────────────────────
    # def _import_config_from_excel(self, file_path: str, sheet_name: str, column_name_tag: str, column_name_value: str) -> None:
    #     """Return non-empty cell values from the first row as tag names."""

    #     # Read the entire sheet into a DataFrame
    #     df = pd.read_excel(file_path, sheet_name=sheet_name, dtype={column_name_tag: str})
    #     # Safeguard: normalize column names to lowercase and strip spaces
    #     df.columns = [c.strip().lower() for c in df.columns]
    #     if not {column_name_tag, column_name_value}.issubset(df.columns):
    #         raise ValueError("Excel must contain 'tags' and 'values' columns.")

    #     # Generate a dictionary "tags -> values" from the DataFrame
    #     kv: Dict[str, Any] = (df
    #                         .assign(tags=df[column_name_tag].str.strip().str.lower())
    #                         .dropna(subset=[column_name_tag])
    #                         .groupby(column_name_tag, as_index=True)[column_name_value]
    #                         .last()
    #                         .to_dict())

    #     # Validate required tags
    #     required = {"start_date", "end_date", "interval", "recording"}
    #     missing = required - kv.keys()
    #     if missing:
    #         raise ValueError(f"Missing required tag(s): {', '.join(sorted(missing))}")

    #     start_date = kv["start_date"]
    #     end_date   = kv["end_date"]
    #     interval   = str(kv.get("interval", "1m")).strip()
    #     recorded   = kv.get("recording", False)
    #     incremental  = kv.get("incremental", True)

    #     print(f"[CONFIG] start_date={start_date}, end_date={end_date}, interval={interval}, recording={recorded}, incremental={incremental}")

    #     self.cfg.start_date = start_date
    #     self.cfg.end_date   = end_date
    #     self.cfg.interval   = interval
    #     self.cfg.recorded   = recorded
    #     self.cfg.incremental = incremental

    #     return None
    
    # ── Internal helpers ─────────────────────────────────────────────────────
    def _import_config_from_excel(
        self,
        file_path: str,
        sheet_name: str,
        column_name_tag: str,
        column_name_value: str,
    ) -> None:
        df = pd.read_excel(file_path, sheet_name=sheet_name, dtype={column_name_tag: str})
        df.columns = [c.strip().lower() for c in df.columns]
        if not {column_name_tag, column_name_value}.issubset(df.columns):
            raise ValueError("Excel must contain 'tags' and 'values' columns.")

        kv = (
            df.assign(tags=df[column_name_tag].str.strip().str.lower())
            .dropna(subset=[column_name_tag])
            .groupby(column_name_tag, as_index=True)[column_name_value]
            .last()
            .to_dict()
        )

        # Required keys (end_date required only if we are NOT overriding with 'now')
        required = {"start_date", "interval", "recording"}
        if not self.cfg.override_end_with_now:
            required |= {"end_date"}

        missing = required - kv.keys()
        if missing:
            raise ValueError(f"Missing required tag(s): {', '.join(sorted(missing))}")

        # Apply config from Excel
        self.cfg.start_date  = str(kv["start_date"])
        self.cfg.interval    = str(kv.get("interval", "1m")).strip()
        self.cfg.recorded    = bool(kv.get("recording", False))
        self.cfg.incremental = bool(kv.get("incremental", True))
        if not self.cfg.override_end_with_now:
            self.cfg.end_date = str(kv["end_date"])

        # Log what we ended up with (note end_date shows 'NOW (override)' if we're overriding)
        print(
            "[CONFIG] start_date="
            f"{self.cfg.start_date}, end_date="
            f"{('NOW (override)' if self.cfg.override_end_with_now else self.cfg.end_date)}, "
            f"interval={self.cfg.interval}, recording={self.cfg.recorded}, incremental={self.cfg.incremental}"
        )

    def _parse_interval(self, s: Optional[str]) -> pd.Timedelta:
        """Parse an interval string (e.g., '1m', '5m', '1h') into a Timedelta."""
        if not s:
            return pd.Timedelta(0)
        try:
            return pd.to_timedelta(s)
        except Exception:
            # Fallback for simple minute format like '1m' if needed
            if s.endswith('m') and s[:-1].isdigit():
                return pd.Timedelta(minutes=int(s[:-1]))
            raise

    def _read_existing(self) -> Optional[pd.DataFrame]:
        """Return existing result (CSV/Parquet) if available; otherwise None."""
        if self.cfg.out_csv and os.path.exists(self.cfg.out_csv):
            df = pd.read_csv(self.cfg.out_csv, parse_dates=['timestamp'])
            df = df.set_index('timestamp')
            df.index = df.index.tz_localize(self.cfg.tz) if df.index.tz is None else df.index.tz_convert(self.cfg.tz)
            return df.sort_index()
        if self.cfg.out_parquet and os.path.exists(self.cfg.out_parquet):
            df = pd.read_parquet(self.cfg.out_parquet)
            # If parquet saved 'timestamp' as a column, move it into the index
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError("Existing parquet does not have a datetime index.")
            df.index = df.index.tz_localize(self.cfg.tz) if df.index.tz is None else df.index.tz_convert(self.cfg.tz)
            return df.sort_index()
        return None

    def _decide_window(self, existing: Optional[pd.DataFrame]) -> tuple[pd.Timestamp, pd.Timestamp]:
        """
        Decide the download window based on incremental mode and existing results.
        Returns (start_ts, end_ts).
        """
        start_ts = pd.Timestamp(self.cfg.start_date, tz=self.cfg.tz)
        end_ts   = pd.Timestamp(self.cfg.end_date,   tz=self.cfg.tz)

        if self.cfg.incremental and existing is not None and not existing.empty:
            last_ts = existing.index.max()
            if pd.isna(last_ts):
                return start_ts, end_ts
            if self.cfg.recorded:
                # For recorded mode, there's no fixed cadence; start 1ns after the last timestamp
                inc = pd.Timedelta(nanoseconds=1)
            else:
                inc = self._parse_interval(self.cfg.interval)
                if inc <= pd.Timedelta(0):
                    inc = pd.Timedelta(nanoseconds=1)
            new_start = last_ts + inc
            # Avoid going backward in time
            start_ts = max(start_ts, new_start)
        return start_ts, end_ts

    def _save_all(self, df: pd.DataFrame):
        """Persist outputs as CSV/Parquet according to config."""
        if self.cfg.out_csv:
            os.makedirs(os.path.dirname(self.cfg.out_csv), exist_ok=True)
            df.to_csv(self.cfg.out_csv)
            print(f"[SAVE] CSV  → {self.cfg.out_csv} | shape={df.shape}")
        if self.cfg.out_parquet:
            os.makedirs(os.path.dirname(self.cfg.out_parquet), exist_ok=True)
            df.to_parquet(self.cfg.out_parquet)
            print(f"[SAVE] PARQ → {self.cfg.out_parquet} | shape={df.shape}")

    def _tag_list_import_excel(self, file_path: str, sheet_name: str, column_name: str) -> Optional[pd.Series]:
        """Return non-empty cell values from the first row as tag names."""
        try:
            # Read the entire sheet into a DataFrame
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # Check if the column exists in the DataFrame
            if column_name not in df.columns:
                raise ValueError(
                    f"'{column_name}' column does not exist in sheet '{sheet_name}'. "
                    f"Available columns: {list(df.columns)}"
                )
            
            # Return only the specified column
            return df[column_name]
        
        except Exception as e:
            # Print error if something goes wrong
            print(f"Error occurred: {e}")
        return None


    def _server_kwargs(self) -> Dict:
        """Build kwargs for PIServer based on AUTH_MODE and credentials."""
        if self.cfg.auth_mode == AuthenticationMode.WINDOWS_AUTHENTICATION:
            return dict(
                server=self.cfg.pi_server,
                authentication_mode=self.cfg.auth_mode,
                username=self.cfg.pi_username or None,
                password=self.cfg.pi_password or None,
                domain=self.cfg.pi_domain,
            )
        else:
            return dict(
                server=self.cfg.pi_server,
                authentication_mode=AuthenticationMode.PI_USER_AUTHENTICATION,
                username=self.cfg.pi_username,
                password=self.cfg.pi_password,
            )

    def _get_point(self, srv: PI.PIServer, tag: str):
        """Resolve a tag to a PI point. Prefer exact match, fail on ambiguity."""
        pts = srv.search(tag)
        if not pts:
            raise KeyError(f"Tag not found: {tag}")
        exact = [p for p in pts if getattr(p, "name", None) == tag]
        if exact:
            return exact[0]
        if len(pts) > 1:
            names = ", ".join(getattr(p, "name", "<no-name>") for p in pts[:6])
            raise KeyError(f"Ambiguous search for '{tag}'. Candidates: {names} ...")
        return pts[0]

    def _slices(self, start_ts: pd.Timestamp, end_ts: pd.Timestamp, step_days: int):
        """Yield (start,end) slices to chunk a large request."""
        cur = start_ts
        step = pd.Timedelta(days=step_days)
        while cur < end_ts:
            nxt = min(cur + step, end_ts)
            yield cur, nxt
            cur = nxt

    def _fetch_tags_firstrow(
        self,
        *,
        excel_path: str,
        start: str,
        end_exclusive: str,
        interval: Optional[str] = "1m",
        chunk_days: int = 7,
        recorded: bool = False,
    ) -> pd.DataFrame:
        """
        Download all tags listed in the first row of 'excel_path'.
        - If recorded=False → interpolated at 'interval' (regular cadence)
        - If recorded=True  → recorded (irregular cadence)
        Returns a wide DataFrame indexed by timestamp.
        """
        tags = self._tag_list_import_excel(excel_path, self.cfg.sheet_name, self.cfg.column_name)
        print(f"[TAGS] {len(tags)} tags read from first row of {excel_path}")

        start_ts = pd.Timestamp(start, tz=self.cfg.tz)
        end_ts   = pd.Timestamp(end_exclusive, tz=self.cfg.tz)

        frames = []
        with PI.PIServer(**self._server_kwargs()) as srv:
            # Resolve all points once
            points = {t: self._get_point(srv, t) for t in tags}

            # Pull by chunks
            for s0, s1 in self._slices(start_ts, end_ts, chunk_days):
                cols = {}
                for t, pt in points.items():
                    if recorded:
                        ser = pt.recorded_values(s0, s1)                # irregular
                    else:
                        ser = pt.interpolated_values(s0, s1, interval)  # regular
                    cols[t] = pd.to_numeric(ser, errors="coerce")
                if cols:
                    frames.append(pd.DataFrame(cols))

        if not frames:
            raise RuntimeError("No data returned for requested tags/time window.")

        # Concatenate, sort, de-duplicate chunk boundaries
        df = pd.concat(frames).sort_index()
        df.index.name = "timestamp"
        df = df[~df.index.duplicated(keep="last")]

        # Enforce end exclusivity robustly (no need to parse INTERVAL)
        df = df[(df.index >= start_ts) & (df.index < end_ts)]
        return df

    # ── Public API ───────────────────────────────────────────────────────────
    def run(self) -> pd.DataFrame:
        """
        Run the pipeline:
        - Load existing results (if any)
        - Decide the time window (incremental vs. full)
        - Download data via fetch_tags_firstrow(...)
        - Merge with existing, drop duplicate timestamps
        - Save and return the combined result
        """
        # 0) Import config from Excel (overrides existing settings)
        self._import_config_from_excel(
            file_path=self.cfg.tags_excel,
            sheet_name=self.cfg.sheet_name_config,
            column_name_tag=self.cfg.column_name_tag_config,
            column_name_value=self.cfg.column_name_value_config
        )



        # NEW: override end_date with "now" if requested
        if self.cfg.override_end_with_now:
            now_ts = pd.Timestamp.now(tz=self.cfg.tz) - pd.Timedelta(minutes=self.cfg.safety_lag_minutes)
            if not self.cfg.recorded and self.cfg.align_end_to_interval:
                # snap to interval boundary (e.g., 1m)
                step = self._parse_interval(self.cfg.interval)
                if step > pd.Timedelta(0):
                    # floor to the interval
                    now_ts = pd.Timestamp(((now_ts.value // step.value) * step.value), tz=self.cfg.tz)
                else:
                    now_ts = now_ts.floor('T')
            else:
                now_ts = now_ts.floor('T')  # at least minute precision

            self.cfg.end_date = now_ts.strftime('%Y-%m-%d %H:%M')
            print(f"[CONFIG] end_date overridden → {self.cfg.end_date} ({self.cfg.tz})")


        # 1) Load existing output (if available)
        existing = self._read_existing()

        # 2) Decide the download window
        start_ts, end_ts = self._decide_window(existing)

        print(start_ts, end_ts)

        if start_ts >= end_ts:
            # Nothing new to fetch; return existing as-is
            print(f"[SKIP] No new window to fetch (start={start_ts}, end={end_ts}).")
            if existing is None:
                raise RuntimeError("No existing file and no window to fetch.")
            return existing

        print(self.cfg)
        # 3) Fetch new data
        new_df = self._fetch_tags_firstrow(
            excel_path=self.cfg.tags_excel,
            start=start_ts.strftime('%Y-%m-%d %H:%M'),
            end_exclusive=end_ts.strftime('%Y-%m-%d %H:%M'),
            interval=None if self.cfg.recorded else self.cfg.interval,
            chunk_days=self.cfg.chunk_days,
            recorded=self.cfg.recorded,
        )

        # 4) Merge with existing and drop duplicate timestamps (keep the latest)
        combined = new_df if existing is None else pd.concat([existing, new_df]).sort_index()
        combined = combined[~combined.index.duplicated(keep='last')]

        # 5) Save outputs
        self._save_all(combined)

        return combined