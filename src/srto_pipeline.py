# ======================================================================
# OOP SRTO DLL Integration and RCOT Sweep Pipeline
# ======================================================================

import os
import ctypes as ct
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from tqdm import tqdm
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
# ---- caching / persistence helpers ----
import json, hashlib
from pathlib import Path

# add near other imports
from .ml_predictor import (
    build_virtual_rcots_inplace,
    geometry_from_row,
    active_rcots_for_mean,
    compute_ratio,
)
# ── SRTO/KS2019 component-license sets (1-based indices from Appendix A) ──
# LPG = 1..15 (C1–C4 family, etc.) + 211–213 (H2O/CO/CO2) → never in RPG/PFO
LPG_SET_1B  = set(range(1, 16)) | {211, 212, 213}

# PyGAS components (RPG bucket):
# 16–19, 22–32, 35–38, 41–70, and 75 (Styrene) per Appendix A.
PYGAS_SET_1B = (
    set(range(16, 20)) |
    set(range(22, 33)) |
    set(range(35, 39)) |
    set(range(41, 71)) |
    {75}
)

# Everything else (excluding LPG & PyGAS) defaults to AGO/HGO → PFO
ALL_COMPS_1B = set(range(1, 214))
PFO_SET_1B   = ALL_COMPS_1B - LPG_SET_1B - PYGAS_SET_1B

def _sum_1b(spyout, one_based_indices):
    """Sum SPYOUT over 1-based indices (SPYRO manual), 'spyout' is 0-based."""
    return float(sum(spyout[i - 1] for i in one_based_indices if 1 <= i <= 213))

# ==================== Configuration Classes ====================

@dataclass
class SRTOConfig:
    """Configuration for SRTO DLL interface"""
    dll_folder: Path
    selected_spy7: List[Path]
    component_index: Dict[str, int]
    molecular_weights: Dict[str, float]
    
    def __post_init__(self):
        """Validate configuration"""
        self.dll_folder = Path(self.dll_folder)
        if not self.dll_folder.exists():
            raise ValueError(f"DLL folder does not exist: {self.dll_folder}")
        
        self.selected_spy7 = [Path(p) for p in self.selected_spy7]
        for spy7_path in self.selected_spy7:
            if not spy7_path.exists():
                warnings.warn(f"SPY7 file not found: {spy7_path}")

@dataclass
class RCOTSweepConfig:
    """Configuration for RCOT sweep"""
    rcot_min: float = 780.0
    rcot_max: float = 900.0
    rcot_step: float = 5.0
    chunk_size: int = 10
    n_jobs: int = 4
    save_checkpoints: bool = True
    checkpoint_frequency: int = 5
    output_path: Optional[Path] = None
    
    @property
    def rcot_targets(self) -> np.ndarray:
        """Generate RCOT target values"""
        return np.arange(self.rcot_min, self.rcot_max + self.rcot_step, self.rcot_step)

@dataclass
class FeedConfig:
    """Configuration for feed data"""
    gas_components: List[str] = field(default_factory=lambda: [
        'Ethylene', 'Ethane', 'Propylene', 'Propane', 'n-Butane', 'i-Butane'
    ])
    
    def is_gas_component(self, component: str) -> bool:
        """Check if component is a gas"""
        return component in self.gas_components



# ==================== SRTO DLL Interface ====================

class SRTOInterface:
    """Manages interaction with SRTO.dll"""
    
    def __init__(self, config: SRTOConfig):
        self.config = config
        self.dll = None
        self.buffers = {}
        self._lock = threading.Lock()  # Thread safety for DLL calls
        self._initialize_dll()
    
    def _initialize_dll(self):
        """Initialize DLL and set up function prototypes"""
        dll_path = self.config.dll_folder / "SRTO.dll"
        if not dll_path.exists():
            raise FileNotFoundError(f"SRTO.dll not found at {dll_path}")
        
        self.dll = ct.CDLL(str(dll_path))
        
        # Set up function prototypes
        c_double_p = ct.POINTER(ct.c_double)
        c_int_p = ct.POINTER(ct.c_int)
        
        functions = {
            "SRTO_START": ([c_int_p, c_int_p], None),
            "SRTO_FGEOM2SPYIN": ([ct.c_char_p, ct.c_int64, c_double_p, c_int_p], None),
            "UASSPY7": ([ct.c_char_p, ct.c_int64, c_double_p, c_double_p, 
                        c_double_p, c_double_p, c_int_p], None)
        }
        
        for func_name, (argtypes, restype) in functions.items():
            func = getattr(self.dll, func_name)
            func.argtypes = argtypes
            func.restype = restype
    
    def create_buffers(self) -> Dict[str, Any]:
        """Create SRTO buffers"""
        return {
            'SPYIN': (ct.c_double * 257)(),
            'DSPYIN': (ct.c_double * 257)(),
            'SPYOUT': (ct.c_double * 729)(),
            'DSPYOUT': (ct.c_double * (257 * 729))(),
            'NSERVERS': ct.c_int(0),
            'IRET': ct.c_int()
        }
    
    def get_base_spyin(self, geometry_path: Path) -> np.ndarray:
        """Get base SPYIN from geometry file"""
        with self._lock:
            buffers = self.create_buffers()
            fgeom = ct.create_string_buffer(str(geometry_path).encode('utf-8'), 128)
            
            self.dll.SRTO_FGEOM2SPYIN(
                fgeom, 
                ct.c_int64(128), 
                buffers['SPYIN'], 
                ct.byref(buffers['IRET'])
            )
            
            if buffers['IRET'].value < 0:
                raise RuntimeError(f"FGEOM2SPYIN failed for {geometry_path} (IRET={buffers['IRET'].value})")
            
            return np.array(buffers['SPYIN'])
    
    def run_simulation(self, 
                       geometry_path: Path, 
                       spyin_buffer: ct.Array,
                       buffers: Optional[Dict] = None) -> Dict[str, float]:
        """Run SRTO simulation"""
        if buffers is None:
            buffers = self.create_buffers()
        
        fgeom = ct.create_string_buffer(str(geometry_path).encode('utf-8'), 128)
        
        with self._lock:
            self.dll.SRTO_START(ct.byref(buffers['NSERVERS']), ct.byref(buffers['IRET']))
            
            self.dll.UASSPY7(
                fgeom, ct.c_int64(128),
                spyin_buffer, buffers['DSPYIN'],
                buffers['SPYOUT'], buffers['DSPYOUT'],
                ct.byref(buffers['IRET'])
            )
        
        # Extract results
        spyout = buffers['SPYOUT']

        # (optional) helper aliases (0-based)
        H2   = float(spyout[0])  # 1: Hydrogen
        CH4  = float(spyout[1])  # 2: Methane
        C2H4 = float(spyout[3])  # 4: Ethylene  (see note below)
        C3H6 = float(spyout[6])  # 7: Propylene
        C2H6 = float(spyout[4])  # 5: Ethane
        C3H8 = float(spyout[7])  # 8: Propane

        # Mixed C4s (C4H4..i-C4H10) → 9..15 (1-based) == spyout[8:15] zero-based
        mixed_c4 = float(sum(spyout[j] for j in range(8, 15)))

        # RPG: sum of all PyGAS-licensed components (per Appendix A)
        rpg = _sum_1b(spyout, PYGAS_SET_1B)

        # PFO: AGO ∪ HGO (everything not in LPG or PyGAS)
        pfo = _sum_1b(spyout, PFO_SET_1B)

        # Fuel Gas policy:
        #  - We now report Hydrogen separately as 'Hydrogen'
        #  - 'Fuel_Gas' = methane only by default (set to CH4+CO+CO2 if you prefer)
        FUEL_GAS_CH4_ONLY = True
        if FUEL_GAS_CH4_ONLY:
            fuel_gas = CH4
        else:
            CO  = float(spyout[211 - 1])  # 212: CO (1-based)
            CO2 = float(spyout[213 - 1])  # 213: CO2 (1-based)
            fuel_gas = CH4 + CO + CO2

        return {
            # light components you were already returning
            'Ethane':    C2H6,
            'Propane':   C3H8,
            'C2H4':      C2H4,
            'C3H6':      C3H6,
            'MixedC4':   mixed_c4,

            # new/changed buckets
            'RPG':       rpg,         # PyGAS license ⇒ Reformate/PyGas bucket
            'PFO':       pfo,         # AGO + HGO license ⇒ PFO bucket
            'Hydrogen':  H2,          # report H2 separately
            'Tail_Gas':  fuel_gas,    # methane-only by default (see toggle above)

            'IRET':      buffers['IRET'].value
        }
# ==================== SPYIN Buffer Builder ====================

class SPYINBuilder:
    def __init__(self, config: SRTOConfig, feed_config: FeedConfig, warn_tol: float | None = None):
        """
        warn_tol: if set (e.g., 2.0), warn when sum of provided percents
                  deviates from 100% by more than ±warn_tol. None = no warning.
        """
        self.config = config
        self.feed_config = feed_config
        self.warn_tol = warn_tol


    def build(self, 
              base_spyin: np.ndarray,
              row: pd.Series,
              columns: list[str],
              is_naphtha: bool) -> ct.Array:
        """
        Use values exactly as provided (already in percent).
        No molecular-weight weighting; no normalization.
        """
        buf = base_spyin.copy()
        buf[10:140] = 0.0  # clear HC slice

        comps = []
        vals  = []
        for col in columns:
            idx = self.config.component_index.get(col)
            if idx and 11 <= idx <= 140:
                comps.append(idx - 1)
                vals.append(float(row.get(col, 0.0)))

        if not comps:
            raise ValueError("No valid components found in row")

        # Optional sanity warning only (does not modify values)
        if self.warn_tol is not None:
            s = float(np.nansum(vals))
            if not (100.0 - self.warn_tol <= s <= 100.0 + self.warn_tol):
                warnings.warn(f"Composition sum {s:.3f}% outside ±{self.warn_tol}% window.")

        # Write values as-is (percent)
        for idx, v in zip(comps, vals):
            buf[idx] = v

        return (ct.c_double * 257)(*buf)

    
    def set_rcot(self, spyin_buffer: ct.Array, rcot_value: float):
        """Set RCOT value in SPYIN buffer"""
        # spyin_buffer[4] = 1  # CONOP1
        spyin_buffer[5] = 1  # CONVAL1
        spyin_buffer[6] = rcot_value

# ==================== Feed Data Processor ====================

class FeedDataProcessor:
    """Processes and prepares feed data"""
    
    def __init__(self, feed_config: FeedConfig):
        self.feed_config = feed_config
    
    def prepare_with_ffill(self, feed_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare feed data by forward-filling zero gas components.
        
        Args:
            feed_df: Raw feed DataFrame
        
        Returns:
            DataFrame with fixed gas components
        """
        df_fixed = feed_df.copy()
        
        # Check for zero gas rows
        gas_cols = [col for col in self.feed_config.gas_components if col in df_fixed.columns]
        if not gas_cols:
            return df_fixed
        
        gas_sum = df_fixed[gas_cols].sum(axis=1)
        zero_gas_rows = gas_sum < 0.01
        
        if zero_gas_rows.sum() > 0:
            print(f"Found {zero_gas_rows.sum()} rows with zero/near-zero gas components")
            print("Applying forward-fill correction...")
            
            for col in gas_cols:
                # Replace zeros with NaN, then forward fill
                df_fixed[col] = df_fixed[col].replace(0, np.nan)
                df_fixed[col] = df_fixed[col].fillna(method='ffill').fillna(method='bfill')
            
            # Verify fix
            gas_sum_fixed = df_fixed[gas_cols].sum(axis=1)
            still_zero = gas_sum_fixed < 0.01
            if still_zero.sum() > 0:
                warnings.warn(f"{still_zero.sum()} rows still have zero gas after correction")
        
        return df_fixed
    
    def identify_feed_type(self, geometry_name: str) -> Tuple[str, bool]:
        """
        Identify feed type from geometry name.
        
        Returns:
            Tuple of (feed_type, is_naphtha)
        """
        geom_upper = geometry_name.upper()
        
        if 'NAPH' in geom_upper:
            if 'GAS' not in geom_upper:
                return 'naphtha', True
            else:
                return 'hybrid_naph', True
        else:
            return 'gas', False
    
    def get_columns_for_geometry(self, 
                                 geometry_name: str, 
                                 available_columns: List[str],
                                 component_index: Dict[str, int]) -> List[str]:
        """Get appropriate columns for geometry type"""
        feed_type, _ = self.identify_feed_type(geometry_name)
        
        if feed_type == 'gas':
            # Only gas components
            return [c for c in self.feed_config.gas_components 
                   if c in available_columns and c in component_index]
        else:
            # Naphtha or hybrid - exclude gas components
            return [c for c in available_columns 
                   if c in component_index and c not in self.feed_config.gas_components]

# ==================== RCOT Sweeper ====================

class RCOTSweeper:
    """Performs RCOT sweep simulations"""
    
    def __init__(self,
                 srto_interface: SRTOInterface,
                 spyin_builder: SPYINBuilder,
                 feed_processor: FeedDataProcessor,
                 sweep_config: RCOTSweepConfig):
        self.srto = srto_interface
        self.spyin_builder = spyin_builder
        self.feed_processor = feed_processor
        self.config = sweep_config
        self.results = []
    
    def sweep_sequential(self, 
                        feed_df: pd.DataFrame,
                        progress_bar: bool = True) -> pd.DataFrame:
        """
        Perform sequential RCOT sweep.
        
        Args:
            feed_df: Feed data DataFrame
            progress_bar: Show progress bar
        
        Returns:
            DataFrame with sweep results
        """
        # Prepare feed data
        feed_df = self.feed_processor.prepare_with_ffill(feed_df)
        
        results = []
        
        for geom_path in self.srto.config.selected_spy7:
            geom_name = geom_path.name
            feed_type, is_naphtha = self.feed_processor.identify_feed_type(geom_name)
            
            # Get appropriate columns
            cols = self.feed_processor.get_columns_for_geometry(
                geom_name, 
                list(feed_df.columns),
                self.srto.config.component_index
            )
            
            if not cols:
                warnings.warn(f"No columns for {geom_name}, skipping...")
                continue
            
            print(f"\nProcessing {geom_name} ({feed_type}) with {len(cols)} components...")
            
            # Get base SPYIN
            try:
                base_spyin = self.srto.get_base_spyin(geom_path)
            except Exception as e:
                warnings.warn(f"Failed to get base SPYIN for {geom_name}: {e}")
                continue
            
            # Create local buffers for this geometry
            buffers = self.srto.create_buffers()
            
            # Process each feed sample
            iterator = feed_df.iterrows()
            if progress_bar:
                iterator = tqdm(iterator, total=len(feed_df), 
                              desc=f"{geom_name[:20]}", leave=False)
            
            for _, row in iterator:
                for rcot in self.config.rcot_targets:
                    try:
                        # Build SPYIN
                        spyin = self.spyin_builder.build(
                            base_spyin, row, cols, is_naphtha
                        )
                        
                        # Set RCOT
                        self.spyin_builder.set_rcot(spyin, rcot)
                        
                        # Run simulation
                        sim_results = self.srto.run_simulation(
                            geom_path, spyin, buffers
                        )
                        
                        # Store results
                        results.append({
                            'geometry': geom_name,
                            'feed_type': feed_type,
                            'date': row.get('date', pd.NaT),
                            'RCOT': rcot,
                            **sim_results
                        })
                        
                    except Exception as e:
                        if progress_bar:
                            iterator.set_postfix_str(f"Error: {str(e)[:30]}")
                        continue
            
            # Save checkpoint if configured
            if self.config.save_checkpoints and len(results) > 0:
                self._save_checkpoint(results, geom_name)
        
        return pd.DataFrame(results)
    
    def sweep_parallel(self, 
                      feed_df: pd.DataFrame,
                      max_workers: Optional[int] = None) -> pd.DataFrame:
        """
        Perform parallel RCOT sweep using threading.
        
        Args:
            feed_df: Feed data DataFrame
            max_workers: Maximum number of parallel workers
        
        Returns:
            DataFrame with sweep results
        """
        # Prepare feed data
        feed_df = self.feed_processor.prepare_with_ffill(feed_df)
        
        if max_workers is None:
            max_workers = self.config.n_jobs
        
        # Split RCOT targets into chunks
        rcot_chunks = np.array_split(
            self.config.rcot_targets,
            max(1, len(self.config.rcot_targets) // self.config.chunk_size)
        )
        
        print(f"Processing {len(rcot_chunks)} chunks with {max_workers} workers")
        
        all_results = []
        
        def process_chunk(chunk_idx: int, rcot_chunk: np.ndarray) -> List[Dict]:
            """Process one RCOT chunk"""
            chunk_results = []
            
            for geom_path in self.srto.config.selected_spy7:
                geom_name = geom_path.name
                feed_type, is_naphtha = self.feed_processor.identify_feed_type(geom_name)
                
                cols = self.feed_processor.get_columns_for_geometry(
                    geom_name, list(feed_df.columns), 
                    self.srto.config.component_index
                )
                
                if not cols:
                    continue
                
                try:
                    base_spyin = self.srto.get_base_spyin(geom_path)
                    buffers = self.srto.create_buffers()
                    
                    for _, row in feed_df.iterrows():
                        for rcot in rcot_chunk:
                            try:
                                spyin = self.spyin_builder.build(
                                    base_spyin, row, cols, is_naphtha
                                )
                                self.spyin_builder.set_rcot(spyin, rcot)
                                
                                sim_results = self.srto.run_simulation(
                                    geom_path, spyin, buffers
                                )
                                
                                chunk_results.append({
                                    'geometry': geom_name,
                                    'feed_type': feed_type,
                                    'date': row.get('date', pd.NaT),
                                    'RCOT': rcot,
                                    **sim_results
                                })
                            except Exception:
                                continue
                
                except Exception as e:
                    warnings.warn(f"Chunk {chunk_idx} failed for {geom_name}: {e}")
            
            return chunk_results
        
        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_chunk, i, chunk): i 
                for i, chunk in enumerate(rcot_chunks)
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Chunks"):
                try:
                    chunk_results = future.result()
                    all_results.extend(chunk_results)
                    
                    # Save checkpoint
                    if self.config.save_checkpoints and len(all_results) > 0:
                        chunk_idx = futures[future]
                        if (chunk_idx + 1) % self.config.checkpoint_frequency == 0:
                            self._save_checkpoint(all_results, f"chunk_{chunk_idx}")
                
                except Exception as e:
                    warnings.warn(f"Chunk processing failed: {e}")
        
        return pd.DataFrame(all_results)
    
    def _save_checkpoint(self, results: List[Dict], identifier: str):
        """Save checkpoint to file"""
        if self.config.output_path:
            checkpoint_path = self.config.output_path.parent / f"checkpoint_{identifier}.csv"
            pd.DataFrame(results).to_csv(checkpoint_path, index=False)
            print(f"Saved checkpoint: {checkpoint_path}")

# ==================== Main Pipeline ====================

class SRTOPipeline:
    """Main pipeline for SRTO simulations and RCOT sweeps"""
    
    def __init__(self,
                 srto_config: SRTOConfig,
                 sweep_config: RCOTSweepConfig,
                 feed_config: Optional[FeedConfig] = None):
        self.srto_config = srto_config
        self.sweep_config = sweep_config
        self.feed_config = feed_config or FeedConfig()
        self._base_spyin_cache = {}
        
        
        # Initialize components
        self.srto_interface = SRTOInterface(srto_config)
        self.spyin_builder = SPYINBuilder(srto_config, self.feed_config)
        self.feed_processor = FeedDataProcessor(self.feed_config)
        self.sweeper = RCOTSweeper(
            self.srto_interface,
            self.spyin_builder,
            self.feed_processor,
            sweep_config
        )
    def _base_spyin_cached(self, geom_path: Path) -> np.ndarray:
        key = str(geom_path)
        arr = self._base_spyin_cache.get(key)
        if arr is None:
            arr = self.srto_interface.get_base_spyin(geom_path)
            self._base_spyin_cache[key] = arr
        return arr


    def predict_spot(self,
                     composition_row: pd.Series,
                     geometry_name_or_path: str,
                     rcot: float) -> dict:
        """
        Single-run SRTO prediction for a given composition, geometry, and RCOT.
        """
        # Resolve geometry path
        candidates = [p for p in self.srto_config.selected_spy7
                      if p.name == os.path.basename(geometry_name_or_path) or str(p) == geometry_name_or_path]
        if not candidates:
            raise ValueError(f"Geometry not found in config: {geometry_name_or_path}")
        geom_path = candidates[0]
        geom_name = geom_path.name

        # Identify feed type & columns
        feed_type, is_naphtha = self.feed_processor.identify_feed_type(geom_name)
        cols = self.feed_processor.get_columns_for_geometry(
            geom_name, list(composition_row.index), self.srto_config.component_index
        )
        if not cols:
            raise ValueError(f"No valid component columns for geometry {geom_name}")

        # Build SPYIN
        base_spyin = self._base_spyin_cached(geom_path)
        spyin = self.spyin_builder.build(base_spyin, composition_row, cols, is_naphtha)
        self.spyin_builder.set_rcot(spyin, rcot)

        # Run sim
        out = self.srto_interface.run_simulation(geom_path, spyin)
        out.update({
            'geometry': geom_name,
            'feed_type': feed_type,
            'RCOT': rcot,
            'date': composition_row.get('date', pd.NaT)
        })
        return out
        
    def run_sweep(self, 
                 feed_df: pd.DataFrame,
                 parallel: bool = False,
                 save_results: bool = True) -> pd.DataFrame:
        """
        Run RCOT sweep.
        
        Args:
            feed_df: Feed data DataFrame
            parallel: Use parallel processing
            save_results: Save results to file
        
        Returns:
            DataFrame with sweep results
        """
        print(f"Starting RCOT sweep from {self.sweep_config.rcot_min} to {self.sweep_config.rcot_max}")
        print(f"Processing {len(self.srto_config.selected_spy7)} geometries")
        print(f"Feed samples: {len(feed_df)}")
        
        if parallel:
            results_df = self.sweeper.sweep_parallel(feed_df)
        else:
            results_df = self.sweeper.sweep_sequential(feed_df)
        
        if save_results and self.sweep_config.output_path:
            results_df.to_csv(self.sweep_config.output_path, index=False)
            print(f"Results saved to {self.sweep_config.output_path}")
        
        # Print summary
        self._print_summary(results_df)
        
        return results_df
    
    def _print_summary(self, results_df: pd.DataFrame):
        """Print results summary"""
        print("\n" + "="*50)
        print("SWEEP RESULTS SUMMARY")
        print("="*50)
        print(f"Total rows: {len(results_df)}")
        
        print("\nResults by geometry:")
        print(results_df.groupby(['geometry', 'feed_type']).size())
        
        print("\nAverage yields by geometry:")
        yield_cols = ['C2H4', 'C3H6', 'MixedC4', 'RPG']
        print(results_df.groupby('geometry')[yield_cols].mean())
    
    def validate_results(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        validation = {
            'total_rows': len(results_df),
            'geometries': results_df['geometry'].nunique(),
            'rcot_range': (results_df['RCOT'].min(), results_df['RCOT'].max()),  # ← fix
            'failed_runs': (results_df['IRET'] < 0).sum(),
            'missing_data': results_df.isnull().sum().to_dict()
        }
        
        # Check for anomalies
        anomalies = []
        
        # Check for negative yields
        yield_cols = ['C2H4', 'C3H6', 'MixedC4', 'RPG']
        for col in yield_cols:
            if (results_df[col] < 0).any():
                anomalies.append(f"Negative values in {col}")
        
        # Check for unrealistic yields (>100%)
        total_yield = results_df[yield_cols].sum(axis=1)
        if (total_yield > 100).any():
            anomalies.append(f"Total yield > 100% in {(total_yield > 100).sum()} rows")
        
        validation['anomalies'] = anomalies
        
        return validation
        
    def _find_geometry_path(self, geometry_key_or_name: str) -> Path:
        """
        Accepts exact filename/path or a key among {'GF_GAS','LF_NAPH','GF_HYB_NAPH'}.
        Tries to match spy7 names by tokens if a key is provided.
        """
        # exact match first
        for p in self.srto_config.selected_spy7:
            if p.name == os.path.basename(geometry_key_or_name) or str(p) == geometry_key_or_name:
                return p

        key = geometry_key_or_name.upper()
        want = []
        if key == 'GF_GAS':       want = ['gf', 'gas']
        elif key == 'LF_NAPH':    want = ['lf', 'naph']
        elif key == 'GF_HYB_NAPH':want = ['hyb', 'naph']  # or just ['hyb'] if your naming differs

        def score(path: Path) -> int:
            name = path.name.lower()
            return sum(1 for t in want if t in name)

        if want:
            ranked = sorted(self.srto_config.selected_spy7, key=score, reverse=True)
            if ranked and score(ranked[0]) > 0:
                return ranked[0]
        raise ValueError(f"Could not resolve geometry: {geometry_key_or_name}")

    def _ensure_virtual_rcots_on_row(self, row: pd.Series) -> pd.Series:
        """If 'RCOT_naphtha/gas_chamber{4..6}' are missing, compute them from coil tags."""
        if any(k.startswith('RCOT_naphtha_chamber') for k in row.index):
            return row
        df = pd.DataFrame([row])
        build_virtual_rcots_inplace(df)
        return df.iloc[0]
        
    def predict_spot_auto(self,
                        X_row: pd.Series,
                        composition_row: pd.Series,
                        prefer_geometry: Optional[str] = None,
                        hybrid_strategy: str = "weighted") -> dict:
        """
        Use furnace state (X_row) to choose geometry & RCOT, then run SRTO.
        If hybrid, run LF_NAPH and GF_GAS and blend yields by fresh feed ratios.
        Never blend IRET; report each leg's IRET and mark status.
        """
        xr = self._ensure_virtual_rcots_on_row(X_row)
        auto_geom = geometry_from_row(xr)
        geom_key  = prefer_geometry or auto_geom

        # representative RCOTs
        rc_n, rc_g = active_rcots_for_mean(xr)

        # --- RCOT sanity/guardrails ---
        NAPH_BOUNDS = (780.0, 900.0)
        GAS_BOUNDS  = (780.0, 900.0)
        def sane(v, lo, hi): return np.isfinite(v) and (lo <= v <= hi)
        if not sane(rc_n, *NAPH_BOUNDS):
            rc_n = float(np.clip(rc_n if np.isfinite(rc_n) else 820.0, *NAPH_BOUNDS))
        if not sane(rc_g, *GAS_BOUNDS):
            # fallback: use naphtha RCOT if gas looks bogus
            rc_g = float(np.clip(rc_n, *GAS_BOUNDS))

        ratio_n = compute_ratio(xr)
        ratio_g = 1.0 - ratio_n

        def _run_one(key: str, rc: float) -> dict:
            gpath = self._find_geometry_path(key)
            return self.predict_spot(
                composition_row=composition_row,
                geometry_name_or_path=str(gpath),
                rcot=float(rc)
            )

        if geom_key == 'GF_GAS':
            spot = _run_one('GF_GAS', rc_g)
            spot.update({
                'geometry_auto': auto_geom, 'geometry_used': 'GF_GAS',
                'rcot_n': float(rc_n), 'rcot_g': float(rc_g),
                'ratio_naphtha': float(ratio_n), 'ratio_gas': float(ratio_g),
                'blend': False,
                'status': 'ok' if spot.get('IRET', 0) >= 0 else 'error'
            })
            return spot

        if geom_key == 'LF_NAPH':
            spot = _run_one('LF_NAPH', rc_n)
            spot.update({
                'geometry_auto': auto_geom, 'geometry_used': 'LF_NAPH',
                'rcot_n': float(rc_n), 'rcot_g': float(rc_g),
                'ratio_naphtha': float(ratio_n), 'ratio_gas': float(ratio_g),
                'blend': False,
                'status': 'ok' if spot.get('IRET', 0) >= 0 else 'error'
            })
            return spot

        # HYBRID → run both legs; blend only yield-like fields
        n_leg = _run_one('LF_NAPH', rc_n)
        g_leg = _run_one('GF_GAS',  rc_g)

        # Fields to blend (yields/flow-ish). DO NOT blend IRET.
        BLEND_FIELDS = {'C2H4','C3H6','MixedC4','RPG','PFO','Ethane','Propane','Tail_Gas','Hydrogen'}
        out = {}

        # weighted numeric yields
        for k in BLEND_FIELDS:
            if k in n_leg and k in g_leg:
                out[k] = ratio_n * float(n_leg[k]) + ratio_g * float(g_leg[k])

        # carry context & per-leg info
        out.update({
            'date': composition_row.get('date', pd.NaT),
            'geometry_auto': auto_geom,
            'geometry_used': 'GF_HYB_NAPH',
            'rcot_n': float(rc_n),
            'rcot_g': float(rc_g),
            'ratio_naphtha': float(ratio_n),
            'ratio_gas': float(ratio_g),
            'blend': True,
            # keep per-leg IRET and a combined status
            'irets': {'naph': n_leg.get('IRET', np.nan), 'gas': g_leg.get('IRET', np.nan)},
            'status': 'ok' if (n_leg.get('IRET', 0) >= 0 and g_leg.get('IRET', 0) >= 0) else 'error',
            # optional: keep a separate blended RCOT for reference (not a real setpoint)
            'RCOT_blend': ratio_n * rc_n + ratio_g * rc_g
        })
        return out
    
    # ──────────────────────────────────────────────────────────────────────
    # Chamber-aware spot run → absolute t/h totals
    # ──────────────────────────────────────────────────────────────────────
    def _chamber_legs_from_state(self, xr: pd.Series, feed_thr: float = 0.1):
        """
        Build a list of 'legs' to simulate at chamber level.
        Each leg = one geometry ('LF_NAPH' | 'GF_HYB_NAPH' | 'GF_GAS') for a chamber,
        with its own RCOT and feed flow (t/h).
        """
        legs = []

        # 1–3: naphtha only
        for ch in (1, 2, 3):
            feed_n = float(xr.get(f'Naphtha_chamber{ch}', 0.0))
            if feed_n > feed_thr:
                rc = float(xr.get(f'RCOT_chamber{ch}', np.nan))
                legs.append(dict(
                    chamber=ch, feed='naphtha', geometry='LF_NAPH',
                    rcot=rc, feed_tph=feed_n
                ))

        # 4–6: possibly both feeds
        for ch in (4, 5, 6):
            feed_n = float(xr.get(f'Naphtha_chamber{ch}', 0.0))
            feed_g = float(xr.get(f'Gas Feed_chamber{ch}', 0.0))

            if feed_n > feed_thr:
                rc_n = float(xr.get(f'RCOT_naphtha_chamber{ch}', np.nan))
                legs.append(dict(
                    chamber=ch, feed='naphtha', geometry='GF_HYB_NAPH',
                    rcot=rc_n, feed_tph=feed_n
                ))

            if feed_g > feed_thr:
                rc_g = float(xr.get(f'RCOT_gas_chamber{ch}', np.nan))
                legs.append(dict(
                    chamber=ch, feed='gas', geometry='GF_GAS',
                    rcot=rc_g, feed_tph=feed_g
                ))

        return legs

    def _sanitize_rcot(self, rc: float, bounds=(780.0, 900.0)) -> float:
        lo, hi = bounds
        if not np.isfinite(rc):
            # neutral midpoint if missing
            return 0.5 * (lo + hi)
        return float(min(max(rc, lo), hi))

    def _spot_leg(self, composition_row: pd.Series, geometry_key: str, rcot: float) -> dict:
        """Run a single leg by geometry key, resolving to the actual .SPY7 path."""
        gpath = self._find_geometry_path(geometry_key)
        return self.predict_spot(
            composition_row=composition_row,
            geometry_name_or_path=str(gpath),
            rcot=rcot
        )

    def predict_spot_plant(self,
                           X_row: pd.Series,
                           composition_row: pd.Series,
                           feed_thr: float = 0.1,
                           rcot_bounds=(780.0, 900.0)) -> dict:
        """
        Chamber-aware spot run:
          - detect active furnaces (per chamber, per feed),
          - map to geometry (LF_NAPH for ch1–3; GF_HYB_NAPH/GF_GAS for ch4–6),
          - use chamber's own RCOT for that leg,
          - run SRTO per leg,
          - convert % yields → absolute t/h using that leg's feed flow,
          - sum totals across all legs.
        Returns:
          {
            date, total_feed_tph,
            totals_tph: {Ethylene, Propylene, MixedC4, RPG, Ethane, Propane, Fuel_Gas},
            legs: [ per-leg dicts including pct & tph and IRET ],
            status: 'ok' | 'partial' | 'error'
          }
        """
        # Ensure virtual RCOTs exist for ch4–6
        xr = self._ensure_virtual_rcots_on_row(X_row)

        # Decide legs
        legs = self._chamber_legs_from_state(xr, feed_thr=feed_thr)
        if not legs:
            return dict(
                date=composition_row.get('date', pd.NaT),
                total_feed_tph=0.0,
                totals_tph={k: 0.0 for k in ('Ethylene','Propylene','MixedC4','RPG','PFO','Ethane','Propane','Fuel_Gas','Hydrogen')},
                legs=[],
                status='error',
                message='No active legs found (all feeds below threshold)'
            )

        # aggregate totals (t/h)
        totals = dict(Ethylene=0.0, Propylene=0.0, MixedC4=0.0, RPG=0.0,
                    PFO=0.0, Ethane=0.0, Propane=0.0, Tail_Gas=0.0, Hydrogen=0.0)
        total_feed = 0.0
        leg_outputs = []
        ok_flags = []

        for leg in legs:
            rc = self._sanitize_rcot(leg['rcot'], bounds=rcot_bounds)
            spot = self._spot_leg(composition_row, leg['geometry'], rc)

            # status per leg
            iret = float(spot.get('IRET', 0))
            ok = (iret >= 0)
            ok_flags.append(ok)

            # SRTO yields are % of feed → convert using this leg's feed (t/h)
            f = float(leg['feed_tph'])
            total_feed += f

            # map SRTO keys → canonical names
            # pct mapping inside the loop:
            pct = dict(
                Ethylene=float(spot.get('C2H4', 0.0)),
                Propylene=float(spot.get('C3H6', 0.0)),
                MixedC4=float(spot.get('MixedC4', 0.0)),
                RPG=float(spot.get('RPG', 0.0)),
                PFO=float(spot.get('PFO', 0.0)),
                Ethane=float(spot.get('Ethane', 0.0)),
                Propane=float(spot.get('Propane', 0.0)),
                Tail_Gas=float(spot.get('Tail_Gas', 0.0)),
                Hydrogen=float(spot.get('Hydrogen', 0.0)),
            )

            tph = {k: v * 0.01 * f for k, v in pct.items()}

            # accumulate totals
            for k in totals:
                totals[k] += tph[k]

            leg_outputs.append({
                'chamber': leg['chamber'],
                'feed': leg['feed'],
                'geometry': leg['geometry'],
                'rcot_used': rc,
                'feed_tph': f,
                'pct': pct,      # SRTO % yields
                'tph': tph,      # absolute t/h for this leg
                'IRET': iret,
                'status': 'ok' if ok else 'error'
            })

        status = 'ok' if all(ok_flags) else ('partial' if any(ok_flags) else 'error')

        return dict(
            date=composition_row.get('date', pd.NaT),
            total_feed_tph=total_feed,
            totals_tph=totals,
            legs=leg_outputs,
            status=status
        )


# ==================== Factory Class ====================

class SRTOFactory:
    """Factory for creating SRTO pipeline instances"""
    
    @staticmethod
    def from_files(dll_folder: str,
                   spy7_files: List[str],
                   component_index_file: Optional[str] = None,
                   molecular_weights_file: Optional[str] = None,
                   **kwargs) -> SRTOPipeline:
        """
        Create pipeline from file paths.
        
        Args:
            dll_folder: Path to DLL folder
            spy7_files: List of SPY7 file paths
            component_index_file: Optional JSON file with component indices
            molecular_weights_file: Optional JSON file with molecular weights
            **kwargs: Additional configuration parameters
        
        Returns:
            Configured SRTOPipeline instance
        """
        import json
        
        # Load component index
        if component_index_file and Path(component_index_file).exists():
            with open(component_index_file, 'r') as f:
                component_index = json.load(f)
        else:
            # Use default (would need to be provided)
            component_index = kwargs.get('component_index', {})
        
        # Load molecular weights
        if molecular_weights_file and Path(molecular_weights_file).exists():
            with open(molecular_weights_file, 'r') as f:
                molecular_weights = json.load(f)
        else:
            molecular_weights = kwargs.get('molecular_weights', {})
        
        # Create configs
        srto_config = SRTOConfig(
            dll_folder=Path(dll_folder),
            selected_spy7=[Path(f) for f in spy7_files],
            component_index=component_index,
            molecular_weights=molecular_weights
        )
        
        sweep_config = RCOTSweepConfig(
            rcot_min=kwargs.get('rcot_min', 780.0),
            rcot_max=kwargs.get('rcot_max', 900.0),
            rcot_step=kwargs.get('rcot_step', 5.0),
            chunk_size=kwargs.get('chunk_size', 10),
            n_jobs=kwargs.get('n_jobs', 4),
            output_path=Path(kwargs['output_path']) if 'output_path' in kwargs else None
        )
        
        feed_config = FeedConfig(
            gas_components=kwargs.get('gas_components', 
                                     ['Ethylene', 'Ethane', 'Propylene', 
                                      'Propane', 'n-Butane', 'i-Butane'])
        )
        
        return SRTOPipeline(srto_config, sweep_config, feed_config)
