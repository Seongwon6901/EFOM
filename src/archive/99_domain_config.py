from dataclasses import dataclass
from typing import Dict, Sequence, Tuple, Optional
import pandas as pd
import numpy as np

@dataclass(frozen=True)
class RCOTBounds:
    naph: Tuple[float, float]
    gas:  Tuple[float, float]

@dataclass(frozen=True)
class FuelGasConstants:
    cp_kcal_per_tonK: float
    dH_eth: float
    dH_pro: float
    dH_fg:  float
    fg_hhv_kcal_per_ton: float
    rcot_ref_C: float

@dataclass
class RollingConfig:
    window_train: int = 360
    window_test:  int = 2
    min_train:    int = 700
    min_test:     int = 2
    start_frac:   float = 0.8
    trust_delta_C: float = 10.0
    objective: str = "per_hour"   # or "per_ton_fresh"

