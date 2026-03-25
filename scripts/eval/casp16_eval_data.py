"""Load EMA / CASP tables shared across the pipeline (paths from casp16_eval_paths)."""
import os

import pandas as pd

from casp16_eval_paths import (
    CASP_MODEL_SCORES_CSV,
    EMA_LOCAL_STOCH_CSV,
    RAW_16_DEFAULT,
)

df_local_stoch = pd.read_csv(EMA_LOCAL_STOCH_CSV)
truth = pd.read_csv(CASP_MODEL_SCORES_CSV)
raw = os.environ.get("CASP16_RAW_ROOT", RAW_16_DEFAULT)
