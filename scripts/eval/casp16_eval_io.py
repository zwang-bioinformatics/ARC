"""Config and dataframe helpers shared by the eval orchestrator."""
import json
import os

import pandas as pd

# Packaged per-target GNN local scores under outputs/predictions/CASP16/<target>/<group>/
_LOCAL_PACKAGED_JSON_NAMES = ("relaxed_LOCAL.json", "LOCAL.json")


def packaged_local_json_path(base, target, group):
    """First existing file under base/target/group/ for known packaged-local filenames."""
    d = os.path.join(base, target, group)
    for name in _LOCAL_PACKAGED_JSON_NAMES:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return ""


def load_target_sizes(json_path):
    try:
        with open(json_path) as f:
            return json.load(f).get("approx_target_size", {})
    except Exception:
        return {}


def stoich_filtered_local_df(df_local_stoch):
    return df_local_stoch[df_local_stoch["n_mdl_chains"] == df_local_stoch["n_trg_chains"]]


def target_chains_for_targets(df_local_stoch, target_ids):
    """Map target name (no .pdb) -> n_trg_chains for stoichiometry-filtered rows."""
    df_f = stoich_filtered_local_df(df_local_stoch)
    tcdf = df_f[["trg", "n_trg_chains"]].drop_duplicates()
    tcdf = tcdf.assign(trg=tcdf["trg"].str.replace(".pdb", "", regex=False))
    tcdf = tcdf[tcdf["trg"].isin(target_ids)]
    return dict(zip(tcdf["trg"], tcdf["n_trg_chains"]))


def target_chains_for_targets_safe(df_local_stoch, target_ids, context="target chain mapping"):
    """Same as target_chains_for_targets but warn and return {} on failure."""
    try:
        return target_chains_for_targets(df_local_stoch, target_ids)
    except Exception as e:
        print(f"Warning: Could not load {context}: {e}")
        return {}
