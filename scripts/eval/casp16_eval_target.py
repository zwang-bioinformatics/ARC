"""Per-target EMA local evaluation: loads predictors, computes metrics, returns rows for local_eval.csv."""
import json
import os

import numpy as np
import pandas as pd

from casp16_eval_constants import (
    CASP16_EMA_RESULTS,
    PACKAGED_PREDICTIONS_BASE,
    group_mappings,
)
from casp16_eval_data import df_local_stoch, raw, truth
from casp16_eval_io import packaged_local_json_path
from casp16_eval_metrics import compute_rs
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


def _load_packaged_local(base, target, group):
    p = packaged_local_json_path(base, target, group)
    if not p:
        return None
    try:
        return json.load(open(p, "r"))
    except Exception:
        return None


_MAX_IDS_IN_LOG = 12


def _fmt_model_ids(ids: list[str]) -> str:
    if not ids:
        return "(none)"
    if len(ids) <= _MAX_IDS_IN_LOG:
        return ", ".join(ids)
    head = ", ".join(ids[:_MAX_IDS_IN_LOG])
    return f"{head} ... (+{len(ids) - _MAX_IDS_IN_LOG} more)"


def _proc_stats(
    target_all_models: set,
    target_filtered_models: set,
    target_kicked_out_models: list,
    true_iface_stats: list,
    log_lines: list[str],
) -> dict:
    """Stats dict for casp16_eval merge; ``_target_log_lines`` is consumed for console only."""
    return {
        "all_models": target_all_models,
        "filtered_models": target_filtered_models,
        "kicked_out_models": target_kicked_out_models,
        "true_iface_stats": true_iface_stats,
        "_target_log_lines": log_lines,
    }


def process_target(target):
    """Process a single target and return results dictionary and statistics"""
    target_results = {
        "method": [],
        "target": [],
        "score": [],
        "metric": [],
        "value": []
    }
    
    # Track models for this target
    target_all_models = set()
    target_filtered_models = set()
    target_kicked_out_models = []
    
    # Track detailed statistics for true_interface_residue metrics (for comparison with notebook)
    true_iface_stats = []
    
    # Store raw data for pooling across targets (for ROC curves and performance tables)
    target_raw_data = {}  # {method: {score_type: DataFrame}}
    
    if not os.path.exists(f"{raw}/{target}/QA_1/"):
        return target_results, _proc_stats(
            target_all_models,
            target_filtered_models,
            target_kicked_out_models,
            [],
            [f"Skipped: no raw QA tree at {raw}/{target}/QA_1/."],
        ), {}
    
    ###################################
    
    # Filter by stoichiometry
    df = df_local_stoch[df_local_stoch["trg"].str.contains(target)]
    df_filtered = df[df["n_mdl_chains"] == df["n_trg_chains"]]
    
    mdls = set(df["mdl"])
    filtered_mdls = set(df_filtered["mdl"])
    assert(filtered_mdls.issubset(mdls))
    kicked_out_mdls = mdls.difference(filtered_mdls)
    kicked_out_mdls = sorted(list(kicked_out_mdls))
    target_kicked_out_models.extend(kicked_out_mdls)
    log_lines: list[str] = [
        f"Stoichiometry: removed {len(kicked_out_mdls)} model(s) (chain mismatch); "
        f"{len(mdls)} CASP models -> {len(filtered_mdls)} retained."
    ]
    if kicked_out_mdls:
        log_lines.append(f"  Chain-mismatch IDs: {_fmt_model_ids(kicked_out_mdls)}")

    ###################################

    # Filter the target based on QSBEST quality score (0.6)
    targ_truth = truth[truth["TARGET"] == target]
    targ_truth = targ_truth[targ_truth["MODEL"].isin(filtered_mdls)]

    if len(targ_truth) == 0:
        log_lines.append(
            "Skipped: no casp_model_scores rows for this target after stoichiometry filter."
        )
        return target_results, _proc_stats(
            target_all_models,
            target_filtered_models,
            target_kicked_out_models,
            [],
            log_lines,
        ), {}

    qsbest_max = float(targ_truth["QSBEST"].max())
    if targ_truth["QSBEST"].max() < 0.6:
        log_lines.append(f"Skipped: QSBEST max {qsbest_max:.3f} < 0.6 (target not evaluated).")
        return target_results, _proc_stats(
            target_all_models,
            target_filtered_models,
            target_kicked_out_models,
            [],
            log_lines,
        ), {}

    log_lines.append(f"QSBEST: max {qsbest_max:.3f} (require >= 0.6).")

    targ_truth = targ_truth.set_index("MODEL")
    targ_truth = targ_truth.to_dict(orient="index")

    ###################################

    # Index(['MODEL', 'GR.CODE', 'GROUP', 'QSGLOB', 'QSBEST', 'MOL.SIZE', 'STOICH.',
    #        'SYMM.', 'SYMMSIZE', 'ICS(F1)', 'PREC.IFACE', 'RECAL.IFACE', 'LDDT',
    #        'DOCKQ_AVG', 'IPS(JACCCOEF)', 'QS_INTERFACES', 'QSGLOB_PERINTERFACE',
    #        'TMSCORE', 'TARGET'],
    #       dtype='object')

    info = json.load(open(f"{raw}/{target}/info.json", 'r'))

    # if info["expected_num_res"] < 1500: continue


    # Load the predictors from the json file if it exists
    
    predictors = {}

    for tag in os.listdir(f"{raw}/{target}/QA_1/{target.replace('o','')}/"):    # For each Predictors
        t, group = tag.split("QA")
        group = group.split("_")[0]

        assert target.replace('o', '') == t
        if os.path.exists(f"{raw}/{target}/local_preds/{group_mappings[group]}_local.json"):    # If the predictor is already in the json file
            predictors[group_mappings[group]] = json.load(open(f"{raw}/{target}/local_preds/{group_mappings[group]}_local.json",'r'))
            continue

        ###################

        preds = {}
        curr_model = None
        qmode_2 = False
        for ln in open(f"{raw}/{target}/QA_1/{target.replace('o','')}/{tag}", "r"):
            ln = ln.strip()
            if not len(ln): continue
            if ln == "END": break
            if ln == "QMODE 2": qmode_2 = True; continue
            elif ln == "QMODE 1": qmode_2 = False; break
            if not qmode_2: continue

            entries = ln.split()
            if target.replace('o','') in entries[0]: 
                curr_model = entries[0]
                assert curr_model not in preds
                preds[curr_model] = {}
                entries = entries[3:]
            for entry in entries:
                rid, score = entry.split(":")
                preds[curr_model][rid] = float(score)

        if qmode_2:
            os.makedirs(f"{raw}/{target}/local_preds/", exist_ok=True)
            with open(f"{raw}/{target}/local_preds/{group_mappings[group]}_local.json", "w") as json_file:
                json.dump(preds, json_file, indent=4) 

            predictors[group_mappings[group]] = json.load(open(f"{raw}/{target}/local_preds/{group_mappings[group]}_local.json",'r'))

    if not len(predictors):
        log_lines.append("Skipped: no predictor payloads parsed from QA/local_preds (empty predictors dict).")
        return target_results, _proc_stats(
            target_all_models,
            target_filtered_models,
            target_kicked_out_models,
            [],
            log_lines,
        ), {}

    ###################

    for group in ['ARC_GLFP', 'ARC_TransConv', 'ARC_ResGatedGraphConv', 'ARC_GINEConv', 'ARC_GENConv', 'ARC_GeneralConv', 'ARC_PDNConv']:
        loaded = _load_packaged_local(PACKAGED_PREDICTIONS_BASE, target, group)
        if loaded is not None:
            predictors[group] = loaded

    ###################

    log_lines.append(f"Predictor groups available: {len(predictors)} (QA + packaged ARC GNNs).")

    ###################################

    for group in predictors:
        predictions = predictors[group]

        if group == "ARC":
            repl = _load_packaged_local(PACKAGED_PREDICTIONS_BASE, target, "ARC")
            if repl is not None:
                predictions = repl

        if group == "APOLLO":
            repl = _load_packaged_local(PACKAGED_PREDICTIONS_BASE, target, "APOLLO")
            if repl is not None:
                predictions = repl

        if len(predictions)/len(targ_truth) < 0.8: continue # Check if the predictor covers at least 80% of the models in the target

        patch_qs = []
        patch_dockq = []
        true_iface = []
        pred = []
        model_ids = []  # one per residue, for (target, model) filter in pooled analysis

        local_lddt = []
        lddt_pred = []
        model_lddt = []

        local_cad = []
        cad_pred = []
        model_cad = []

        # Track models that contribute to true_interface_residue for this group
        models_used_for_true_iface = set()

        global_r = {
            "pred": [],
            "QSGLOB": [],
            "QSBEST": [],
            "DOCKQ_AVG": [],
            "model": []
        }
        models_checked = 0
        models_filtered = 0
        for model in info.keys():
            if model in ['expected_num_res', 'num_mod', 'models']: continue

            if model not in predictions: continue
            if not len(predictions[model]): continue
            
            models_checked += 1
            # Filter by stoichiometry: skip models that were filtered out at the top
            if model not in filtered_mdls: 
                models_filtered += 1
                continue

            ###################

            # local results
            local_truth = None
            if os.path.exists(os.path.join(CASP16_EMA_RESULTS, f"{model}_{target}.json")):

                local_truth = json.load(open(os.path.join(CASP16_EMA_RESULTS, f"{model}_{target}.json"), 'r'))

                if "model_interface_residues" not in local_truth:
                    continue

                # chain_mapping maps reference chain -> model chain
                # We need model chain -> reference chain, so reverse it
                mapping = {v : k for k, v in local_truth["chain_mapping"].items()}
                
                # Only process models that are in targ_truth (have QS Best scores)
                # This matches the notebook's approach where it filters by global_qs_best
                if model not in targ_truth:
                    continue
                
                # Track all models we attempt to process (only those with valid local_truth)
                target_all_models.add(model)
                
                # Track models that pass filtering (stoichiometry filter already applied above)
                target_filtered_models.add(model)
                
                # Track models used for this group's true_interface_residue computation
                models_used_for_true_iface.add(model)

                for r, residue in enumerate(local_truth["model_interface_residues"]):
                    rkey = "".join(residue.split(".")[:-1])
                    if rkey in predictions[model]: pred += [predictions[model][rkey]]
                    else: pred += [None]

                    patch_qs += [local_truth["patch_qs"][r]]
                    patch_dockq += [local_truth["patch_dockq"][r]]
                    model_ids += [model]

                    # See: https://git.scicore.unibas.ch/schwede/casp16_ema/-/blob/main/analysis/collect_local_data.py?ref_type=heads

                    int_res_data = residue.split('.')
                    assert(len(int_res_data)==3)
                    cname = int_res_data[0]  # model chain name
                    rnum = int(int_res_data[1])
                    ins_code = int_res_data[2]
                    assert(ins_code == "")

                    # Map model chain to reference chain
                    trg_cname = mapping.get(cname)  # Use .get() to handle missing mappings
                    
                    # Check if this residue is a true interface residue
                    # Only check if we have a valid target chain mapping
                    if trg_cname is not None:
                        true_iface += [f"{trg_cname}.{rnum}." in local_truth["reference_interface_residues"]]
                    else:
                        # If model chain doesn't map to reference chain, it can't be a true interface residue
                        true_iface += [False]

                for residue in local_truth["local_cad_score"]:
                    rkey = "".join(residue.split(".")[:-1])
                    if rkey in predictions[model]: cad_pred += [predictions[model][rkey]]
                    else: cad_pred += [None]
                    local_cad += [local_truth["local_cad_score"][residue]]
                    model_cad += [model]

                for residue in local_truth["local_lddt"]:
                    rkey = "".join(residue.split(".")[:-1])
                    if rkey in predictions[model]: lddt_pred += [predictions[model][rkey]]
                    else: lddt_pred += [None]
                    local_lddt += [local_truth["local_lddt"][residue]]
                    model_lddt += [model]

                ###################

                # per-iface predictions (global_interface_res used later in global results)
                iface_predictions = {}
                global_interface_res = []

                if model in targ_truth and local_truth is not None:
                    mod_truth = targ_truth[model]

                    interface_residues = {}
                    global_interface_res = []
                    for contact in local_truth["model_contacts"]: 
                        r_a, r_b = contact
                        chain_a, res_idx_a, _ = r_a.split(".")
                        chain_b, res_idx_b, _ = r_b.split(".")
                        iface = "".join(sorted([chain_a, chain_b]))
                        if chain_a == chain_b: continue # not interface contact

                        if iface not in interface_residues: 
                            interface_residues[iface] = [] # up-count number of residues by number of involved contacts between chains...
                        interface_residues[iface] += [r_a, r_b]
                        global_interface_res += [r_a, r_b]

                    for interface in interface_residues:
                        resscore = [
                            predictions[model][res.replace('.','')]
                            # for res in info[model][interface] if res.replace('.','') in predictions[model]
                            for res in interface_residues[interface] if res.replace('.','') in predictions[model]
                        ]
                        if not len(resscore): continue
                        iface_predictions[interface] = np.mean(np.clip(resscore, a_min = 10**(-16), a_max = None))

                ###################

                # global results
                # Note: We already filtered by targ_truth above, so model should be in targ_truth
                mod_truth = targ_truth[model]
                    
                for key in global_r:
                    if key == "model":
                        global_r["model"] += [model]
                    elif key in mod_truth:
                        global_r[key] += [mod_truth[key]]

                if not len(iface_predictions): global_r["pred"] += [None]
                else: global_r["pred"] += [np.mean([
                    predictions[model][res.replace('.','')]
                    for res in global_interface_res if res.replace('.','') in predictions[model]
                ])]
                

        ###################

        if not len(pred) or not len(cad_pred) or not len(lddt_pred):
            if models_checked > 0:
                log_lines.append(
                    f"  Method {group}: skip (insufficient aligned scores: "
                    f"pred={len(pred)}, cad={len(cad_pred)}, lddt={len(lddt_pred)})."
                )
            continue

        global_r = pd.DataFrame(global_r)
        loc_df = pd.DataFrame({
            "patch_qs": patch_qs,
            "patch_dockq": patch_dockq,
            "true_interface_residue": true_iface,
            "pred": pred,
            "model": model_ids
        })

        loc_cad = pd.DataFrame({
            "local_cad": local_cad,
            "pred": cad_pred,
            "model": model_cad
        })

        loc_lddt = pd.DataFrame({
            "local_lddt": local_lddt,
            "pred": lddt_pred,
            "model": model_lddt
        })

        # Store raw dataframes for pooling (with target identifier)
        if group not in target_raw_data:
            target_raw_data[group] = {}
        
        # Add target column to all dataframes for later filtering
        loc_df_with_target = loc_df.copy()
        loc_df_with_target["target"] = target
        loc_cad_with_target = loc_cad.copy()
        loc_cad_with_target["target"] = target
        loc_lddt_with_target = loc_lddt.copy()
        loc_lddt_with_target["target"] = target
        global_r_with_target = global_r.copy()
        global_r_with_target["target"] = target
        
        # Store raw data for pooling (include "model" for (target, model) filter)
        target_raw_data[group]["patch_qs"] = loc_df_with_target[["patch_qs", "pred", "target", "model"]].copy()
        target_raw_data[group]["patch_dockq"] = loc_df_with_target[["patch_dockq", "pred", "target", "model"]].copy()
        target_raw_data[group]["local_cad"] = loc_cad_with_target[["local_cad", "pred", "target", "model"]].copy()
        target_raw_data[group]["local_lddt"] = loc_lddt_with_target[["local_lddt", "pred", "target", "model"]].copy()
        target_raw_data[group]["true_interface_residue"] = loc_df_with_target[["true_interface_residue", "pred", "target", "model"]].copy()
        target_raw_data[group]["QSGLOB"] = global_r_with_target[["QSGLOB", "pred", "target", "model"]].copy()
        target_raw_data[group]["QSBEST"] = global_r_with_target[["QSBEST", "pred", "target", "model"]].copy()
        target_raw_data[group]["DOCKQ_AVG"] = global_r_with_target[["DOCKQ_AVG", "pred", "target", "model"]].copy()

        ###################

        results = [
            ("patch_dockq", compute_rs(loc_df, "patch_dockq")),
            ("patch_qs", compute_rs(loc_df, "patch_qs")),
            ("local_cad", compute_rs(loc_cad, "local_cad")),
            ("local_lddt", compute_rs(loc_lddt, "local_lddt")),
            ("QSGLOB", compute_rs(global_r, "QSGLOB")),
            ("QSBEST", compute_rs(global_r, "QSBEST")),
            ("DOCKQ_AVG", compute_rs(global_r, "DOCKQ_AVG")),
        ]

        for (score_type, mets) in results:
            for i, met_type in enumerate(["pearson", "spearman", "adaptive_rocauc"]):
                target_results["method"] += [group]
                target_results["target"] += [target]
                target_results["score"] += [score_type]
                target_results["metric"] += [met_type]
                target_results["value"] += [mets[i]]

        opt_row = global_r.loc[global_r['pred'].idxmax()]

        for score_type in ["QSGLOB", "QSBEST", "DOCKQ_AVG"]:
            target_results["method"] += [group]
            target_results["target"] += [target]
            target_results["score"] += [score_type]
            target_results["metric"] += ["RL"]
            target_results["value"] += [global_r[score_type].max() - opt_row[score_type]]     

        ###################

        # For aucroc/aucprc, match notebook's approach:
        # 1. Filter out null predictions (residue-level filtering)
        sub_df = loc_df[loc_df["pred"].isnull()==False]
        
        # 2. Check 80% coverage at residue level (matching notebook's n_pred/n_exp >= 0.8 check)
        # Note: This is stricter than the model-level check above. The notebook requires 80% residue-level
        # coverage before computing metrics, which means some methods/targets that pass model-level
        # coverage might still be skipped here if they don't have predictions for enough residues.
        n_exp = len(loc_df)  # Total expected data points (all residues)
        n_pred = len(sub_df)  # Non-null predictions
        
        # Only compute metrics if we have sufficient coverage (matching notebook logic)
        if len(sub_df) > 0 and float(n_pred)/n_exp >= 0.8:
            # Ensure we have both classes (true and false) for meaningful metrics
            if sub_df["true_interface_residue"].nunique() >= 2:
                precision, recall, thresholds = precision_recall_curve(sub_df["true_interface_residue"], sub_df["pred"])
                
                roc_auc_val = roc_auc_score(sub_df["true_interface_residue"], sub_df["pred"])
                pr_auc_val = auc(recall, precision)
                
                # Track statistics for comparison
                n_false = sub_df[sub_df["true_interface_residue"]==False].shape[0]
                n_true = sub_df[sub_df["true_interface_residue"]==True].shape[0]
                
                # Count models from dataframe that have predictions for this group (matching notebook approach)
                # The notebook's dataframe includes ALL models with JSON files, and predictions are added as columns
                # We need to count unique models from df_filtered that have non-null predictions for this group
                # Find the group ID (column name) that corresponds to this group name
                group_id = None
                for gid, gname in group_mappings.items():
                    if gname == group:
                        group_id = gid
                        break
                
                # df_filtered already contains rows for this target (filtered by stoichiometry)
                # The dataframe includes ALL models with JSON files (from collect_local_data.py)
                # Predictions are added as columns (None if missing) in collect_ema_data_local.py
                # We count unique models from df_filtered that have non-null predictions for this group
                if group_id and group_id in df_filtered.columns:
                    # Count models from filtered dataframe that have non-null predictions for this group
                    # This matches the notebook's approach: count unique models with predictions
                    # df_filtered already has the correct target and stoichiometry filter applied
                    target_df_group_with_pred = df_filtered[df_filtered[group_id].notna()]
                    n_models_used = len(target_df_group_with_pred["mdl"].unique())
                else:
                    # Fallback: count models that actually contributed to computation
                    # (group_id is missing for derived/auxiliary methods e.g. individual ARC_* GNNs)
                    n_models_used = len(models_used_for_true_iface)
                
                true_iface_stats.append({
                    "target": target,
                    "group": group,
                    "n_exp": n_exp,
                    "n_pred": n_pred,
                    "n_false": n_false,
                    "n_true": n_true,
                    "n_models": n_models_used,
                    "roc_auc": roc_auc_val,
                    "pr_auc": pr_auc_val,
                    "coverage": float(n_pred)/n_exp
                })
                
                target_results["method"] += [group]*2
                target_results["target"] += [target]*2
                target_results["score"] += ["true_interface_residue"]*2
                target_results["metric"] += ["aucroc", "aucprc"]
                
                target_results["value"] += [
                    roc_auc_val,
                    pr_auc_val
                ]

    log_lines.append(
        f"Done: {len(target_results['method'])} metric rows; "
        f"{len(true_iface_stats)} true-interface stat record(s)."
    )

    return target_results, _proc_stats(
        target_all_models,
        target_filtered_models,
        target_kicked_out_models,
        true_iface_stats,
        log_lines,
    ), target_raw_data
