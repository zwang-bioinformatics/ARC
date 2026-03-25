"""Static mappings, ARC GNN list, target list, paths, and bundled analysis data (e.g. APPROX_TARGET_SIZE)."""
import os

from casp16_eval_io import load_target_sizes
from casp16_eval_paths import (
    ARC_PREDICTIONS_CASP16,
    CASP16_APPROX_TARGET_SIZES_JSON,
    CASP16_EMA_REFERENCE_RESULTS_DIR,
)

group_mappings = {
  "000": "AssemblyConsensus",
  "002": "JFK-THG-AMBER",
  "003": "JFK-THG-AMBERstable",
  "004": "JFK-THG-CHARMM",
  "005": "JFK-THG-CHARMMstable",
  "006": "RNA_Dojo",
  "008": "HADDOCK",
  "014": "Cool-PSP",
  "015": "PEZYFoldings",
  "016": "haiping",
  "017": "Seder2024hard",
  "018": "AttStructureScorer",
  "019": "Zheng-Server",
  "020": "comppharmunibas",
  "022": "Yang",
  "023": "FTBiot0119",
  "026": "SwRI",
  "027": "ModFOLDdock2R",
  "028": "NKRNA-s",
  "029": "zyh_mae_try1",
  "030": "SNU-CHEM_aff",
  "031": "MassiveFold",
  "032": "Bryant",
  "033": "Diff",
  "039": "arosko",
  "040": "DELCLAB",
  "049": "UTMB",
  "050": "SHORTLE",
  "051": "MULTICOM",
  "052": "Yang-Server",
  "055": "LCDD-team",
  "059": "DeepFold",
  "063": "RNApolis",
  "074": "ModFOLDdock2S",
  "075": "GHZ-ISM",
  "077": "coogs2",
  "079": "MRAFold",
  "080": "pDockQ",
  "082": "VnsDock",
  "084": "Vendruscolo",
  "085": "Bates",
  "088": "orangeballs",
  "091": "Huang-HUST",
  "092": "Seamount",
  "094": "SimRNA-server",
  "097": "JFK-THG-IDPCONFGEN",
  "100": "zurite_lab",
  "102": "Psi-Phi",
  "105": "PFSC-PFVM",
  "110": "MIEnsembles-Server",
  "112": "Seder2024easy",
  "114": "COAST",
  "117": "Vakser",
  "120": "Cerebra",
  "121": "Pascal_Auffinger",
  "122": "MQA_server",
  "128": "TheMeilerMethod",
  "132": "profold2",
  "135": "Lindorff-LarsenCLVDS",
  "136": "Lindorff-LarsenM3PPS",
  "137": "Lindorff-LarsenM3PWS",
  "138": "Shengyi",
  "139": "DeepFold-refine",
  "143": "dMNAfold",
  "145": "colabfold_baseline",
  "147": "Zheng-Multimer",
  "148": "Guijunlab-Complex",
  "156": "SoutheRNA",
  "159": "406",
  "163": "MultiFOLD2",
  "164": "McGuffin",
  "165": "dfr",
  "167": "OpenComplex",
  "169": "thermomaps",
  "171": "ChaePred",
  "172": "VoroAffinity",
  "174": "colabfold_foldseek",
  "177": "aicb",
  "183": "GuangzhouRNA-human",
  "187": "Ayush",
  "188": "VifChartreuseJaune",
  "189": "LCBio",
  "191": "Schneidman",
  "196": "HYU_MLLAB",
  "197": "D3D",
  "198": "colabfold",
  "201": "Drugit",
  "202": "test001",
  "204": "Zou",
  "207": "MULTICOM_ligand",
  "208": "falcon2",
  "209": "colabfold_human",
  "212": "PIEFold_human",
  "217": "zyh_mae_try1E",
  "218": "HIT-LinYang",
  "219": "XGroup-server",
  "221": "CSSB_FAKER",
  "226": "Pfaender",
  "227": "KUMC",
  "231": "B-LAB",
  "235": "isyslab-hust",
  "237": "Convex-PL-R",
  "238": "BRIQX",
  "241": "elofsson",
  "261": "UNRES",
  "262": "CoDock",
  "264": "GuijunLab-Human",
  "267": "kiharalab_server",
  "269": "CSSB_server",
  "271": "mialab_prediction2",
  "272": "GromihaLab",
  "273": "MQA_base",
  "274": "kozakovvajda",
  "275": "Seminoles",
  "276": "FrederickFolding",
  "281": "T2DUCC",
  "284": "Unicorn",
  "286": "CSSB_experimental",
  "287": "plmfold",
  "290": "Pierce",
  "293": "MRAH",
  "294": "KiharaLab",
  "295": "VoroAffinityB",
  "298": "ShanghaiTech-human",
  "300": "ARC",
  "301": "GHZ-MAN",
  "304": "AF3-server",
  "306": "GeneSilicoRNA-server",
  "307": "nfRNA",
  "308": "MoMAteam1",
  "309": "Koes",
  "311": "RAGfold_Prot1",
  "312": "GuijunLab-Assembly",
  "314": "GuijunLab-PAthreader",
  "317": "GuangzhouRNA_AI",
  "319": "MULTICOM_LLM",
  "322": "XGroup",
  "323": "Yan",
  "325": "405",
  "331": "MULTICOM_AI",
  "337": "APOLLO",
  "338": "GeneSilico",
  "345": "MULTICOM_human",
  "349": "cheatham-lab",
  "351": "digiwiser-ensemble",
  "353": "KORP-PL-W",
  "355": "CMOD",
  "357": "UTAustin",
  "358": "PerezLab_Gators",
  "361": "Cerebra_server",
  "363": "2Vinardo",
  "367": "AIR",
  "369": "Bhattacharya",
  "370": "DrAshokAndFriends",
  "375": "milliseconds",
  "376": "OFsingleseq",
  "380": "mialab_prediction",
  "384": "pert-plddt",
  "386": "ShanghaiTech-Ligand",
  "388": "DeepFold-server",
  "391": "bussilab_replex",
  "393": "GuijunLab-QA",
  "397": "smg_ulaval",
  "399": "NEC_Compute",
  "400": "OmniFold",
  "403": "mmagnus",
  "408": "SNU-CHEM-lig",
  "412": "cheatham-lab_villa",
  "416": "GPLAffinity",
  "417": "GuangzhouRNA-meta",
  "418": "Lee-Shin",
  "419": "CSSB-Human",
  "420": "Zou_aff2",
  "423": "ShanghaiTech-server",
  "425": "MULTICOM_GATE",
  "432": "DIMAIO",
  "435": "RNAFOLDX",
  "436": "Yoshiaki",
  "439": "Dokholyan",
  "441": "ModFOLDdock2",
  "443": "MQA",
  "446": "pDockQ2",
  "447": "UDMod",
  "448": "dNAfold",
  "450": "OpenComplex_Server",
  "456": "Yang-Multimer",
  "461": "forlilab",
  "462": "Zheng",
  "464": "PocketTracer",
  "465": "Wallner",
  "466": "coogs3",
  "468": "MIALAB_gong",
  "469": "GruLab",
  "471": "Pcons",
  "474": "CCB-AlGDock",
  "475": "ptq",
  "476": "VifChartreuse",
  "479": "DoraemonXia",
  "481": "Vfold",
  "485": "bussilab_plain_md",
  "489": "Fernandez-Recio",
  "494": "ClusPro",
  "496": "AF_unmasked",
  "501": "Baseline_pLDDT",
  "502": "Baseline_pTM",
  "510": "Baseline_pTMo",
  "511": "Baseline_IpTM",
  "512": "Baseline_IpTM_PTM",
  "513": "Baseline_mean_pLDDT"
}

# ==============================================================================
# Eval constants: ARC GNN list, paths, stratification helpers
# ==============================================================================

# Fixed order of 7 individual ARC GNNs (index K = 0..6)
ARC_GNNS = [
    'ARC_GLFP', 'ARC_TransConv', 'ARC_ResGatedGraphConv', 'ARC_GINEConv',
    'ARC_GENConv', 'ARC_GeneralConv', 'ARC_PDNConv'
]
PACKAGED_PREDICTIONS_BASE = ARC_PREDICTIONS_CASP16

# Per-model assessor JSONs ({model}_{target}.json); symlink from casp16_ema results - see data/README.md
CASP16_EMA_RESULTS = CASP16_EMA_REFERENCE_RESULTS_DIR

# Approximate target sizes for size-based stratification (Small/Medium/Large/Huge buckets).
# Canonical values + _TBD_provenance: data/casp16_approx_target_sizes.json (edit there; not duplicated as literals).
APPROX_TARGET_SIZE = load_target_sizes(CASP16_APPROX_TARGET_SIZES_JSON)

targets = [
    'H1204', 'T1270o', 'H1236', 'T1249v2o', 'H1217', 'H1258', 'H1225', 
    'H1222', 'T1246o', 'H1267', 'T1259o', 'T1234o', 'T1257o', 'T1292o', 
    'T1240o', 'T1279o', 'T1228v2o', 'T1269v1o', 'T1219v1o', 'T1266o', 
    'T1294v2o', 'H1213', 'T1235o', 'H1232', 'H1245', 'T1295o', 'H1230', 
    'H1208', 'H1202', 'T1218o', 'H1223', 'H1229', 'T1294v1o', 'T1219v2o', 
    'H1272', 'T1228v1o', 'H1244', 'H1233', 'T1237o', 'T1219o', 'H1220', 
    'T1206o', 'H1265', 'H1215', 'T1201o', 'T1238o', 'H1227', 'T1249v1o', 
    'T1239v1o', 'T1298o'
]

# Parallel target processing in main pipeline (set ARC_USE_PARALLEL=0 for sequential fallback).
USE_PARALLEL = os.environ.get("ARC_USE_PARALLEL", "1").strip().lower() in ("1", "true", "yes")

# Methods removed before z-score / stratification (missing preds, consensus, auxiliary GNNs)
EXCLUDED_LOCAL_EVAL_METHODS = frozenset(
    {
        "ChaePred",
        "AF_unmasked",
        "MQA",
        "ModFOLDdock2",
        "ModFOLDdock2R",
        "GuijunLab-QA",
        "GuijunLab-Human",
        *ARC_GNNS,
    }
)

# Stratification labels (e.g. pooled plot titles, ensemble tables)
STRAT_DISPLAY = {
    "Dimer only": "Dimers",
    "Multimer only": r"Multimers (n $\geq$ 3)",
    "Dimer_only": "Dimers",
    "Multimer_only": r"Multimers (n $\geq$ 3)",
}

# Log banners for casp16_eval.py pipeline (no numeric section prefixes).
EVAL_LOG_SEC_1 = "Per-target evaluation"
EVAL_LOG_SEC_2 = "ARC ensemble analysis"
EVAL_LOG_SEC_2_2 = "Residue variance / mean CSV"
EVAL_LOG_SEC_2_3 = "All-combinations run (ensemble vs individual, per target)"
EVAL_LOG_SEC_2_4 = "All-combinations summary (per target, L; BH-corrected DeLong)"
EVAL_LOG_SEC_2_5 = "Manuscript highlights (L=6; main text Section 3.3)"
EVAL_LOG_SEC_2_6 = "Wilcoxon tests (ensemble vs individual, BH by L)"
EVAL_LOG_SEC_2_7 = "Ensemble summary CSV (arc_ensemble_summary_all_comb.csv)"
EVAL_LOG_SEC_3 = "Pooled analysis (ROC curves, performance tables)"
EVAL_LOG_SEC_3B = "Pooled log summary (supplementary Table S2 layout)"
EVAL_LOG_SEC_3C = "ARC rankings summary (all stratifications)"
EVAL_LOG_SEC_4 = "CASP-style rank tables (writes supplementary Table S1 CSVs)"
