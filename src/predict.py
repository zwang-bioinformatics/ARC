"""Batch quality inference for CASP16 geometric graphs (ARC ensemble).

Loads seven trained QA checkpoints (GLFP, TransformerConv, ResGatedGraphConv,
GINEConv, GENConv, GeneralConv, PDNConv) from ``ARC/models/<run>/``,
runs each on every model structure under ``ARC/data/<collection>/<target>/``, and
writes JSON predictions in two tiers:

**Per-base predictors** - one subdirectory per training run (folder basename), e.g.
``outputs/predictions/<collection>/<target>/ARC_GENConv/``. Optional via
``--no-save-base-predictors`` (default is to save them).

**Ensemble** - mean of the seven networks' per-node scores, then the same
scoring rules; written to ``outputs/predictions/<collection>/<target>/ARC/``.
Optional ``--outdir`` duplicates only the ensemble JSON to another folder.

**JSON layout** (both ``LOCAL.json`` and ``QSCORE.json``):

- ``QSCORE.json``: map **CASP model name** -> single float (mean predicted score
  over interface nodes).
- ``LOCAL.json``: map **CASP model name** -> map **residue UUID (str)** ->
  per-residue score (interface nodes only).

**CLI** (typical)::

    python predict.py -t H1202 --collection CASP16

``-t`` names a folder under ``data/<collection>/<target>/``. ``--collection``
defaults to ``CASP16``; use another name for a parallel tree (same layout:
``meta.json`` + ``data.st`` per model). Outputs go under
``outputs/predictions/<collection>/<target>/`` (ensemble in ``ARC/``, each
checkpoint under its run label). Optional ``-o`` copies ensemble JSON only to
an extra path.

**Run context:** execute from ``ARC/src`` (or set ``PYTHONPATH`` so
``models`` resolves). Requires ``torch``, PyG, and dataset definitions under
``src/data_scripts/``.
"""

import argparse
import json
import os
import sys

parser = argparse.ArgumentParser()

parser.add_argument("-t", "--target", action="store", type=str, required=True)
parser.add_argument(
	"--collection",
	action="store",
	type=str,
	default="CASP16",
	help="Folder under data/<collection>/<target>/ (inputs) and outputs/predictions/<collection>/<target>/ (writes).",
)
parser.add_argument(
	"-o",
	"--outdir",
	action="store",
	type=str,
	default=None,
	help="Optional extra directory for ensemble LOCAL.json / QSCORE.json (canonical tree is always written).",
)
_save_group = parser.add_mutually_exclusive_group()
_save_group.add_argument(
	"--save-base-predictors",
	dest="save_base_predictors",
	action="store_true",
	help="Write per-base predictors JSONs under outputs/predictions/<collection>/<target>/<run_label>/ (default).",
)
_save_group.add_argument(
	"--no-save-base-predictors",
	dest="save_base_predictors",
	action="store_false",
	help="Only write ensemble under outputs/predictions/<collection>/<target>/ARC/ (and -o if set).",
)
parser.set_defaults(save_base_predictors=True)

args = parser.parse_args()

# Repository root: parent of this file's directory (.../ARC).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# --- CLI summary ---
print("\n" + "*" * 15, "\n")

print("Target:", args.target, "\n")
print("Collection:", args.collection, "\n")
print("Extra out directory (-o):", args.outdir, "\n")
print("Save base predictors:", args.save_base_predictors, "\n")

print("*" * 15, "\n")

# --- Dataset (graphs from ARC/data/<collection>/<target>/)
if args.outdir is not None:
	os.makedirs(args.outdir, exist_ok=True)

sys.path.append(os.path.join(_REPO_ROOT, "src", "data_scripts"))

import torch
from geometric_dataset import *

data_set = CASP16GeometricDataset(
	target=args.target,
	collection=args.collection,
	params={"type": "comprehensive_CASP16", "include": ["is_interface"]},
)

input_dimensions = data_set.get_dimensions()

import numpy as np
from tqdm import tqdm

# --- Device (CPU inference)
device = "cpu"
print(f"Device: {device}\n")
print("*" * 15, "\n")

from models import *

models_path = [
	os.path.join(_REPO_ROOT, "models", "ARC_GLFP"),
	os.path.join(_REPO_ROOT, "models", "ARC_TransConv"),
	os.path.join(_REPO_ROOT, "models", "ARC_ResGatedGraphConv"),
	os.path.join(_REPO_ROOT, "models", "ARC_GINEConv"),
	os.path.join(_REPO_ROOT, "models", "ARC_GENConv"),
	os.path.join(_REPO_ROOT, "models", "ARC_GeneralConv"),
	os.path.join(_REPO_ROOT, "models", "ARC_PDNConv"),
]

# Inference-only model manifest (folder name -> architecture).
_MODEL_ARCH = {
	"ARC_GLFP": "GLFP",
	"ARC_TransConv": "TransformerConv",
	"ARC_ResGatedGraphConv": "ResGatedGraphConv",
	"ARC_GINEConv": "GINEConv",
	"ARC_GENConv": "GENConv",
	"ARC_GeneralConv": "GeneralConv",
	"ARC_PDNConv": "PDNConv",
}

# (subdir_label e.g. ARC_GENConv, nn.Module)
model_entries = []
for mroot in models_path:
	label = os.path.basename(os.path.normpath(mroot))
	arch_name = _MODEL_ARCH[label]
	ckpt = os.path.join(mroot, "model.pt")
	nn_model = create_model(
		model_name=arch_name,
		config={
			"node_dim": input_dimensions["node_dim"],
			"edge_dim": input_dimensions["edge_dim"],
			"device": device,
		},
	).double().to(device)
	nn_model.load_state_dict(torch.load(ckpt, map_location=device)["model_state_dict"])
	nn_model.eval()
	model_entries.append((label, nn_model))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _per_node_scores(t_out):
	"""Normalize network output to a 1D tensor with one entry per graph node.

	Last-layer convolutions often return shape ``[N, 1]``; ``squeeze()`` removes
	all size-1 dimensions, which turns a **single-node** graph into a 0-D scalar
	and breaks indexing. This always returns ``shape == (N,)``.

	Args:
		t_out: Raw forward output from a QA model.

	Returns:
		``torch.Tensor`` of shape ``(num_nodes,)``.
	"""
	t = t_out.squeeze()
	if t.dim() == 0:
		t = t.unsqueeze(0)
	return t.reshape(-1)


def predictions_for_example(example, out_vec):
	"""Turn per-node scores into global and per-residue predictions for one structure.

	Does not mutate ``example`` or any aggregate dicts. ``example.model`` is the
	CASP decoy/model name; scores are averaged over nodes marked interface in
	``example.is_interface``.

	Args:
		example: PyG ``Data`` with ``model``, ``is_interface``, ``r_uuid``.
		out_vec: 1D tensor, one scalar score per node (same length as num nodes).

	Returns:
		Tuple ``(casp_model_id, qscore, local)`` where:

		- ``casp_model_id``: ``example.model`` (str).
		- ``qscore``: mean of ``out_vec`` over interface nodes (float).
		- ``local``: dict mapping residue UUID string -> score for each interface node.

	Raises:
		AssertionError: duplicate residue UUID for the same structure.
	"""
	mask = example.is_interface == 1
	mid = example.model
	qscore = out_vec[mask].mean().item()
	local = {}
	for i in range(out_vec.shape[0]):
		if not example.is_interface[i]:
			continue
		r_key = example.r_uuid[i]
		if hasattr(r_key, "item"):
			r_key = r_key.item()
		r_key = str(r_key)
		assert r_key not in local, "ERROR: duplicate residue keys! - " + r_key
		local[r_key] = out_vec[i].item()
	return mid, qscore, local


# --- Accumulators (filled in the inference loop)
QSCORE_PREDS = {}
LOCAL_PREDS = {}
per_q = {label: {} for label, _ in model_entries}
per_l = {label: {} for label, _ in model_entries}

# --- Inference: per-base predictors scores + ensemble mean per node
with torch.no_grad():

	for example in tqdm(data_set, desc="Inferring Qualities"):
		if example.is_interface.sum() == 0:
			continue
		example.batch_idx = torch.zeros(example.node_features.shape[0], dtype=torch.long)
		example.edge_index = example.edge_index.transpose(0, 1)

		outs = []
		for label, nn_model in model_entries:
			t_raw = nn_model(
				example.node_features,
				example.edge_index,
				example.edge_features,
				example.batch_idx,
			)
			t_out = _per_node_scores(t_raw)
			n_nodes = example.node_features.shape[0]
			assert t_out.shape[0] == n_nodes, (label, t_out.shape, n_nodes)
			outs.append(t_out)
			assert example.model not in per_q[label], "ERROR: duplicate model names!"
			assert example.model not in per_l[label], "ERROR: duplicate model names!"
			mid, q, loc = predictions_for_example(example, t_out)
			per_q[label][mid] = q
			per_l[label][mid] = loc

		node_count = example.node_features.shape[0]
		consensus_out = [
			float(np.mean([outs[j][i].item() for j in range(len(outs))]))
			for i in range(node_count)
		]
		out = torch.tensor(consensus_out, dtype=torch.double)

		assert example.model not in QSCORE_PREDS, "ERROR: duplicate model names!"
		assert example.model not in LOCAL_PREDS, "ERROR: duplicate model names!"
		mid, q, loc = predictions_for_example(example, out)
		QSCORE_PREDS[mid] = q
		LOCAL_PREDS[mid] = loc

# ---------------------------------------------------------------------------
# Write JSON artifacts
# ---------------------------------------------------------------------------


def save_predictions(out_dir, local_preds, qscore_preds):
	"""Write ``LOCAL.json`` and ``QSCORE.json`` under ``out_dir`` (creates directory)."""
	os.makedirs(out_dir, exist_ok=True)
	with open(os.path.join(out_dir, "LOCAL.json"), "w") as f:
		json.dump(local_preds, f, indent=4)
	with open(os.path.join(out_dir, "QSCORE.json"), "w") as f:
		json.dump(qscore_preds, f, indent=4)


# Canonical layout: ``outputs/predictions/<collection>/<target>/ARC/`` (+ per-base predictors subdirs).
structured_base = os.path.join(
	_REPO_ROOT, "outputs", "predictions", args.collection, args.target
)
save_predictions(os.path.join(structured_base, "ARC"), LOCAL_PREDS, QSCORE_PREDS)
if args.outdir is not None:
	save_predictions(args.outdir, LOCAL_PREDS, QSCORE_PREDS)
if args.save_base_predictors:
	for label, _ in model_entries:
		save_predictions(os.path.join(structured_base, label), per_l[label], per_q[label])

# --- Done
print("\n" + "*" * 15, "\n")

if args.outdir is not None:
	print("Saved (ensemble -o):", os.path.join(args.outdir, "LOCAL.json"), "\n")
	print("Saved (ensemble -o):", os.path.join(args.outdir, "QSCORE.json"), "\n")
print("Saved (ensemble):", os.path.join(structured_base, "ARC"), "\n")
if args.save_base_predictors:
	print("Per-base predictors:", structured_base, "\n")
else:
	print("Per-base predictors: skipped (--no-save-base-predictors)\n")

print("*" * 15, "\n")

print("Done!")

print("\n" + "*" * 15 + "\n")
