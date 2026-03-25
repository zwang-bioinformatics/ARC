"""Drive batch ``predict.py`` over targets under ``ARC/data/<collection>/``.

``predict.py`` loads the seven QA checkpoints from ``ARC/models/*/``,
runs ensemble inference on every graph example for each target, and writes
under ``outputs/predictions/<collection>/<target>/`` (ensemble in ``ARC/``,
per-base predictors subfolders when enabled).

Discovers targets the same way as ``CASP16GeometricDataset`` (subdirs with
``meta.json`` and ``data.st``), skips targets that already have
``outputs/predictions/<collection>/<target>/ARC/QSCORE.json``, and stores one
log per target under ``outputs/logs/<collection>/``.
"""

import argparse
import os
import subprocess
import sys

from tqdm import tqdm

_ARC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
predict_py = os.path.join(_ARC_ROOT, "src", "predict.py")
_predict_cwd = os.path.join(_ARC_ROOT, "src")


def _targets_with_graphs(collection):
	"""Return ``(target_name, n_models)`` for each folder under data/<collection>/ that has usable graphs."""
	root = os.path.join(_ARC_ROOT, "data", collection)
	out = []
	if not os.path.isdir(root):
		return out
	for name in sorted(os.listdir(root)):
		t_path = os.path.join(root, name)
		if not os.path.isdir(t_path):
			continue
		n_models = 0
		for model in os.listdir(t_path):
			m_dir = os.path.join(t_path, model)
			if os.path.isfile(os.path.join(m_dir, "meta.json")) and os.path.isfile(
				os.path.join(m_dir, "data.st")
			):
				n_models += 1
		if n_models:
			out.append((name, n_models))
	return out


def main():
	parser = argparse.ArgumentParser(
		description="Run predict.py for every target under data/<collection>/."
	)
	parser.add_argument(
		"--collection",
		default="CASP16",
		help="Folder under data/ and outputs/predictions/ (default: CASP16).",
	)
	args = parser.parse_args()
	collection = args.collection
	log_root = os.path.join(_ARC_ROOT, "outputs", "logs", collection)

	os.makedirs(log_root, exist_ok=True)
	entries = _targets_with_graphs(collection)
	entries.sort(key=lambda x: x[1])

	for target_name, num_models in tqdm(entries, desc=f"{collection} targets"):
		ensemble_dir = os.path.join(
			_ARC_ROOT, "outputs", "predictions", collection, target_name, "ARC"
		)
		if os.path.isfile(os.path.join(ensemble_dir, "QSCORE.json")):
			continue

		# print(target_name, num_models)

		cmd = [
			sys.executable,
			"-u",
			predict_py,
			"-t",
			target_name,
			"--collection",
			collection,
		]
		# print(" ".join(cmd), "\n")

		log_path = os.path.join(log_root, f"{target_name}.log")
		with open(log_path, "w") as log_f:
			ret = subprocess.run(
				cmd,
				cwd=_predict_cwd,
				stdout=log_f,
				stderr=subprocess.STDOUT,
				check=False,
			)
		print(log_path, "exit", ret.returncode, "\n")

	print("\n" + "*" * 15, "\n")
	print("Done")
	print("\n" + "*" * 15 + "\n")


if __name__ == "__main__":
	main()
