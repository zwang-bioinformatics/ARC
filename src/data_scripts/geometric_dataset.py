import os
import json
from definitions import *

# ARC repository root (this file: ARC/src/data_scripts/geometric_dataset.py)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_NORM_STATS_PATH = os.path.join(_REPO_ROOT, "src", "data_scripts", "norm_stats.json")

import torch
from safetensors import safe_open
from torch_geometric.data import Dataset, Data

torch.set_num_threads(8)

###################################

def residue_embedding_normfunc(res_emb): return torch.tanh((res_emb - torch.mean(res_emb,dim=0)) / torch.std(res_emb,dim=0))

###################################

class CASP16GeometricDataset(Dataset):
	def __init__(self, target=None, params=None, collection="CASP16"):
		
		#### Checks ####

		assert target is not None, "target is required!"
		assert params is not None, "params is required!"
		# Graphs live under data/<collection>/{target}/{model}/

		assert os.path.exists(_NORM_STATS_PATH), "norm stats not found: " + _NORM_STATS_PATH

		#### Inits ####

		self.params = params
		self.norm_stats = json.load(open(_NORM_STATS_PATH, "r"))

		self.example_map = {}

		self.target = target
		self.collection = collection
		self.targ_src = os.path.join(_REPO_ROOT, "data", self.collection, self.target)

		assert os.path.exists(self.targ_src), "ERROR: target not found or not formatted properly: " + self.targ_src 

		for model in os.listdir(self.targ_src): 
			if os.path.exists(os.path.join(self.targ_src, model, "meta.json")) and os.path.exists(os.path.join(self.targ_src, model, "data.st")): 
				self.example_map[str(len(self.example_map))] = model

		super().__init__()

	def get_dimensions(self):
		return {
			"node_dim": sum(map(lambda node_feat: FEATURE_DIMS[node_feat], DATASET_TYPES[self.params["type"]]["node_features"])),
			"edge_dim": sum(map(lambda edge_feat: FEATURE_DIMS[edge_feat], DATASET_TYPES[self.params["type"]]["edge_features"]))
		}

	def len(self): return len(self.example_map)

	def get(self, idx): 

		example = Data()

		with safe_open(os.path.join(self.targ_src, self.example_map[str(idx)], "data.st"), framework="pt", device="cpu") as raw_example:

			#### Node Features ####

			for nfeat_key in DATASET_TYPES[self.params["type"]]["node_features"]:

				if nfeat_key in ANGLE_FEATS: 
					
					rad_tensor = torch.unsqueeze(torch.deg2rad(raw_example.get_tensor(nfeat_key)),dim=1)

					assert len(rad_tensor.shape) == 2, (rad_tensor.shape,nfeat_key, self.example_map[str(idx)])

					if "node_features" not in example: example["node_features"] = torch.sin(rad_tensor)
					else: example["node_features"] = torch.cat((example["node_features"],torch.sin(rad_tensor)),dim=1)

					example["node_features"] = torch.cat((example["node_features"],torch.cos(rad_tensor)),dim=1)

					del rad_tensor

				else: 

					nfeat = raw_example.get_tensor(nfeat_key)

					if nfeat_key in self.norm_stats: 
						assert self.norm_stats[nfeat_key]["sigma"] != 0, ("ZERO STANDARD DEVIATION!",nfeat_key)

						nfeat = torch.tanh((nfeat - self.norm_stats[nfeat_key]["mu"]) / self.norm_stats[nfeat_key]["sigma"])

					if len(nfeat.shape) == 1: nfeat = torch.unsqueeze(nfeat,dim=1)
					if "node_features" not in example: example["node_features"] = nfeat
					else: example["node_features"] = torch.cat((example["node_features"],nfeat),dim=1)

					del nfeat

			#### Edge Features ####

			example["edge_index"] = raw_example.get_tensor("edge_index")

			for efeat_key in DATASET_TYPES[self.params["type"]]["edge_features"]:

				efeat = raw_example.get_tensor(efeat_key)

				if efeat_key in ANGLE_FEATS: 
					
					rad_tensor = torch.unsqueeze(raw_example.get_tensor(efeat_key),dim=1)

					if efeat_key in ["alphafold_angle", "esmfold_angle", "ref_angle_monomeric"]: 
						neg_filter = rad_tensor == -1

						rad_tensor = torch.deg2rad(rad_tensor)

						sin_tensor = (torch.sin(rad_tensor) + 1) / 2
						cos_tensor = (torch.cos(rad_tensor) + 1) / 2

						if rad_tensor[neg_filter].shape[0] != 0:
							sin_tensor[neg_filter] = -1
							cos_tensor[neg_filter] = -1

						if "edge_features" not in example: example["edge_features"] = sin_tensor
						else: example["edge_features"] = torch.cat((example["edge_features"],sin_tensor),dim=1)

						example["edge_features"] = torch.cat((example["edge_features"],cos_tensor),dim=1)

						del sin_tensor
						del cos_tensor

					else: 

						rad_tensor = torch.deg2rad(rad_tensor)

						assert len(rad_tensor.shape) == 2, (rad_tensor.shape,efeat_key, self.example_map[str(idx)])

						if "edge_features" not in example: example["edge_features"] = torch.sin(rad_tensor)
						else: example["edge_features"] = torch.cat((example["edge_features"],torch.sin(rad_tensor)),dim=1)

						example["edge_features"] = torch.cat((example["edge_features"],torch.cos(rad_tensor)),dim=1)

					del rad_tensor

				else: 

					if efeat_key in self.norm_stats: 
						assert self.norm_stats[efeat_key]["sigma"] != 0, ("ZERO STANDARD DEVIATION!",efeat_key)
						efeat = torch.tanh((efeat - self.norm_stats[efeat_key]["mu"]) / self.norm_stats[efeat_key]["sigma"])

					if len(efeat.shape) == 1: efeat = torch.unsqueeze(efeat,dim=1)

					if "edge_features" not in example: example["edge_features"] = efeat
					else: example["edge_features"] = torch.cat((example["edge_features"],efeat),dim=1)

					del efeat

			#### Include ####

			for include_key in self.params["include"]: 
				example[include_key] = raw_example.get_tensor(include_key)
				if len(example[include_key].shape) == 0: example[include_key] = torch.unsqueeze(example[include_key],dim=0)

		meta = json.load(open(os.path.join(self.targ_src, self.example_map[str(idx)], "meta.json"), "r"))

		example["model"] = self.example_map[str(idx)]
		example["target"] = self.target
		example['r_uuid'] = meta["r_uuid"]

		####################
					
		return example

		####################

###################################

def parse_casp_scores(target):

	global_score_fl = os.path.join(_REPO_ROOT, "data", "scorebase", "CASP16", target + ".txt")

	parsed_scores = {}

	if os.path.exists(global_score_fl): 

		for line in open(global_score_fl,'r'): 
			row = line.strip().split()
			if len(row) != 30: continue # columns ...

			model = row[1]
			QSbest = row[5]
			if QSbest == "-": QSbest = None

			TMscore = row[27]
			if TMscore == "-": TMscore = None

			if model not in parsed_scores: parsed_scores[model] = {
				"TM-Score": TMscore,
				"QS-Best": QSbest
			}

	return parsed_scores

###################################

# This is test code...

# data_set = CASP16GeometricDataset(
# 	target = "H1106",
# 	params = {"type": "comprehensive_CASP16","include": []}
# )

# print(data_set.get_dimensions())

# for example in data_set: 
# 	print(example,"\n")
# 	break

###################################

