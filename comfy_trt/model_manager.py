# -*- coding: utf-8 -*-

# modified from https://github.com/NVIDIA/Stable-Diffusion-WebUI-TensorRT/blob/main/model_manager.py
# CHANGE: retrieve checkpoint info from comfy
# STATUS: ok i guess

import hashlib
import json
import os
import logging
import copy
import torch

from .datastructures import ModelConfig, ModelConfigEncoder


BASE_PATH = os.path.dirname(os.path.realpath(__file__))
ONNX_MODEL_DIR = os.path.join(BASE_PATH, "Unet-onnx")
if not os.path.exists(ONNX_MODEL_DIR):
	os.makedirs(ONNX_MODEL_DIR)
TRT_MODEL_DIR = os.path.join(BASE_PATH, "Unet-trt")
if not os.path.exists(TRT_MODEL_DIR):
	os.makedirs(TRT_MODEL_DIR)

MODEL_FILE = os.path.join(TRT_MODEL_DIR, "model.json")


def get_cc() -> tuple[int]:
	res = torch.cuda.get_device_properties(int(os.getenv("CUDA_VISIBLE_DEVICES", 0)))
	return res.major, res.minor
cc_major, cc_minor = get_cc()


class ModelManager:
	def __init__(self, model_file: str = MODEL_FILE):
		self.all_models = {}
		self.model_file = model_file
		self.cc = f"cc{cc_major}{cc_minor}"
		if not os.path.exists(model_file):
			logging.warning("Model file does not exist. Creating new one.")
		else:
			self.all_models = self.read_json()
		self.update()

	@staticmethod
	def get_onnx_path(model_name: str) -> tuple[str]:
		onnx_filename = f"{model_name}.onnx"
		onnx_path = os.path.join(ONNX_MODEL_DIR, onnx_filename)
		return onnx_filename, onnx_path

	def get_trt_path(self, model_name: str, profile: dict, static_shape: bool) -> tuple[str]:
		profile_hash = []
		n_profiles = 1 if static_shape else 3
		for k, v in profile.items():
			dim_hash = []
			for i in range(n_profiles):
				dim_hash.append("x".join([str(x) for x in v[i]]))
			profile_hash.append(k + "=" + "+".join(dim_hash))

		# shorter hash coz windows file path length limit
		hash_str = hashlib.blake2b("-".join(profile_hash).encode("utf-8"), digest_size=16).hexdigest()  # 16 digest = 32 char (original >110 char)
		trt_filename = model_name + "_" + hash_str + ".trt"
		trt_path = os.path.join(TRT_MODEL_DIR, trt_filename)

		return trt_filename, trt_path

	def get_weights_map_path(self, model_name: str):
		return os.path.join(TRT_MODEL_DIR, f"{model_name}_weights_map.json")

	def update(self) -> None:
		trt_engines = [trt_file for trt_file in os.listdir(TRT_MODEL_DIR) if trt_file.endswith(".trt")]

		tmp_all_models = copy.deepcopy(self.all_models)
		for cc, base_models in tmp_all_models.items():
			for base_model, models in base_models.items():
				tmp_config_list = {}
				for model_config in models:
					if model_config["filepath"] not in trt_engines:
						logging.info(f"Model config outdated. {model_config['filepath']} was not found")
						continue
					tmp_config_list[model_config["filepath"]] = model_config

				tmp_config_list = list(tmp_config_list.values())
				if len(tmp_config_list) == 0:
					self.all_models[cc].pop(base_model)
				else:
					self.all_models[cc][base_model] = models
		self.write_json()

	def add_entry(
		self,
		model_name: str,
		profile: dict,
		static_shapes: bool,
		fp32: bool,
		baseline_model: str,
		prediction_type: str,
		inpaint: bool,
		refit: bool,
		unet_hidden_dim: int,
		lora: bool
	) -> None:
		config = ModelConfig(profile, static_shapes, fp32, baseline_model, prediction_type, inpaint, refit, lora, unet_hidden_dim)
		trt_name, trt_path = self.get_trt_path(model_name, profile, static_shapes)

		base_model_name = model_name
		if self.cc not in self.all_models:
			self.all_models[self.cc] = {}

		if base_model_name not in self.all_models[self.cc]:
			self.all_models[self.cc][base_model_name] = []
		self.all_models[self.cc][base_model_name].append({"filepath": trt_name, "config": config})
		self.write_json()

	def add_lora_entry(
		self,
		base_model: str,
		lora_name: str,
		trt_lora_path: str,
		fp32: bool,
		baseline_model: str,
		prediction_type: str,
		inpaint: bool,
		unet_hidden_dim: int
	) -> None:
		config = ModelConfig([[], [], []], False, fp32, baseline_model, prediction_type, inpaint, True, True, unet_hidden_dim)
		self.all_models[self.cc][lora_name] = [{"filepath": trt_lora_path, "base_model": base_model, "config": config}]
		self.write_json()

	def write_json(self) -> None:
		with open(self.model_file, "w") as f:
			json.dump(self.all_models, f, indent=2, cls=ModelConfigEncoder)

	def read_json(self, encode_config: bool = True) -> dict:
		with open(self.model_file, "r") as f:
			out = json.load(f)

		if not encode_config:
			return out

		for cc, models in out.items():
			for base_model, configs in models.items():
				for i in range(len(configs)):
					out[cc][base_model][i]["config"] = ModelConfig(**configs[i]["config"])
		return out

	def available_models(self) -> dict:
		return self.all_models.get(self.cc, {})

	def available_loras(self):
		return {
			os.path.splitext(p)[0]: os.path.join(TRT_MODEL_DIR, p)
			for p in os.listdir(TRT_MODEL_DIR)
			if p.endswith(".lora")
		}

	def get_timing_cache(self) -> str:
		cache = os.path.join(
			BASE_PATH,
			"timing_caches",
			"timing_cache_{}_{}.cache".format("win" if os.name == "nt" else "linux", self.cc),
		)
		return cache

	def get_valid_models_from_dict(self, base_model: str, feed_dict: dict) -> tuple[list[bool], list[float]]:
		valid_models, distances = [], []
		for model in self.available_models()[base_model]:
			valid, distance = model["config"].is_compatible_from_dict(feed_dict)
			if valid:
				valid_models.append(model)
				distances.append(distance)
		return valid_models, distances

	def get_valid_models(self, base_model: str, width: int, height: int, batch_size: int, max_embedding: int) -> tuple[list[bool], list[float]]:
		valid_models, distances, idx = [], [], []
		for i, model in enumerate(self.available_models()[base_model]):
			valid, distance = model["config"].is_compatible(width, height, batch_size, max_embedding)
			if valid:
				valid_models.append(model)
				distances.append(distance)
				idx.append(i)
		return valid_models, distances, idx


modelmanager = ModelManager()
