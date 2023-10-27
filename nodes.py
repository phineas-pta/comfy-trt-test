# -*- coding: utf-8 -*-

# modified from https://github.com/NVIDIA/Stable-Diffusion-WebUI-TensorRT/blob/main/scripts/trt.py
# STATUS: draft !!!

import os
import numpy as np
import torch
from torch.cuda import nvtx
from .comfy_trt.model_manager import TRT_MODEL_DIR, modelmanager
from .comfy_trt.utilities import Engine
from comfy.model_base import BaseModel

LIST_ENGINES = {
	f: os.path.join(TRT_MODEL_DIR, f)
	for f in os.listdir(TRT_MODEL_DIR)
	if f.endswith(".trt")
}


class TrtUnetOption:
	def __init__(self, name: str, filename: list):
		self.label = f"[TRT] {name}"
		self.model_name = name
		self.configs = filename

	def create_unet(self):
		lora_path = None
		if self.configs[0]["config"].lora:
			lora_path = os.path.join(TRT_MODEL_DIR, self.configs[0]["filepath"])
			self.model_name = self.configs[0]["base_model"]
			self.configs = modelmanager.available_models()[self.model_name]
		return TrtUnet(self.model_name, self.configs, lora_path)

MODELS = modelmanager.available_models()
YOLO = [
	TrtUnetOption(
		"{} ({})".format(k, v[0]["base_model"]) if v[0]["config"].lora else k,
		v
	) for k, v in MODELS.items()
]


class TRT_Unet_Loader:
	RETURN_TYPES = ("MODEL",)

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"engine_file": (list(LIST_ENGINES.keys()),),
			}
		}

	def _load(self, engine_file):
		pass

class TrtModelWrapper(BaseModel):
	def __init__(self, model_config):
		super().__init__(model_config)
		self.diffusion_model = None

	def apply_model(self, x, t, c_concat=None, c_crossattn=None, c_adm=None, control=None, transformer_options={}):
		return None

	def get_dtype(self):
		return torch.float16

class TrtUnet:
	def __init__(self, model_name: str, configs: list, lora_path):
		self.configs = configs
		self.stream = None
		self.model_name = model_name
		self.lora_path = lora_path
		self.engine_vram_req = 0
		self.loaded_config = self.configs[0]
		self.shape_hash = 0
		self.engine = Engine(os.path.join(TRT_MODEL_DIR, self.loaded_config["filepath"]))

	def forward(self, x, timesteps, context, *args, **kwargs):
		nvtx.range_push("forward")
		feed_dict = {
			"sample": x.float(),
			"timesteps": timesteps.float(),
			"encoder_hidden_states": context.float(),
		}
		if "y" in kwargs:
			feed_dict["y"] = kwargs["y"].float()

		# Need to check compatability on the fly
		if self.shape_hash != hash(x.shape):
			nvtx.range_push("switch_engine")
			if x.shape[-1] % 64 != 0 or x.shape[-2] % 64 != 0:
				raise ValueError("Input shape must be divisible by 64 in both dimensions.")
			self.switch_engine(feed_dict)
			self.shape_hash = hash(x.shape)
			nvtx.range_pop()

		tmp = torch.empty(self.engine_vram_req, dtype=torch.uint8, device="cuda")
		self.engine.context.device_memory = tmp.data_ptr()
		self.cudaStream = torch.cuda.current_stream().cuda_stream
		self.engine.allocate_buffers(feed_dict)

		out = self.engine.infer(feed_dict, self.cudaStream)["latent"]
		nvtx.range_pop()
		return out

	def switch_engine(self, feed_dict):
		valid_models, distances = modelmanager.get_valid_models(self.model_name, feed_dict)
		if len(valid_models) == 0:
			raise ValueError("No valid profile found.")

		best = valid_models[np.argmin(distances)]
		if best["filepath"] == self.loaded_config["filepath"]:
			return
		self.deactivate()
		self.engine = Engine(os.path.join(TRT_MODEL_DIR, best["filepath"]))
		self.activate()
		self.loaded_config = best

	def activate(self):
		self.engine.load()
		print(self.engine)
		self.engine_vram_req = self.engine.engine.device_memory_size
		self.engine.activate(True)

		if self.lora_path is not None:
			self.engine.refit_from_dump(self.lora_path)

	def deactivate(self):
		self.shape_hash = 0
		del self.engine
