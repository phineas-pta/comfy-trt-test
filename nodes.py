# -*- coding: utf-8 -*-

# modified from https://github.com/NVIDIA/Stable-Diffusion-WebUI-TensorRT/blob/main/scripts/trt.py
# STATUS: draft !!!

# rabbit hole 1: a1111 unet loader
# - https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/dev/modules/sd_unet.py

# rabbit hole 2: comfy unet loader
# - https://github.com/comfyanonymous/ComfyUI/blob/master/nodes.py  >>>  UNETLoader
# - https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/sd.py  >>>  load_unet
# - https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/model_patcher.py
# - https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/model_base.py


import os
import numpy as np
import torch
from torch.cuda import nvtx
from .comfy_trt.model_manager import TRT_MODEL_DIR, modelmanager
from .comfy_trt.utilities import Engine

LIST_ENGINES = modelmanager.available_models()


class TRT_Unet_Loader:
	"""ComfyUI node"""

	RETURN_TYPES = ("MODEL",)
	CATEGORY = "advanced/loaders"
	FUNCTION = "load_trt"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"engine_file": (list(LIST_ENGINES.keys()),),
			}
		}

	def load_trt(self, engine_file):
		configs: list = LIST_ENGINES[engine_file]
		if configs[0]["config"].lora:
			model_name = configs[0]["base_model"]
			lora_path = os.path.join(TRT_MODEL_DIR, configs[0]["filepath"])
		else:
			model_name = engine_file
			lora_path = None
		return (TrtUnetWrapper(model_name, configs, lora_path),)


class TrtUnetWrapper:
	"""ComfyUI unet"""

	def __init__(self, model_name: str, configs: list, lora_path: str):
		self.diffusion_model = TrtUnet(model_name, configs, lora_path)
		self.latent_format = None # model_config.latent_format
		self.model_config = None # model_config
		self.model_type = None # model_type
		self.model_sampling = None # model_sampling(model_config, model_type)
		self.adm_channels = None # unet_config.get("adm_in_channels", None)
		self.inpaint_model = False

	def apply_model(self, x, t, c_concat=None, c_crossattn=None, c_adm=None, control=None, transformer_options={}):
		return None

	def get_dtype(self):
		return torch.float16

	def is_adm(self):
		return self.adm_channels > 0

	def encode_adm(self, **kwargs):
		return None

	def extra_conds(self, **kwargs):
		return None

	def set_inpaint(self):
		self.inpaint_model = True


class TrtUnet:
	"""A1111 unet"""

	def __init__(self, model_name: str, configs: list, lora_path: str):
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
