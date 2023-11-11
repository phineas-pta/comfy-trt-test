# -*- coding: utf-8 -*-

# modified from https://github.com/NVIDIA/Stable-Diffusion-WebUI-TensorRT/blob/main/scripts/trt.py
# STATUS: working but need clean vram to change model

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

from comfy.model_base import ModelType, model_sampling
from comfy.supported_models import models as LIST_MODELS
from comfy import model_management

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
		return (TrtUnetWrapper_Patch(model_name, configs, lora_path),)


class TrtUnetWrapper_Patch:
	"""ComfyUI unet, see comfy/model_patcher.py"""
	def __init__(self, model_name: str, configs: list, lora_path: str):
		self.model = TrtUnetWrapper_Base(model_name, configs, lora_path)
		self.latent_format = self.model.latent_format
		self.model_sampling = self.model.model_sampling

		# workaround `model_patcher.py`
		self.current_device = self.offload_device = "cpu"
		self.model_options = {"transformer_options": {}}

		# workaround `latent_preview.py`
		self.load_device = model_management.get_torch_device()

	def model_dtype(self):
		return self.model.dtype

	def process_latent_in(self, latent):
		return self.latent_format.process_in(latent)

	def process_latent_out(self, latent):
		return self.latent_format.process_out(latent)

	def is_clone(self, other):
		if hasattr(other, "model") and self.model is other.model:
			return True
		return False

	def model_size(self):
		return os.stat(self.model.diffusion_model.engine.engine_path).st_size

	def model_patches_to(self, device):
		pass

	def patch_model(self, device_to=None):
		pass

	def unpatch_model(self, device_to=None):
		pass


class TrtUnetWrapper_Base:
	"""ComfyUI unet, see comfy/model_base.py"""

	def __init__(self, model_name: str, configs: list, lora_path: str):
		self.diffusion_model = TrtUnet(model_name, configs, lora_path)

		sd_ver: str = configs[0]["config"].diffusion_model
		for mod in LIST_MODELS:
			if mod.__qualname__ == sd_ver:
				self.model_config = mod
				break

		self.latent_format = self.model_config.latent_format()  # must init here
		self.model_type = ModelType.EPS
		self.model_sampling = model_sampling(self.model_config, self.model_type)
		self.adm_channels = self.model_config.unet_config.get("adm_in_channels", 0)
		self.inpaint_model: bool = configs[0]["config"].inpaint
		self.dtype = torch.float32 if configs[0]["config"].fp32 else torch.float16


	def apply_model(self, x, t, c_concat=None, c_crossattn=None, c_adm=None, control=None, transformer_options={}, **kwargs):
		self.diffusion_model.activate()  # must init here
		sigma = t
		xc = self.model_sampling.calculate_input(sigma, x)
		if c_concat is not None:
			xc = torch.cat([xc] + [c_concat], dim=1)

		context = c_crossattn
		dtype = self.dtype
		xc = xc.to(dtype)
		t = self.model_sampling.timestep(t).float()
		context = context.to(dtype)
		extra_conds = {}
		for o in kwargs:
			extra = kwargs[o]
			if hasattr(extra, "to"):
				extra = extra.to(dtype)
			extra_conds[o] = extra
		model_output = self.diffusion_model.forward(x=xc, timesteps=t, context=context, control=control, transformer_options=transformer_options, **extra_conds).float()
		return self.model_sampling.calculate_denoised(sigma, model_output, x)

	def set_inpaint(self):
		self.inpaint_model = True

	def process_latent_in(self, latent):
		return self.latent_format.process_in(latent)

	def process_latent_out(self, latent):
		return self.latent_format.process_out(latent)


class TrtUnet:
	"""A1111 unet, see A1111 TensorRT >>> trt.py"""

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
			if x.shape[-1] % 8 != 0 or x.shape[-2] % 8 != 0:  # not 64 here coz latent
				raise ValueError("Input shape must be divisible by 8 in both dimensions.")
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
