# -*- coding: utf-8 -*-

# modified from https://github.com/NVIDIA/Stable-Diffusion-WebUI-TensorRT/blob/main/scripts/trt.py
# CHANGE: wrap TrtUnet to make comfy node
# STATUS: working but need clean vram to change model

# rabbit hole 0: original unet implementation
# - https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/openaimodel.py >>> UNetModel

# rabbit hole 1: a1111 unet loader
# - https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/dev/modules/sd_unet.py

# rabbit hole 2: comfy unet loader
# - https://github.com/comfyanonymous/ComfyUI/blob/master/nodes.py >>> UNETLoader
# - https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/sd.py >>> load_unet
# - https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/model_patcher.py
# - https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/model_base.py

import os
from numpy import argmin
import torch
from torch.cuda import nvtx

from .model_manager import TRT_MODEL_DIR, modelmanager
from .utilities import Engine

from comfy.model_base import ModelType, model_sampling  # ModelType used in eval() - do not remove
import comfy.supported_models as LIST_MODELS  # used in eval() - do not remove
from comfy import model_management


LIST_ENGINES = modelmanager.available_models()


class TRT_Unet_Loader:
	"""ComfyUI node"""

	RETURN_TYPES = ("MODEL",)
	CATEGORY = "advanced/loaders"
	FUNCTION = "load_trt"

	@classmethod
	def INPUT_TYPES(cls): return {"required": {
		"engine_file": (list(LIST_ENGINES.keys()),),
		################################################# test: convert directly in GUI
		# "model" : ("MODEL",),
		# "batch_min": ("INT", {"default": 1, "min": 1, "max": 16}),
		# "batch_opt": ("INT", {"default": 1, "min": 1, "max": 16}),
		# "batch_max": ("INT", {"default": 1, "min": 1, "max": 16}),
		# "height_min": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 64}),
		# "height_opt": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 64}),
		# "height_max": ("INT", {"default": 768, "min": 256, "max": 4096, "step": 64}),
		# "width_min": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 64}),
		# "width_opt": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 64}),
		# "width_max": ("INT", {"default": 768, "min": 256, "max": 4096, "step": 64}),
		# "token_count_min": ("INT", {"default": 75, "min": 75, "max": 750}),
		# "token_count_opt": ("INT", {"default": 75, "min": 75, "max": 750}),
		# "token_count_max": ("INT", {"default": 75, "min": 75, "max": 750}),
		# "force_export": ("BOOLEAN", {"default": False}),
		# "static_shapes": ("BOOLEAN", {"default": False}),
		# "use_float32": ("BOOLEAN", {"default": False}),
	}}

	def load_trt(self, engine_file: str) -> tuple:
		configs: list = LIST_ENGINES[engine_file]
		if configs[0]["config"].lora:
			model_name = configs[0]["base_model"]
			lora_path = os.path.join(TRT_MODEL_DIR, configs[0]["filepath"])
		else:
			model_name = engine_file
			lora_path = None
		return (TrtUnetWrapper_Patch(model_name, configs, lora_path),)


class TrtUnetWrapper_Patch:
	"""ComfyUI unet patched, see comfy/model_patcher.py"""

	def __init__(self, model_name: str, configs: list, lora_path: str):
		self.model = TrtUnetWrapper_Base(model_name, configs, lora_path)
		self.latent_format = self.model.latent_format
		self.model_sampling = self.model.model_sampling
		self.current_device = self.offload_device = "cpu"
		self.model_options = {"transformer_options": {}}
		self.load_device = model_management.get_torch_device() # workaround `latent_preview.py`

	def model_dtype(self) -> torch.dtype:
		return self.model.dtype

	def process_latent_in(self, latent: torch.Tensor) -> torch.Tensor:
		return self.latent_format.process_in(latent)

	def process_latent_out(self, latent: torch.Tensor) -> torch.Tensor:
		return self.latent_format.process_out(latent)

	def is_clone(self, other) -> bool:
		if hasattr(other, "model") and self.model is other.model:
			return True
		else:
			return False

	def model_size(self) -> int:  # get file size as workaround, but incorrect if engine built with batch size > 1
		return os.stat(self.model.diffusion_model.engine.engine_path).st_size

	def memory_required(self, input_shape: torch.Size) -> float:
		return self.model.memory_required(input_shape=input_shape)

	def model_patches_to(self, device: torch.dtype | torch.device) -> None:
		pass

	def patch_model(self, device_to: torch.device = None) -> None:
		if self.model.diffusion_model.engine.engine is None:
			self.model.diffusion_model.activate()

	def unpatch_model(self, device_to: torch.device = None) -> None:
		self.model.diffusion_model.deactivate()


class TrtUnetWrapper_Base:
	"""ComfyUI unet base, see comfy/model_base.py"""

	def __init__(self, model_name: str, configs: list, lora_path: str):
		self.model_name = model_name
		self.diffusion_model = TrtUnet(model_name, configs, lora_path)
		self.model_config = eval("LIST_MODELS." + configs[0]["config"].baseline_model)  # e.g. "LIST_MODELS.SDXL"
		self.model_type = eval(configs[0]["config"].prediction_type)  # e.g. "ModelType.EPS"
		self.latent_format = self.model_config.latent_format()  # must init here
		self.model_sampling = model_sampling(self.model_config, self.model_type)
		self.adm_channels = self.model_config.unet_config.get("adm_in_channels", 0)
		self.inpaint_model: bool = configs[0]["config"].inpaint
		self.dtype = torch.float32 if configs[0]["config"].fp32 else torch.float16

	def apply_model(self, x: torch.Tensor, t: torch.Tensor, c_concat=None, c_crossattn=None, c_adm=None, control=None, transformer_options={}, **kwargs) -> torch.Tensor:
		sigma = t
		xc = self.model_sampling.calculate_input(sigma, x)
		if c_concat is not None:
			xc = torch.cat([xc] + [c_concat], dim=1)

		context = c_crossattn
		dtype = self.dtype
		xc = xc.to(dtype)
		t = self.model_sampling.timestep(t).float()
		context = context.to(dtype)
		extra_conds = {"c_adm": c_adm, "control": control, "transformer_options": transformer_options}
		for o in kwargs:
			extra = kwargs[o]
			if hasattr(extra, "to"):
				extra = extra.to(dtype)
			extra_conds[o] = extra
		model_output = self.diffusion_model.forward(x=xc, timesteps=t, context=context, **extra_conds).float()
		return self.model_sampling.calculate_denoised(sigma, model_output, x)

	def set_inpaint(self) -> None:
		self.inpaint_model = True

	def process_latent_in(self, latent: torch.Tensor) -> torch.Tensor:
		return self.latent_format.process_in(latent)

	def process_latent_out(self, latent: torch.Tensor) -> torch.Tensor:
		return self.latent_format.process_out(latent)

	def memory_required(self, input_shape: torch.Size) -> float:  # regularly watch comfy to update this formula
		area = input_shape[0] * input_shape[2] * input_shape[3]
		return (area * .6 / .9 + 1024) * 1024**2


class TrtUnet:
	"""A1111 unet, see A1111 TensorRT >>> trt.py"""

	def __init__(self, model_name: str, configs: list[dict], lora_path: str):
		self.configs = configs
		self.stream = None
		self.model_name = model_name
		self.lora_path = lora_path
		self.engine_vram_req = 0
		self.loaded_config = self.configs[0]
		self.shape_hash = 0
		self.engine = Engine(os.path.join(TRT_MODEL_DIR, self.loaded_config["filepath"]))

	def forward(self, x: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor, *args, **kwargs) -> torch.Tensor:
		"""
		Apply the model to an input batch

		Args:
			x: an [N × C × D × H × W] tensor of inputs
				N = batch size
				C = number of feature maps = number of channels
				D = image depth (this dimension not always present)
				H = image height
				W = image width
			timesteps: a 1-D batch of timesteps
			context: conditioning plugged in via cross-attention

		Returns:
			tensor with same shape as inputs
		"""
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
				raise ValueError("Input shape must be divisible by 64 in both dimensions.")
			self.switch_engine(feed_dict)
			self.shape_hash = hash(x.shape)
			nvtx.range_pop()

		tmp = torch.empty(self.engine_vram_req, dtype=torch.uint8, device="cuda")
		self.engine.context.device_memory = tmp.data_ptr()
		self.stream = torch.cuda.current_stream().cuda_stream
		self.engine.allocate_buffers(feed_dict)

		out = self.engine.infer(feed_dict, self.stream)["latent"]
		nvtx.range_pop()
		return out

	def switch_engine(self, feed_dict: dict) -> None:
		valid_models, distances = modelmanager.get_valid_models_from_dict(self.model_name, feed_dict)
		if len(valid_models) == 0:
			raise ValueError("No valid profile found.") # TODO: SDXL cannot get valid models

		best = valid_models[argmin(distances)]
		if best["filepath"] == self.loaded_config["filepath"]:
			return
		self.deactivate()
		self.engine = Engine(os.path.join(TRT_MODEL_DIR, best["filepath"]))
		self.activate()
		self.loaded_config = best

	def activate(self) -> None:
		self.engine.load()
		self.engine_vram_req = self.engine.engine.device_memory_size
		self.engine.activate(True)

		if self.lora_path is not None:
			self.engine.refit_from_dump(self.lora_path)

	def deactivate(self) -> None:
		self.shape_hash = 0
		backup = self.engine.engine_path
		del self.engine
		self.engine = Engine(backup)
