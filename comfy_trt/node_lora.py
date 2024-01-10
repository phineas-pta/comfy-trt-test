# -*- coding: utf-8 -*-

# modified from https://github.com/NVIDIA/Stable-Diffusion-WebUI-TensorRT/blob/main/scripts/lora.py
# combined with https://github.com/LucianoCirino/efficiency-nodes-comfyui/blob/main/efficiency_nodes.py >>> TSC_LoRA_Stacker
# STATUS: draft !!! still require onnx + engine built with lora

# cannot map lora state dict to unet state dict coz onnx export mess up all state dict keys

import numpy as np
import onnx
import torch
from torch.cuda import nvtx

from .model_manager import ONNX_MODEL_DIR, modelmanager

import folder_paths  # comfy stuff
from comfy.utils import load_torch_file


class TRT_Lora_Loader:

	RETURN_TYPES = ("MODEL",)
	CATEGORY = "advanced/loaders"
	FUNCTION = "lora_stacker"

	@classmethod
	def INPUT_TYPES(cls):
		min_val, max_val = 1, 3  # need JS to hide when big value

		loras = ["None"] + folder_paths.get_filename_list("loras")
		inputs = {"required": {
			"trt_unet_model": ("MODEL",),  # object from TrtUnetWrapper_Patch
			"LORA_COUNT": ("INT", {"default": min_val, "min": min_val, "max": max_val, "step": 1}),
		}}

		for i in range(min_val, max_val + 1):
			inputs["required"].update({
				f"lora_name_{i}": (loras,),
				f"lora_wt_{i}": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
			})

		return inputs

	def lora_stacker(self, trt_unet_model, LORA_COUNT, **kwargs):
		_, onnx_base_path = modelmanager.get_onnx_path(trt_unet_model.model.model_name)
		loras_list: list[tuple[str, float]] = [
			(name, kwargs.get(f"lora_wt_{i+1}"))
			for i in range(LORA_COUNT)
			if (name := kwargs.get(f"lora_name_{i+1}")) != "None"
		]
		refit_dict = apply_loras(onnx_base_path, loras_list)  # see def below

		if trt_unet_model.model.diffusion_model.engine is None:
			trt_unet_model.model.diffusion_model.activate()
		trt_unet_model.model.diffusion_model.apply_loras(refit_dict)  # see node_unet.py >>> class TrtUnet
		return (trt_unet_model,)


def apply_loras(onnx_base_path: str, loras_list: list[tuple[str, float]]) -> dict:
	refit_dict = {}
	for lora, weight in loras_list:
		lora_dict = load_torch_file(folder_paths.get_full_path("loras", lora))
		for k, v in lora_dict.items():
			if k in refit_dict:
				refit_dict[k] += weight * v
			else:
				refit_dict[k] = weight * v
	base = onnx.load(onnx_base_path)
	for initializer in base.graph.initializer:
		if initializer.name in refit_dict:
			wt = refit_dict[initializer.name]
			initializer_data = onnx.numpy_helper.to_array(initializer, base_dir=ONNX_MODEL_DIR).astype(np.float16)
			delta = torch.tensor(initializer_data).to(wt.device) + wt
			refit_dict[initializer.name] = delta.contiguous()
	return refit_dict
