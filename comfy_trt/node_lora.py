# -*- coding: utf-8 -*-

# modified from https://github.com/NVIDIA/Stable-Diffusion-WebUI-TensorRT/blob/lora_v2/scripts/lora.py
# combined with https://github.com/LucianoCirino/efficiency-nodes-comfyui/blob/main/efficiency_nodes.py >>> TSC_LoRA_Stacker
# STATUS: draft !!! still require onnx + engine built with lora

# cannot map lora state dict to unet state dict coz onnx export mess up all state dict keys

import onnx_graphsurgeon as gs
import onnx
from torch.cuda import nvtx

from .model_manager import modelmanager

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
			"lora_count": ("INT", {"default": min_val, "min": min_val, "max": max_val, "step": 1}),
		}}

		for i in range(min_val, max_val + 1):
			inputs["required"].update({
				f"lora_name_{i}": (loras,),
				f"lora_wt_{i}": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
			})

		return inputs

	def lora_stacker(self, trt_unet_model, lora_count, **kwargs):
		_, onnx_base_path = modelmanager.get_onnx_base_path(trt_unet_model.model.model_name)
		loras_list: list[tuple[str, float]] = [
			(name, kwargs.get(f"lora_wt_{i+1}"))
			for i in range(lora_count)
			if (name := kwargs.get(f"lora_name_{i+1}")) != "None"
		]
		refit_dict = apply_loras(onnx_base_path, loras_list)

		if trt_unet_model.model.diffusion_model.engine.engine is None:
			trt_unet_model.model.diffusion_model.activate()
		nvtx.range_push("refit")
		trt_unet_model.model.diffusion_model.engine.refit_from_dict(refit_dict)
		nvtx.range_pop()
		return (trt_unet_model,)


def apply_loras(onnx_base_path: str, loras_list: list[tuple[str, float]]) -> dict:
	refit_dict = {}
	for lora, weight in loras_list:
		lora_dict = load_torch_file(folder_paths.get_full_path("loras", lora))
		for k, v in lora_dict.items():
			print(k)
			if k in refit_dict:
				refit_dict[k] += weight * v
			else:
				refit_dict[k] = weight * v

	base = gs.import_onnx(onnx.load(onnx_base_path)).toposort()
	for n in base.nodes:
		if n.op == "Constant" and (name := n.outputs[0].name) in refit_dict:
			refit_dict[name] += n.outputs[0].values

		# Handle scale and bias weights
		elif n.op == "Conv":
			if isinstance(n.inputs[1], gs.Constant) and (name := n.name + "_TRTKERNEL") in refit_dict:
				refit_dict[name] += n.outputs[1].values

			if isinstance(n.inputs[2], gs.Constant) and (name := n.name + "_TRTBIAS") in refit_dict:
				refit_dict[name] += n.outputs[2].values

		# For all other nodes: find node inputs that are initializers (AKA gs.Constant)
		else:
			for inp in n.inputs:
				if isinstance(inp, gs.Constant) and (name := inp.name) in refit_dict:
					refit_dict[name] += inp.values

	return refit_dict
