# -*- coding: utf-8 -*-

# modified from https://github.com/NVIDIA/Stable-Diffusion-WebUI-TensorRT/blob/lora_v2/scripts/lora.py
# combined with https://github.com/LucianoCirino/efficiency-nodes-comfyui/blob/main/efficiency_nodes.py >>> TSC_LoRA_Stacker
# CHANGE: wrap TrtUnet to make comfy node
# STATUS: working but need clean vram to change model

from safetensors.numpy import load_file
import onnx_graphsurgeon as gs
import onnx

import folder_paths


class TRT_Lora_Loader:

	RETURN_TYPES = ("MODEL",)
	CATEGORY = "advanced/loaders"
	FUNCTION = "lora_stacker"

	@classmethod
	def INPUT_TYPES(cls):
		min_val, max_val = 1, 50

		loras = ["None"] + folder_paths.get_filename_list("loras")
		inputs = {"required": {
			"lora_count": ("INT", {"default": 3, "min": min_val, "max": max_val, "step": 1}),
		}}

		for i in range(min_val, max_val + 1):
			inputs["required"].update({
				f"lora_name_{i}": (loras,),
				f"lora_wt_{i}": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
			})

		return inputs

	def lora_stacker(self, lora_count, **kwargs):
		res = []
		for i in range(lora_count):
			name = kwargs.get(f"lora_name_{i+1}")
			weight = kwargs.get(f"lora_wt_{i+1}")
			if name != "None":
				res.append((name, weight, weight))
		return (res,)


def apply_loras(onnx_path: str, loras: list[str], scales: list[str]) -> dict:
	base = gs.import_onnx(onnx.load(onnx_path)).toposort()
	refit_dict = {}
	for lora, scale in zip(loras, scales):
		lora_dict = load_file(lora)
		for k, v in lora_dict.items():
			if k in refit_dict:
				refit_dict[k] += scale * v
			else:
				refit_dict[k] = scale * v

	def add_to_map(refit_dict, name, value):
		if name in refit_dict:
			refit_dict[name] += value

	for n in base.nodes:
		if n.op == "Constant":
			name = n.outputs[0].name
			add_to_map(refit_dict, name, n.outputs[0].values)

		# Handle scale and bias weights
		elif n.op == "Conv":
			if n.inputs[1].__class__ == gs.Constant:
				name = n.name + "_TRTKERNEL"
				add_to_map(refit_dict, name, n.inputs[1].values)

			if n.inputs[2].__class__ == gs.Constant:
				name = n.name + "_TRTBIAS"
				add_to_map(refit_dict, name, n.inputs[2].values)

		# For all other nodes: find node inputs that are initializers (AKA gs.Constant)
		else:
			for inp in n.inputs:
				name = inp.name
				if inp.__class__ == gs.Constant:
					add_to_map(refit_dict, name, inp.values)

	return refit_dict
