# -*- coding: utf-8 -*-

# modified from https://github.com/cubiq/ComfyUI_essentials/blob/main/essentials.py >>> ModelCompile
# combined with https://pytorch.org/TensorRT/tutorials/_rendered_examples/dynamo/torch_compile_stable_diffusion.html

# STATUS: draft !!! need to build torch-tensorrt on windows

import torch
# import torch_tensorrt
torch._dynamo.config.suppress_errors = True

class TRT_Torch_Compile:
	RETURN_TYPES = ("MODEL", )
	FUNCTION = "compile"
	CATEGORY = "essentials"

	@classmethod
	def INPUT_TYPES(s): return {"required": {
		"model": ("MODEL",),
	}}

	def compile(self, model):
		model.model.diffusion_model = torch.compile(
			model.model.diffusion_model,
			backend="torch_tensorrt",
			options={
				"truncate_long_and_double": True,
				"precision": torch.float16,
			},
			dynamic=False,
		)
		return (model,)
