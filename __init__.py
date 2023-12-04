# -*- coding: utf-8 -*-

"""
@author: PTA
@title: TensorRT with ComfyUI (work-in-progress)
@nickname: comfy trt test
@description: attempt to use TensorRT with ComfyUI, not yet compatible with ComfyUI-Manager, see README for instructions
"""

# this file is the entry point so ComfyUI can detect as extension/custom node
# for more specifications see https://github.com/ltdrdata/ComfyUI-Manager

from .comfy_trt.node_unet import TRT_Unet_Loader
# from .comfy_trt.node_lora import TRT_Lora_Loader  # not working yet
# from .comfy_trt.node_torch_compile import TRT_Torch_Compile  # not working yet


NODE_CLASS_MAPPINGS = {
	"TRT_Unet_Loader": TRT_Unet_Loader,
	# "TRT_Lora_Loader": TRT_Lora_Loader,
	# "TRT_Torch_Compile": TRT_Torch_Compile,
}
NODE_DISPLAY_NAME_MAPPINGS = {
	"TRT_Unet_Loader": "load Unet in TensorRT",
	# "TRT_Lora_Loader": "load LoRA in TensorRT",
	# "TRT_Torch_Compile": "inplace compile Unet to TRT",
}
WEB_DIRECTORY = None

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
