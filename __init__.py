# -*- coding: utf-8 -*-

"""
@author: PTA
@title: comfy trt test
@nickname: comfy trt test
@description: failed attempt to use TensorRT with ComfyUI, not yet compatible with ComfyUI-Manager, see README for instructions
"""

# this file is the entry point so ComfyUI can detect as extension/custom node
# for more specifications see https://github.com/ltdrdata/ComfyUI-Manager

from .nodes import TRT_Unet_Loader

NODE_CLASS_MAPPINGS = {
	"TRT_Unet_Loader": TRT_Unet_Loader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
	"TRT_Unet_Loader": "load Unet in TensorRT",
}
WEB_DIRECTORY = None

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
