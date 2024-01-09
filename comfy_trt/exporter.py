# -*- coding: utf-8 -*-

# modified from https://github.com/NVIDIA/Stable-Diffusion-WebUI-TensorRT/blob/main/exporter.py
# CHANGE: remove lora
# STATUS: ok i guess

import logging
import time
import shutil
import os
import json
from collections import OrderedDict
import numpy as np
import torch
import onnx

from .utilities import Engine
from .datastructures import ProfileSettings
from .model_helper import UNetModel


def apply_lora(**kwargs):
	"""DO NOT USE not working yet"""
	pass


def get_refit_weights(state_dict: dict, onnx_opt_path: str, weight_name_mapping: dict, weight_shape_mapping: dict) -> dict:
	refit_weights = OrderedDict()
	onnx_opt_dir = os.path.dirname(onnx_opt_path)
	onnx_opt_model = onnx.load(onnx_opt_path)
	# Create initializer data hashes
	initializer_hash_mapping = {}
	onnx_data_mapping = {}
	for initializer in onnx_opt_model.graph.initializer:
		initializer_data = torch.numpy_helper.to_array(initializer, base_dir=onnx_opt_dir).astype(np.float16)
		initializer_hash = hash(initializer_data.data.tobytes())
		initializer_hash_mapping[initializer.name] = initializer_hash
		onnx_data_mapping[initializer.name] = initializer_data

	for torch_name, initializer_name in weight_name_mapping.items():
		initializer_hash = initializer_hash_mapping[initializer_name]
		wt = state_dict[torch_name]

		# get shape transform info
		initializer_shape, is_transpose = weight_shape_mapping[torch_name]
		wt = torch.transpose(wt, 0, 1) if is_transpose else torch.reshape(wt, initializer_shape)

		# include weight if hashes differ
		wt_hash = hash(wt.cpu().detach().numpy().astype(np.float16).data.tobytes())
		if initializer_hash != wt_hash:
			delta = wt - torch.tensor(onnx_data_mapping[initializer_name]).to(wt.device)
			refit_weights[initializer_name] = delta.contiguous()

	return refit_weights


def export_lora(modelobj: UNetModel, onnx_path: str, weights_map_path: str, lora_name: str, profile: ProfileSettings) -> dict:
	"""DO NOT USE not working yet"""
	logging.info("Exporting to ONNX...")
	inputs = modelobj.get_sample_input(profile.bs_opt * 2, profile.h_opt // 8, profile.w_opt // 8, profile.t_opt)
	with open(weights_map_path, "r") as fp_wts:
		print(f"[I] Loading weights map: {weights_map_path} ")
		[weights_name_mapping, weights_shape_mapping] = json.load(fp_wts)
	with torch.inference_mode(), torch.autocast("cuda"):
		modelobj.unet = apply_lora(modelobj.unet, os.path.splitext(lora_name)[0], inputs)
		refit_dict = get_refit_weights(modelobj.unet.state_dict(), onnx_path, weights_name_mapping, weights_shape_mapping)
	return refit_dict


def export_onnx(onnx_path: str, modelobj: UNetModel, profile: ProfileSettings, opset: int = 17, disable_optimizations: bool = False):
	logging.info("Exporting to ONNX...")
	inputs = modelobj.get_sample_input(profile.bs_opt * 2, profile.h_opt // 8, profile.w_opt // 8, profile.t_opt)
	if not os.path.exists(onnx_path):
		_export_onnx(
			modelobj.unet,
			inputs,
			onnx_path,
			opset,
			modelobj.get_input_names(),
			modelobj.get_output_names(),
			modelobj.get_dynamic_axes(),
			modelobj.optimize if not disable_optimizations else None,
		)


def _export_onnx(
	model: torch.nn.Module,
	inputs: tuple[torch.Tensor],
	path: str,
	opset: int,
	in_names: list[str],
	out_names: list[str],
	dyn_axes: dict,
	optimizer=None,
):
	tmp_dir = os.path.abspath("onnx_tmp")
	os.makedirs(tmp_dir, exist_ok=True)
	tmp_path = os.path.join(tmp_dir, "model.onnx")

	try:
		logging.info("Exporting to ONNX...")
		with torch.inference_mode(), torch.autocast("cuda"):
			torch.onnx.export(
				model,
				inputs,
				tmp_path,
				export_params=True,
				opset_version=opset,
				do_constant_folding=True,
				input_names=in_names,
				output_names=out_names,
				dynamic_axes=dyn_axes,
			)
	except Exception as e:
		logging.error("Exporting to ONNX failed. {}".format(e))
		return

	logging.info("Optimize ONNX.")
	onnx_model = onnx.load(tmp_path, load_external_data=False)

	# avoid using optimum as new dependency
	# https://github.com/huggingface/optimum/blob/main/optimum/onnx/utils.py > check_model_uses_external_data()
	model_tensors = onnx.external_data_helper._get_initializer_tensors(onnx_model)
	model_uses_external_data = any(
		tensor.HasField("data_location") and tensor.data_location == onnx.TensorProto.EXTERNAL
		for tensor in model_tensors
	)

	if model_uses_external_data:
		logging.info("ONNX model uses external data. Saving as external data.")
		onnx_model = onnx.load(tmp_path, load_external_data=True)
		onnx.save(
			onnx_model,
			path,
			save_as_external_data=True,
			all_tensors_to_one_file=True,
			# location=f"{path}_data",  # it breaks trt export
			size_threshold=1024,
		)

	if optimizer is not None:
		try:
			onnx_opt_graph = optimizer("unet", onnx_model)
			onnx.save(onnx_opt_graph, path)
		except Exception as e:
			logging.error(f"Optimizing ONNX failed. {e}")
			return

	if not model_uses_external_data and optimizer is None:
		shutil.move(tmp_path, str(path))
	shutil.rmtree(tmp_dir)


def export_trt(trt_path: str, onnx_path: str, timing_cache: str, profile: dict, use_fp16: bool) -> int:
	engine = Engine(trt_path)
	s = time.time()
	ret = engine.build(onnx_path, use_fp16, enable_refit=True, timing_cache=timing_cache, input_profile=[profile])
	e = time.time()
	logging.info(f"Time taken to build: {e-s}s")
	return ret
