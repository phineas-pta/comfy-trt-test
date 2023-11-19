# -*- coding: utf-8 -*-

# modified from https://github.com/NVIDIA/Stable-Diffusion-WebUI-TensorRT/blob/main/utilities.py
# CHANGE: remove pipeline & progress bar
# STATUS: ok i guess

# Copyright 2022 The HuggingFace Inc. team.
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

from collections import OrderedDict
import logging
import copy
import numpy as np
import onnx
import onnx_graphsurgeon as gs
from polygraphy.backend.common import bytes_from_path
from polygraphy import util as polyutil
from polygraphy.backend.trt import (
	ModifyNetworkOutputs,
	Profile,
	engine_from_bytes,
	engine_from_network,
	network_from_onnx_path,
	save_engine,
)
from polygraphy.logger import G_LOGGER
import tensorrt as trt
import torch
from torch.cuda import nvtx
from safetensors.numpy import save_file as st_save_file, load_file as st_load_file


TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
G_LOGGER.module_severity = G_LOGGER.ERROR

# Map of numpy dtype -> torch dtype
numpy_to_torch_dtype_dict = {
	np.uint8     : torch.uint8,
	np.int8      : torch.int8,
	np.int16     : torch.int16,
	np.int32     : torch.int32,
	np.int64     : torch.int64,
	np.float16   : torch.float16,
	np.float32   : torch.float32,
	np.float64   : torch.float64,
	np.complex64 : torch.complex64,
	np.complex128: torch.complex128,
}
if np.version.full_version >= "1.24.0":
	numpy_to_torch_dtype_dict[np.bool_] = torch.bool
else:
	numpy_to_torch_dtype_dict[np.bool] = torch.bool

# Map of torch dtype -> numpy dtype
torch_to_numpy_dtype_dict = {value: key for (key, value) in numpy_to_torch_dtype_dict.items()}


def change_trt_layer_name(layer_name: str, role: trt.WeightsRole) -> str:
	"""just refactoring a snippet used multiple times: for specialized roles, use a unique name in the map"""
	match role:
		case trt.WeightsRole.KERNEL:
			return layer_name + "_TRTKERNEL"
		case trt.WeightsRole.BIAS:
			return layer_name + "_TRTBIAS"
		case _:
			return layer_name


class Engine:
	def __init__(self, engine_path: str):
		self.engine_path = engine_path
		self.engine: trt.tensorrt.ICudaEngine = None
		self.context: trt.tensorrt.IExecutionContext = None
		self.buffers = OrderedDict()
		self.tensors = OrderedDict()

	def __del__(self):
		del self.engine
		del self.context
		del self.buffers
		del self.tensors

	def refit(self, onnx_path: str, onnx_refit_path: str, dump_refit_path: str | None = None) -> None:
		print(f"Refitting TensorRT engine with {onnx_refit_path} weights")
		refit_nodes = gs.import_onnx(onnx.load(onnx_refit_path)).toposort().nodes

		# Construct mapping from weight names in refit model -> original model
		name_map = {}
		for n, node in enumerate(gs.import_onnx(onnx.load(onnx_path)).toposort().nodes):
			refit_node = refit_nodes[n]
			assert node.op == refit_node.op
			# Constant nodes in ONNX do not have inputs but have a constant output
			if node.op == "Constant":
				name_map[refit_node.outputs[0].name] = node.outputs[0].name
			# Handle scale and bias weights
			elif node.op == "Conv":
				if isinstance(node.inputs[1], gs.Constant):
					name_map[refit_node.name + "_TRTKERNEL"] = node.name + "_TRTKERNEL"
				if isinstance(node.inputs[2], gs.Constant):
					name_map[refit_node.name + "_TRTBIAS"] = node.name + "_TRTBIAS"
			# For all other nodes: find node inputs that are initializers (gs.Constant)
			else:
				for i, inp in enumerate(node.inputs):
					if isinstance(inp, gs.Constant):
						name_map[refit_node.inputs[i].name] = inp.name

		def map_name(name):
			return name_map[name] if name in name_map else name

		# Construct refit dictionary
		refit_dict = {}
		refitter = trt.Refitter(self.engine, TRT_LOGGER)
		all_weights = refitter.get_all()
		for layer_name, role in zip(all_weights[0], all_weights[1]):
			name = change_trt_layer_name(layer_name, role)
			assert name not in refit_dict, "Found duplicate layer: " + name
			refit_dict[name] = None

		def add_to_map(name: str, values: np.ndarray) -> None:
			if name in refit_dict:
				assert refit_dict[name] is None
				if values.dtype == np.int64 and len(values.shape) == 0:
					values = np.int32(values)  # TODO: smarter conversion
				refit_dict[name] = values

		for n in refit_nodes:
			# Constant nodes in ONNX do not have inputs but have a constant output
			if n.op == "Constant":
				name = map_name(n.outputs[0].name)
				print(f"Add Constant {name}\n")
				try:
					add_to_map(name, n.outputs[0].values)
				except:
					logging.error(f"Failed to add Constant {name}\n")

			# Handle scale and bias weights
			elif n.op == "Conv":
				if isinstance(n.inputs[1], gs.Constant):
					name = map_name(n.name + "_TRTKERNEL")
					try:
						add_to_map(name, n.inputs[1].values)
					except:
						logging.error(f"Failed to add Conv {name}\n")

				if isinstance(n.inputs[2], gs.Constant):
					name = map_name(n.name + "_TRTBIAS")
					try:
						add_to_map(name, n.inputs[2].values)
					except:
						logging.error(f"Failed to add Conv {name}\n")

			# For all other nodes: find node inputs that are initializers (AKA gs.Constant)
			else:
				for inp in n.inputs:
					name = map_name(inp.name)
					if isinstance(inp, gs.Constant):
						add_to_map(name, inp.values)

		if dump_refit_path is not None:
			print("Finished refit. Dumping result to disk.")
			st_save_file(refit_dict, dump_refit_path)  # TODO need to come up with delta system to save only changed weights
			return

		for layer_name, weights_role in zip(all_weights[0], all_weights[1]):
			custom_name = change_trt_layer_name(layer_name, weights_role)

			# Skip refitting Trilu for now; scalar weights of type int64 value 1 - for clip model
			if layer_name.startswith("onnx::Trilu"):
				continue

			if refit_dict[custom_name] is not None:
				refitter.set_weights(layer_name, weights_role, refit_dict[custom_name])
			else:
				print(f"[W] No refit weights for layer: {layer_name}")

		if not refitter.refit_cuda_engine():
			print("Failed to refit!")
			exit(0)

	def refit_from_dump(self, dump_refit_path: str) -> None:
		refit_dict = st_load_file(dump_refit_path)  # TODO if deltas are used needs to be unpacked here

		refitter = trt.Refitter(self.engine, TRT_LOGGER)
		all_weights = refitter.get_all()

		for layer_name, weights_role in zip(all_weights[0], all_weights[1]):
			custom_name = change_trt_layer_name(layer_name, weights_role)

			# Skip refitting Trilu for now; scalar weights of type int64 value 1 - for clip model
			if layer_name.startswith("onnx::Trilu"):
				continue

			if refit_dict[custom_name] is not None:
				refitter.set_weights(layer_name, weights_role, refit_dict[custom_name])
			else:
				print(f"[W] No refit weights for layer: {layer_name}")

		if not refitter.refit_cuda_engine():
			print("Failed to refit!")
			exit(0)

	def refit_from_dict(self, refit_dict: dict) -> None:
		refitter = trt.Refitter(self.engine, TRT_LOGGER)
		all_weights = refitter.get_all()

		# TODO ideally iterate over refit_dict as len(refit_dict) < len(all_weights)
		for layer_name, weights_role in zip(all_weights[0], all_weights[1]):
			custom_name = change_trt_layer_name(layer_name, weights_role)

			# Skip refitting Trilu for now; scalar weights of type int64 value 1 - for clip model
			if layer_name.startswith("onnx::Trilu"):
				continue

			if custom_name in refit_dict:
				refitter.set_weights(layer_name, weights_role, refit_dict[custom_name])
				print(f"[I] refit weights for layer: {layer_name}")

		if not refitter.refit_cuda_engine():
			print("Failed to refit!")
			exit(0)

	def build(
		self,
		onnx_path: str,
		fp16: bool,
		input_profile: list[dict] | None = None,
		enable_refit: bool = False,
		enable_all_tactics: bool = False,
		timing_cache: str | None = None,
		update_output_names: str | None = None,
	) -> int:
		print(f"Building TensorRT engine for {onnx_path}: {self.engine_path}")
		if input_profile is None:
			p = [Profile()]
		else:
			p = []
			for i_p in input_profile:
				_p = Profile()
				for name, dims in i_p.items():
					assert len(dims) == 3
					_p.add(name, min=dims[0], opt=dims[1], max=dims[2])
				p.append(_p)

		config_kwargs = {}
		if not enable_all_tactics:
			config_kwargs["tactic_sources"] = []

		network = network_from_onnx_path(onnx_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM])
		if update_output_names:
			print(f"Updating network outputs to {update_output_names}")
			network = ModifyNetworkOutputs(network, update_output_names)

		builder = network[0]
		config = builder.create_builder_config()
		# config.progress_monitor = TQDMProgressMonitor() # need tensorrt v9
		if fp16: config.set_flag(trt.BuilderFlag.FP16)
		if enable_refit: config.set_flag(trt.BuilderFlag.REFIT)

		cache = None
		try:
			with polyutil.LockFile(timing_cache):
				timing_cache_data = polyutil.load_file(timing_cache, description="tactic timing cache")
				cache = config.create_timing_cache(timing_cache_data)
		except FileNotFoundError:
			logging.warning(f"Timing cache file {timing_cache} not found, falling back to empty timing cache.")
		if cache is not None:
			config.set_timing_cache(cache, ignore_mismatch=True)

		profiles = copy.deepcopy(p)
		for profile in profiles:
			# Last profile is used for set_calibration_profile.
			calib_profile = profile.fill_defaults(network[1]).to_trt(builder, network[1])
			config.add_optimization_profile(calib_profile)

		try:
			engine = engine_from_network(network, config, save_timing_cache=timing_cache)
		except Exception as e:
			logging.error(f"Failed to build engine: {e}")
			return 1
		try:
			save_engine(engine, path=self.engine_path)
		except Exception as e:
			logging.error(f"Failed to save engine: {e}")
			return 1
		return 0

	def load(self) -> None:
		print(f"Loading TensorRT engine: {self.engine_path}")
		self.engine = engine_from_bytes(bytes_from_path(self.engine_path))

	def activate(self, reuse_device_memory: bool = False) -> None:
		self.context = (
			self.engine.create_execution_context_without_device_memory()
			if reuse_device_memory
			else self.engine.create_execution_context()
		)

	def allocate_buffers(self, shape_dict: dict = None, device: str = "cuda") -> None:
		nvtx.range_push("allocate_buffers")
		for idx in range(self.engine.num_io_tensors):
			binding = self.engine[idx]
			if shape_dict and binding in shape_dict:
				shape = shape_dict[binding].shape
			else:
				shape = self.context.get_binding_shape(idx)
				shape = tuple(abs(x) for x in shape)  # TODO: find out why sometimes negative values
			dtype = trt.nptype(self.engine.get_binding_dtype(binding))
			if self.engine.binding_is_input(binding):
				self.context.set_binding_shape(idx, shape)
			tensor = torch.empty(tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]).to(device=device)
			self.tensors[binding] = tensor
		nvtx.range_pop()

	def infer(self, feed_dict: dict, stream: int) -> OrderedDict[str, torch.Tensor]:
		nvtx.range_push("set_tensors")
		for name, buf in feed_dict.items():
			self.tensors[name].copy_(buf)

		for name, tensor in self.tensors.items():
			self.context.set_tensor_address(name, tensor.data_ptr())
		nvtx.range_pop()
		nvtx.range_push("execute")
		noerror = self.context.execute_async_v3(stream)
		if not noerror:
			raise ValueError("ERROR: inference failed.")
		nvtx.range_pop()
		return self.tensors

	def __str__(self):
		out = ""
		for opt_profile in range(self.engine.num_optimization_profiles):
			out += f"Profile {opt_profile}:\n"
			for binding_idx in range(self.engine.num_bindings):
				name = self.engine.get_binding_name(binding_idx)
				shape = self.engine.get_profile_shape(opt_profile, name)
				out += f"\t{name} = {shape}\n"
		return out
