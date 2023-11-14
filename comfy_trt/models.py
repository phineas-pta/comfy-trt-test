# -*- coding: utf-8 -*-

# modified from https://github.com/NVIDIA/Stable-Diffusion-WebUI-TensorRT/blob/main/models.py
# CHANGE: remove unused classes, only keep unet
# STATUS: ok i guess

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

import os
import tempfile
import onnx
import onnx_graphsurgeon as gs
import torch
from polygraphy.backend.onnx.loader import fold_constants
from polygraphy.logger import G_LOGGER


G_LOGGER.module_severity = G_LOGGER.ERROR


class Optimizer:
	def __init__(self, onnx_graph, verbose=False):
		self.graph = gs.import_onnx(onnx_graph)
		self.verbose = verbose

	def info(self, prefix):
		if self.verbose:
			print(f"{prefix} .. {len(self.graph.nodes)} nodes, {len(self.graph.tensors().keys())} tensors, {len(self.graph.inputs)} inputs, {len(self.graph.outputs)} outputs")

	def cleanup(self, return_onnx=False):
		self.graph.cleanup().toposort()
		if return_onnx:
			return gs.export_onnx(self.graph)

	def select_outputs(self, keep, names=None):
		self.graph.outputs = [self.graph.outputs[o] for o in keep]
		if names:
			for i, name in enumerate(names):
				self.graph.outputs[i].name = name

	def fold_constants(self, return_onnx=False):
		onnx_graph = fold_constants(gs.export_onnx(self.graph), allow_onnxruntime_shape_inference=True)
		self.graph = gs.import_onnx(onnx_graph)
		if return_onnx:
			return onnx_graph

	def infer_shapes(self, return_onnx=False):
		onnx_graph = gs.export_onnx(self.graph)
		if onnx_graph.ByteSize() > 2147483648:
			temp_dir = tempfile.TemporaryDirectory().name
			os.makedirs(temp_dir, exist_ok=True)
			onnx_orig_path = os.path.join(temp_dir, "model.onnx")
			onnx_inferred_path = os.path.join(temp_dir, "inferred.onnx")
			onnx.save_model(
				onnx_graph,
				onnx_orig_path,
				save_as_external_data=True,
				all_tensors_to_one_file=True,
				convert_attribute=False,
			)
			onnx.shape_inference.infer_shapes_path(onnx_orig_path, onnx_inferred_path)
			onnx_graph = onnx.load(onnx_inferred_path)
		else:
			onnx_graph = onnx.shape_inference.infer_shapes(onnx_graph)

		self.graph = gs.import_onnx(onnx_graph)
		if return_onnx:
			return onnx_graph

	def clip_add_hidden_states(self, return_onnx=False):
		hidden_layers = -1
		onnx_graph = gs.export_onnx(self.graph)
		for i in range(len(onnx_graph.graph.node)):
			for j in range(len(onnx_graph.graph.node[i].output)):
				name = onnx_graph.graph.node[i].output[j]
				if "layers" in name:
					hidden_layers = max(int(name.split(".")[1].split("/")[0]), hidden_layers)
		temp = f"/text_model/encoder/layers.{hidden_layers - 1}/Add_1_output_0"
		for i in range(len(onnx_graph.graph.node)):
			for j in range(len(onnx_graph.graph.node[i].output)):
				if onnx_graph.graph.node[i].output[j] == temp:
					onnx_graph.graph.node[i].output[j] = "hidden_states"
			for j in range(len(onnx_graph.graph.node[i].input)):
				if onnx_graph.graph.node[i].input[j] == temp:
					onnx_graph.graph.node[i].input[j] = "hidden_states"
		if return_onnx:
			return onnx_graph


def get_unet_embedding_dim(version):
	if version == "SD15":
		return 768
	elif version in ["SD20", "SD21UnclipL", "SD21UnclipH"]:
		return 1024
	elif version == "SDXL":
		return 2048
	elif version == "SDXLRefiner":
		return 1280
	else:
		raise ValueError(f"Invalid version {version}")


class BaseModelBis:  # change name to distingush from existing 1 in comfy
	def __init__(
		self,
		version,
		device="cuda",
		fp16=False,
		verbose=True,
		max_batch_size=16,
		text_maxlen=77,
		text_optlen=77,
		unet_dim=4,  # 9 for inpaint model
		embedding_dim=768,
		controlnet=None,
	):
		self.name = self.__class__.__name__
		self.version = version
		self.device = device
		self.verbose = verbose
		self.fp16 = fp16
		self.controlnet = controlnet

		self.min_batch = 1
		self.max_batch = max_batch_size
		self.min_image_shape = 256  # min image resolution: 256x256
		self.max_image_shape = 512 if version == "SD15" else 1024
		self.min_latent_shape = self.min_image_shape // 8
		self.max_latent_shape = self.max_image_shape // 8

		self.text_maxlen = text_maxlen
		self.unet_dim = unet_dim
		self.text_optlen = text_optlen
		self.embedding_dim = embedding_dim
		self.extra_output_names = []

	def get_input_names(self):
		pass

	def get_output_names(self):
		pass

	def get_dynamic_axes(self):
		return None

	def get_sample_input(self, batch_size, image_height, image_width):
		pass

	def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
		return None

	def get_shape_dict(self, batch_size, image_height, image_width):
		return None

	def optimize(self, onnx_graph):
		opt = Optimizer(onnx_graph, verbose=self.verbose)
		opt.info(self.name + ": original")
		opt.cleanup()
		opt.info(self.name + ": cleanup")
		opt.fold_constants()
		opt.info(self.name + ": fold constants")
		opt.infer_shapes()
		opt.info(self.name + ": shape inference")
		onnx_opt_graph = opt.cleanup(return_onnx=True)
		opt.info(self.name + ": finished")
		return onnx_opt_graph

	def check_dims(self, batch_size, image_height, image_width):
		assert self.min_batch <= batch_size <= self.max_batch
		assert image_height % 8 == 0 or image_width % 8 == 0
		latent_height = image_height // 8
		latent_width = image_width // 8
		assert self.min_latent_shape <= latent_height <= self.max_latent_shape
		assert self.min_latent_shape <= latent_width <= self.max_latent_shape
		return latent_height, latent_width

	def get_minmax_dims(self, batch_size, image_height, image_width, static_batch, static_shape):
		min_batch = batch_size if static_batch else self.min_batch
		max_batch = batch_size if static_batch else self.max_batch
		latent_height = image_height // 8
		latent_width = image_width // 8
		min_image_height = image_height if static_shape else self.min_image_shape
		max_image_height = image_height if static_shape else self.max_image_shape
		min_image_width = image_width if static_shape else self.min_image_shape
		max_image_width = image_width if static_shape else self.max_image_shape
		min_latent_height = latent_height if static_shape else self.min_latent_shape
		max_latent_height = latent_height if static_shape else self.max_latent_shape
		min_latent_width = latent_width if static_shape else self.min_latent_shape
		max_latent_width = latent_width if static_shape else self.max_latent_shape
		return (
			min_batch,
			max_batch,
			min_image_height,
			max_image_height,
			min_image_width,
			max_image_width,
			min_latent_height,
			max_latent_height,
			min_latent_width,
			max_latent_width,
		)

	def get_latent_dim(self, min_h, opt_h, max_h, min_w, opt_w, max_w, static_shape):
		if static_shape:
			return opt_h // 8, opt_h // 8, opt_h // 8, opt_w // 8, opt_w // 8, opt_w // 8
		else:
			return min_h // 8, opt_h // 8, max_h // 8, min_w // 8, opt_w // 8, max_w // 8

	def get_batch_dim(self, min_batch, opt_batch, max_batch, static_batch):
		if self.text_maxlen <= 77:
			return min_batch * 2, opt_batch * 2, max_batch * 2
		elif self.text_maxlen > 77 and static_batch:
			return opt_batch, opt_batch, opt_batch
		elif self.text_maxlen > 77 and not static_batch:
			if self.text_optlen > 77:
				return min_batch, opt_batch, max_batch * 2
			return min_batch, opt_batch * 2, max_batch * 2
		else:
			raise Exception("Uncovered case in get_batch_dim")


class OAIUNet(BaseModelBis):
	def __init__(
		self,
		version,
		device="cuda",
		fp16=True,
		verbose=True,
		max_batch_size=16,
		text_maxlen=77,
		text_optlen=77,
		unet_dim=4,  # 9 for inpaint model
		controlnet=None,
	):
		super().__init__(
			version=version,
			fp16=fp16,
			device=device,
			verbose=verbose,
			max_batch_size=max_batch_size,
			text_maxlen=text_maxlen,
			text_optlen=text_optlen,
			unet_dim=unet_dim,
			embedding_dim=get_unet_embedding_dim(version),
			controlnet=controlnet,
		)

	def get_input_names(self):
		if self.controlnet is None:
			return ["sample", "timesteps", "encoder_hidden_states"]
		else:
			return ["sample", "timesteps", "encoder_hidden_states", "images", "controlnet_scales"]

	def get_output_names(self):
		return ["latent"]

	def get_dynamic_axes(self):
		if self.controlnet is None:
			return {
				"sample": {0: "2B", 2: "H", 3: "W"},
				"timesteps": {0: "2B"},
				"encoder_hidden_states": {0: "2B", 1: "77N"},
				"latent": {0: "2B", 2: "H", 3: "W"},
			}
		else:
			return {
				"sample": {0: "2B", 2: "H", 3: "W"},
				"timesteps": {0: "2B"},
				"encoder_hidden_states": {0: "2B", 1: "77N"},
				"images": {1: "2B", 3: "8H", 4: "8W"},
				"latent": {0: "2B", 2: "H", 3: "W"},
			}

	def get_input_profile(self, min_batch, opt_batch, max_batch, min_h, opt_h, max_h, min_w, opt_w, max_w, static_shape):
		min_batch, opt_batch, max_batch = self.get_batch_dim(min_batch, opt_batch, max_batch, static_shape)
		(
			min_latent_height,
			latent_height,
			max_latent_height,
			min_latent_width,
			latent_width,
			max_latent_width,
		) = self.get_latent_dim(min_h, opt_h, max_h, min_w, opt_w, max_w, static_shape)

		if self.controlnet is None:
			return {
				"sample": [
					(min_batch, self.unet_dim, min_latent_height, min_latent_width),
					(opt_batch, self.unet_dim, latent_height, latent_width),
					(max_batch, self.unet_dim, max_latent_height, max_latent_width),
				],
				"timesteps": [
					(min_batch,),
					(opt_batch,),
					(max_batch,)
				],
				"encoder_hidden_states": [
					(min_batch, self.text_optlen, self.embedding_dim),
					(opt_batch, self.text_optlen, self.embedding_dim),
					(max_batch, self.text_maxlen, self.embedding_dim),
				],
			}
		else:
			return {
				"sample": [
					(min_batch, self.unet_dim, min_latent_height, min_latent_width),
					(opt_batch, self.unet_dim, latent_height, latent_width),
					(max_batch, self.unet_dim, max_latent_height, max_latent_width),
				],
				"timesteps": [
					(min_batch,),
					(opt_batch,),
					(max_batch,)
				],
				"encoder_hidden_states": [
					(min_batch, self.text_optlen, self.embedding_dim),
					(opt_batch, self.text_optlen, self.embedding_dim),
					(max_batch, self.text_maxlen, self.embedding_dim),
				],
				"images": [
					(len(self.controlnet), min_batch, 3, min_h, min_w),
					(len(self.controlnet), opt_batch, 3, opt_h, opt_w),
					(len(self.controlnet), max_batch, 3, max_h, max_w),
				],
			}

	def get_shape_dict(self, batch_size, image_height, image_width):
		latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
		if self.controlnet is None:
			return {
				"sample": (2 * batch_size, self.unet_dim, latent_height, latent_width),
				"timesteps": (2 * batch_size,),
				"encoder_hidden_states": (2 * batch_size, self.text_optlen, self.embedding_dim),
				"latent": (2 * batch_size, 4, latent_height, latent_width),
			}
		else:
			return {
				"sample": (2 * batch_size, self.unet_dim, latent_height, latent_width),
				"timesteps": (2 * batch_size,),
				"encoder_hidden_states": (2 * batch_size, self.text_optlen, self.embedding_dim),
				"images": (len(self.controlnet), 2 * batch_size, 3, image_height, image_width),
				"latent": (2 * batch_size, 4, latent_height, latent_width),
			}

	def get_sample_input(self, batch_size, image_height, image_width):
		latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
		dtype = torch.float16 if self.fp16 else torch.float32
		if self.controlnet is None:
			return (
				torch.randn(2 * batch_size, self.unet_dim, latent_height, latent_width, dtype=torch.float32, device=self.device),
				torch.ones((2 * batch_size,), dtype=torch.float32, device=self.device),
				torch.randn(2 * batch_size, self.text_optlen, self.embedding_dim, dtype=dtype, device=self.device),
			)
		else:
			return (
				torch.randn(batch_size, self.unet_dim, latent_height, latent_width, dtype=torch.float32, device=self.device),
				torch.ones((batch_size,), dtype=torch.float32, device=self.device),
				torch.randn(batch_size, self.text_optlen, self.embedding_dim, dtype=dtype, device=self.device),
				torch.randn(len(self.controlnet), batch_size, 3, image_height, image_width, dtype=dtype, device=self.device),
				torch.randn(len(self.controlnet), dtype=dtype, device=self.device),
			)


class OAIUNetXL(BaseModelBis):
	def __init__(
		self,
		version,
		device="cuda",
		fp16=True,
		verbose=True,
		max_batch_size=16,
		text_maxlen=77,
		text_optlen=77,
		unet_dim=4,  # 9 for inpaint model
		time_dim=6,
		num_classes=2816,  # 2560 for refiner
		controlnet=None,
	):
		super().__init__(
			version=version,
			fp16=fp16,
			device=device,
			verbose=verbose,
			max_batch_size=max_batch_size,
			text_maxlen=text_maxlen,
			text_optlen=text_optlen,
			unet_dim=unet_dim,
			embedding_dim=get_unet_embedding_dim(version),
			controlnet=controlnet,
		)
		self.time_dim = time_dim
		self.num_classes = num_classes

	def get_input_names(self):
		return ["sample", "timesteps", "encoder_hidden_states", "y"]

	def get_output_names(self):
		return ["latent"]

	def get_dynamic_axes(self):
		return {
			"sample": {0: "2B", 2: "H", 3: "W"},
			"encoder_hidden_states": {0: "2B", 1: "77N"},
			"timesteps": {0: "2B"},
			"latent": {0: "2B", 2: "H", 3: "W"},
			"y": {0: "2B", 1: "num_classes"},
		}

	def get_input_profile(self, min_batch, opt_batch, max_batch, min_h, opt_h, max_h, min_w, opt_w, max_w, static_shape):
		min_batch, opt_batch, max_batch = self.get_batch_dim(min_batch, opt_batch, max_batch, static_shape)

		(
			min_latent_height,
			latent_height,
			max_latent_height,
			min_latent_width,
			latent_width,
			max_latent_width,
		) = self.get_latent_dim(min_h, opt_h, max_h, min_w, opt_w, max_w, static_shape)

		return {
			"sample": [
				(min_batch, self.unet_dim, min_latent_height, min_latent_width),
				(opt_batch, self.unet_dim, latent_height, latent_width),
				(max_batch, self.unet_dim, max_latent_height, max_latent_width),
			],
			"timesteps": [
				(min_batch,),
				(opt_batch,),
				(max_batch,)
			],
			"encoder_hidden_states": [
				(min_batch, self.text_optlen, self.embedding_dim),
				(opt_batch, self.text_optlen, self.embedding_dim),
				(max_batch, self.text_maxlen, self.embedding_dim),
			],
			"y": [
				(min_batch, self.num_classes),
				(opt_batch, self.num_classes),
				(max_batch, self.num_classes),
			],
		}

	def get_shape_dict(self, batch_size, image_height, image_width):
		latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
		return {
			"sample": (2 * batch_size, self.unet_dim, latent_height, latent_width),
			"timesteps": (2 * batch_size,),
			"encoder_hidden_states": (2 * batch_size, self.text_optlen, self.embedding_dim),
			"y": (2 * batch_size, self.num_classes),
			"latent": (2 * batch_size, 4, latent_height, latent_width),
		}

	def get_sample_input(self, batch_size, image_height, image_width):
		latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
		dtype = torch.float16 if self.fp16 else torch.float32
		return (
			torch.randn(2 * batch_size, self.unet_dim, latent_height, latent_width, dtype=torch.float32, device=self.device),
			torch.ones((2 * batch_size,), dtype=torch.float32, device=self.device),
			torch.randn(2 * batch_size, self.text_optlen, self.embedding_dim, dtype=dtype, device=self.device),
			torch.randn(2 * batch_size, self.num_classes, dtype=dtype, device=self.device),
		)
