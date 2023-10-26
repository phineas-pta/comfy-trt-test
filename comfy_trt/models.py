#
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
# See the License for the specific language governing permissions and
# limitations under the License.
#

import onnx
import os
from polygraphy.backend.onnx.loader import fold_constants
import tempfile
import torch
import onnx_graphsurgeon as gs


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


def get_controlnets_path(controlnet_list):
	"""not yet supported."""
	if controlnet_list is None:
		return None
	return ["lllyasviel/sd-controlnet-" + controlnet for controlnet in controlnet_list]


def get_clip_embedding_dim(version, pipeline):
	if version == "SD15":
		return 768
	elif version in ["SD20", "SD21UnclipL", "SD21UnclipH"]:
		return 1024
	elif version in ["SDXLRefiner", "SDXL"] and pipeline.is_sd_xl_base():
		return 768
	else:
		raise ValueError(f"Invalid version {version} + pipeline {pipeline}")


def get_clipwithproj_embedding_dim(version, pipeline):
	if version in ["SDXLRefiner", "SDXL"]:
		return 1280
	else:
		raise ValueError(f"Invalid version {version} + pipeline {pipeline}")


def get_unet_embedding_dim(version, pipeline):
	if version == "SD15":
		return 768
	elif version in ["SD20", "SD21UnclipL", "SD21UnclipH"]:
		return 1024
	elif version in ["SDXLRefiner", "SDXL"] and pipeline.is_sd_xl_base():
		return 2048
	elif version in ["SDXLRefiner", "SDXL"] and pipeline.is_sd_xl_refiner():
		return 1280
	else:
		raise ValueError(f"Invalid version {version} + pipeline {pipeline}")


class BaseModelBis:  # change name to distingush from existing 1 in comfy
	def __init__(
		self,
		version="SD15",
		pipeline=None,
		hf_token="",
		device="cuda",
		verbose=True,
		fp16=False,
		max_batch_size=16,
		text_maxlen=77,
		embedding_dim=768,
	):
		self.name = self.__class__.__name__
		self.pipeline = pipeline.name
		self.version = version
		self.hf_token = hf_token
		self.hf_safetensor = pipeline.is_sd_xl()
		self.device = device
		self.verbose = verbose

		self.fp16 = fp16

		self.min_batch = 1
		self.max_batch = max_batch_size
		self.min_image_shape = 256  # min image resolution: 256x256
		if version == "SD15":
			self.max_image_shape = 512
		elif version in ["SD20", "SD21UnclipL", "SD21UnclipH"]:
			self.max_image_shape = 768
		elif version in ["SDXLRefiner", "SDXL"]:
			self.max_image_shape = 1024
		self.min_latent_shape = self.min_image_shape // 8
		self.max_latent_shape = self.max_image_shape // 8

		self.text_maxlen = text_maxlen
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


class CLIP(BaseModelBis):
	def __init__(
		self,
		version,
		pipeline,
		hf_token,
		device,
		verbose,
		max_batch_size,
		embedding_dim,
		output_hidden_states=False,
		subfolder="text_encoder",
	):
		super(CLIP, self).__init__(
			version,
			pipeline,
			hf_token,
			device=device,
			verbose=verbose,
			max_batch_size=max_batch_size,
			embedding_dim=embedding_dim,
		)
		self.subfolder = subfolder

		# Output the final hidden state
		if output_hidden_states:
			self.extra_output_names = ["hidden_states"]

	def get_input_names(self):
		return ["input_ids"]

	def get_output_names(self):
		return ["text_embeddings"]

	def get_dynamic_axes(self):
		return {"input_ids": {0: "B"}, "text_embeddings": {0: "B"}}

	def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
		self.check_dims(batch_size, image_height, image_width)
		min_batch, max_batch, _, _, _, _, _, _, _, _ = self.get_minmax_dims(
			batch_size, image_height, image_width, static_batch, static_shape
		)
		return {
			"input_ids": [
				(min_batch, self.text_maxlen),
				(batch_size, self.text_maxlen),
				(max_batch, self.text_maxlen),
			]
		}

	def get_shape_dict(self, batch_size, image_height, image_width):
		self.check_dims(batch_size, image_height, image_width)
		output = {
			"input_ids": (batch_size, self.text_maxlen),
			"text_embeddings": (batch_size, self.text_maxlen, self.embedding_dim),
		}
		if "hidden_states" in self.extra_output_names:
			output["hidden_states"] = (batch_size, self.text_maxlen, self.embedding_dim)
		return output

	def get_sample_input(self, batch_size, image_height, image_width):
		self.check_dims(batch_size, image_height, image_width)
		return torch.zeros(batch_size, self.text_maxlen, dtype=torch.int32, device=self.device)

	def optimize(self, onnx_graph):
		opt = Optimizer(onnx_graph, verbose=self.verbose)
		opt.info(self.name + ": original")
		opt.select_outputs([0])  # delete graph output#1
		opt.cleanup()
		opt.info(self.name + ": remove output[1]")
		opt.fold_constants()
		opt.info(self.name + ": fold constants")
		opt.infer_shapes()
		opt.info(self.name + ": shape inference")
		opt.select_outputs([0], names=["text_embeddings"])  # rename network output
		opt.info(self.name + ": remove output[0]")
		opt_onnx_graph = opt.cleanup(return_onnx=True)
		if "hidden_states" in self.extra_output_names:
			opt_onnx_graph = opt.clip_add_hidden_states(return_onnx=True)
			opt.info(self.name + ": added hidden_states")
		opt.info(self.name + ": finished")
		return opt_onnx_graph


class CLIPWithProj(CLIP):
	def __init__(
		self,
		version,
		pipeline,
		hf_token,
		device="cuda",
		verbose=True,
		max_batch_size=16,
		output_hidden_states=False,
		subfolder="text_encoder_2",
	):
		super(CLIPWithProj, self).__init__(
			version,
			pipeline,
			hf_token,
			device=device,
			verbose=verbose,
			max_batch_size=max_batch_size,
			embedding_dim=get_clipwithproj_embedding_dim(version, pipeline),
			output_hidden_states=output_hidden_states,
		)
		self.subfolder = subfolder

	def get_shape_dict(self, batch_size, image_height, image_width):
		self.check_dims(batch_size, image_height, image_width)
		output = {
			"input_ids": (batch_size, self.text_maxlen),
			"text_embeddings": (batch_size, self.embedding_dim),
		}
		if "hidden_states" in self.extra_output_names:
			output["hidden_states"] = (batch_size, self.text_maxlen, self.embedding_dim)

		return output


class UNet2DConditionControlNetModel(torch.nn.Module):
	def __init__(self, unet, controlnets) -> None:
		super().__init__()
		self.unet = unet
		self.controlnets = controlnets

	def forward(self, sample, timestep, encoder_hidden_states, images, controlnet_scales):
		for i, (image, conditioning_scale, controlnet) in enumerate(zip(images, controlnet_scales, self.controlnets)):
			down_samples, mid_sample = controlnet(
				sample,
				timestep,
				encoder_hidden_states=encoder_hidden_states,
				controlnet_cond=image,
				return_dict=False,
			)

			down_samples = [down_sample * conditioning_scale for down_sample in down_samples]
			mid_sample *= conditioning_scale

			# merge samples
			if i == 0:
				down_block_res_samples, mid_block_res_sample = down_samples, mid_sample
			else:
				down_block_res_samples = [
					samples_prev + samples_curr
					for samples_prev, samples_curr in zip(down_block_res_samples, down_samples)
				]
				mid_block_res_sample += mid_sample

		noise_pred = self.unet(
			sample,
			timestep,
			encoder_hidden_states=encoder_hidden_states,
			down_block_additional_residuals=down_block_res_samples,
			mid_block_additional_residual=mid_block_res_sample,
		)
		return noise_pred


class UNet(BaseModelBis):
	def __init__(
		self,
		version,
		pipeline,
		hf_token,
		device="cuda",
		verbose=True,
		fp16=False,
		max_batch_size=16,
		text_maxlen=77,
		unet_dim=4,
		controlnet=None,
	):
		super(UNet, self).__init__(
			version,
			pipeline,
			hf_token,
			fp16=fp16,
			device=device,
			verbose=verbose,
			max_batch_size=max_batch_size,
			text_maxlen=text_maxlen,
			embedding_dim=get_unet_embedding_dim(version, pipeline),
		)
		self.unet_dim = unet_dim
		self.controlnet = controlnet

	def get_input_names(self):
		if self.controlnet is None:
			return ["sample", "timestep", "encoder_hidden_states"]
		else:
			return ["sample", "timestep", "encoder_hidden_states", "images", "controlnet_scales"]

	def get_output_names(self):
		return ["latent"]

	def get_dynamic_axes(self):
		if self.controlnet is None:
			return {
				"sample": {0: "2B", 2: "H", 3: "W"},
				"encoder_hidden_states": {0: "2B"},
				"latent": {0: "2B", 2: "H", 3: "W"},
			}
		else:
			return {
				"sample": {0: "2B", 2: "H", 3: "W"},
				"encoder_hidden_states": {0: "2B"},
				"images": {1: "2B", 3: "8H", 4: "8W"},
				"latent": {0: "2B", 2: "H", 3: "W"},
			}

	def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
		latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
		(
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
		) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
		if self.controlnet is None:
			return {
				"sample": [
					(2 * min_batch, self.unet_dim, min_latent_height, min_latent_width),
					(2 * batch_size, self.unet_dim, latent_height, latent_width),
					(2 * max_batch, self.unet_dim, max_latent_height, max_latent_width),
				],
				"encoder_hidden_states": [
					(2 * min_batch, self.text_maxlen, self.embedding_dim),
					(2 * batch_size, self.text_maxlen, self.embedding_dim),
					(2 * max_batch, self.text_maxlen, self.embedding_dim),
				],
			}
		else:
			return {
				"sample": [
					(2 * min_batch, self.unet_dim, min_latent_height, min_latent_width),
					(2 * batch_size, self.unet_dim, latent_height, latent_width),
					(2 * max_batch, self.unet_dim, max_latent_height, max_latent_width),
				],
				"encoder_hidden_states": [
					(2 * min_batch, self.text_maxlen, self.embedding_dim),
					(2 * batch_size, self.text_maxlen, self.embedding_dim),
					(2 * max_batch, self.text_maxlen, self.embedding_dim),
				],
				"images": [
					(len(self.controlnet), 2 * min_batch, 3, min_image_height, min_image_width),
					(len(self.controlnet), 2 * batch_size, 3, image_height, image_width),
					(len(self.controlnet), 2 * max_batch, 3, max_image_height, max_image_width),
				],
			}

	def get_shape_dict(self, batch_size, image_height, image_width):
		latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
		if self.controlnet is None:
			return {
				"sample": (2 * batch_size, self.unet_dim, latent_height, latent_width),
				"encoder_hidden_states": (2 * batch_size, self.text_maxlen, self.embedding_dim),
				"latent": (2 * batch_size, 4, latent_height, latent_width),
			}
		else:
			return {
				"sample": (2 * batch_size, self.unet_dim, latent_height, latent_width),
				"encoder_hidden_states": (2 * batch_size, self.text_maxlen, self.embedding_dim),
				"images": (len(self.controlnet), 2 * batch_size, 3, image_height, image_width),
				"latent": (2 * batch_size, 4, latent_height, latent_width),
			}

	def get_sample_input(self, batch_size, image_height, image_width):
		latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
		dtype = torch.float16 if self.fp16 else torch.float32
		if self.controlnet is None:
			return (
				torch.randn(batch_size, self.unet_dim, latent_height, latent_width, dtype=torch.float32, device=self.device),
				torch.tensor([1.0], dtype=torch.float32, device=self.device),
				torch.randn(batch_size, self.text_maxlen, self.embedding_dim, dtype=dtype, device=self.device),
			)
		else:
			return (
				torch.randn(batch_size, self.unet_dim, latent_height, latent_width, dtype=torch.float32, device=self.device),
				torch.tensor(999, dtype=torch.float32, device=self.device),
				torch.randn(batch_size, self.text_maxlen, self.embedding_dim, dtype=dtype, device=self.device),
				torch.randn(len(self.controlnet), batch_size, 3, image_height, image_width, dtype=dtype, device=self.device),
				torch.randn(len(self.controlnet), dtype=dtype, device=self.device),
			)


class OAIUNet(BaseModelBis):
	def __init__(
		self,
		version,
		pipeline,
		device="cuda",
		verbose=True,
		fp16=False,
		max_batch_size=16,
		text_maxlen=77,
		text_optlen=77,
		unet_dim=4,
		controlnet=None,
	):
		super(OAIUNet, self).__init__(
			version,
			pipeline,
			"",
			fp16=fp16,
			device=device,
			verbose=verbose,
			max_batch_size=max_batch_size,
			text_maxlen=text_maxlen,
			embedding_dim=get_unet_embedding_dim(version, pipeline),
		)
		self.unet_dim = unet_dim
		self.controlnet = controlnet
		self.text_optlen = text_optlen

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
		pipeline,
		fp16=False,
		device="cuda",
		verbose=True,
		max_batch_size=16,
		text_maxlen=77,
		text_optlen=77,
		unet_dim=4,
		time_dim=6,
		num_classes=2816,
	):
		super(OAIUNetXL, self).__init__(
			version,
			pipeline,
			"",
			fp16=fp16,
			device=device,
			verbose=verbose,
			max_batch_size=max_batch_size,
			text_maxlen=text_maxlen,
			embedding_dim=get_unet_embedding_dim(version, pipeline),
		)
		self.unet_dim = unet_dim
		self.time_dim = time_dim
		self.num_classes = num_classes
		self.text_optlen = text_optlen

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


class VAE(BaseModelBis):
	def __init__(self, version, pipeline, hf_token, device, verbose, max_batch_size):
		super(VAE, self).__init__(
			version,
			pipeline,
			hf_token,
			device=device,
			verbose=verbose,
			max_batch_size=max_batch_size,
		)

	def get_input_names(self):
		return ["latent"]

	def get_output_names(self):
		return ["images"]

	def get_dynamic_axes(self):
		return {
			"latent": {0: "B", 2: "H", 3: "W"},
			"images": {0: "B", 2: "8H", 3: "8W"},
		}

	def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
		latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
		(
			min_batch,
			max_batch,
			_,
			_,
			_,
			_,
			min_latent_height,
			max_latent_height,
			min_latent_width,
			max_latent_width,
		) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
		return {
			"latent": [
				(min_batch, 4, min_latent_height, min_latent_width),
				(batch_size, 4, latent_height, latent_width),
				(max_batch, 4, max_latent_height, max_latent_width),
			]
		}

	def get_shape_dict(self, batch_size, image_height, image_width):
		latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
		return {
			"latent": (batch_size, 4, latent_height, latent_width),
			"images": (batch_size, 3, image_height, image_width),
		}

	def get_sample_input(self, batch_size, image_height, image_width):
		latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
		return torch.randn(
			batch_size,
			4,
			latent_height,
			latent_width,
			dtype=torch.float32,
			device=self.device,
		)


class VAEEncoder(BaseModelBis):
	def __init__(
		self,
		version,
		pipeline,
		hf_token,
		device,
		verbose,
		max_batch_size,
	):
		super(VAEEncoder, self).__init__(
			version,
			pipeline,
			hf_token,
			device=device,
			verbose=verbose,
			max_batch_size=max_batch_size,
		)

	def get_input_names(self):
		return ["images"]

	def get_output_names(self):
		return ["latent"]

	def get_dynamic_axes(self):
		return {
			"images": {0: "B", 2: "8H", 3: "8W"},
			"latent": {0: "B", 2: "H", 3: "W"},
		}

	def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
		assert self.min_batch <= batch_size <= self.max_batch
		min_batch = batch_size if static_batch else self.min_batch
		max_batch = batch_size if static_batch else self.max_batch
		self.check_dims(batch_size, image_height, image_width)
		(
			min_batch,
			max_batch,
			min_image_height,
			max_image_height,
			min_image_width,
			max_image_width,
			_,
			_,
			_,
			_,
		) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)

		return {
			"images": [
				(min_batch, 3, min_image_height, min_image_width),
				(batch_size, 3, image_height, image_width),
				(max_batch, 3, max_image_height, max_image_width),
			],
		}

	def get_shape_dict(self, batch_size, image_height, image_width):
		latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
		return {
			"images": (batch_size, 3, image_height, image_width),
			"latent": (batch_size, 4, latent_height, latent_width),
		}

	def get_sample_input(self, batch_size, image_height, image_width):
		self.check_dims(batch_size, image_height, image_width)
		return torch.randn(batch_size, 3, image_height, image_width, dtype=torch.float32, device=self.device)
