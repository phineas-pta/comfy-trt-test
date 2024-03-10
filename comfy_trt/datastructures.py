# -*- coding: utf-8 -*-

# modified from https://github.com/NVIDIA/Stable-Diffusion-WebUI-TensorRT/blob/main/datastructures.py
# STATUS: draft

import json
from dataclasses import dataclass
import torch


@dataclass
class ModelConfig:
	profile: dict
	static_shapes: bool = False
	fp32: bool = False
	baseline_model: str = "SD15"  # save model info, for values see `comfy/supported_models.py`, breaking change incompatible A1111
	prediction_type: str = "ModelType.EPS"  # save model info, for values see `comfy/model_base.py`, breaking change incompatible A1111
	inpaint: bool = False
	refit: bool = False
	unet_hidden_dim: int = 4
	lora: bool = False
	controlnet: bool = False

	def is_compatible_from_dict(self, feed_dict: dict) -> tuple[bool, float]:
		distance = 0
		for k, v in feed_dict.items():
			_min, _opt, _max = self.profile[k]
			v_tensor = torch.Tensor(list(v.shape))
			r_min = torch.Tensor(_max) - v_tensor
			r_opt = (torch.Tensor(_opt) - v_tensor).abs()
			r_max = v_tensor - torch.Tensor(_min)
			if torch.any(r_min < 0) or torch.any(r_max < 0):
				return False, distance
			distance += r_opt.sum() + 0.5 * (r_max.sum() + 0.5 * r_min.sum())
		return True, distance

	def is_compatible(self, width: int, height: int, batch_size: int, max_embedding: int) -> tuple[bool, float]:
		sample = self.profile["sample"]
		embedding = self.profile["encoder_hidden_states"]

		batch_size *= 2
		width //=  8
		height //= 8

		_min, _opt, _max = sample
		_min_em, _opt_em, _max_em = embedding
		if (
			_min[0] > batch_size or _max[0] < batch_size
			or _min[2] > height or _max[2] < height
			or _min[3] > width or _max[3] < width
			or _min_em[1] > max_embedding or _max_em[1] < max_embedding
		):
			return False, 0
		else:
			distance = (
				abs(_opt[0] - batch_size)
				+ abs(_opt[2] - height)
				+ abs(_opt[3] - width)
				+ 0.5 * (abs(_max[2] - height) + abs(_max[3] - width))
			)
			return True, distance


class ModelConfigEncoder(json.JSONEncoder):
	def default(self, o: ModelConfig) -> dict:
		return o.__dict__


@dataclass
class ProfileSettings:
	bs_min: int  # batch size
	bs_opt: int
	bs_max: int
	h_min: int  # height
	h_opt: int
	h_max: int
	w_min: int  # width
	w_opt: int
	w_max: int
	t_min: int  # token count
	t_opt: int
	t_max: int
	static_shapes: bool = False

	def __str__(self) -> str:
		return "\n\t- ".join([
			"[I] size & shape parameters:",
			f"batch size: min={self.bs_min}, opt={self.bs_opt}, max={self.bs_max}",
			f"height: min={self.h_min}, opt={self.h_opt}, max={self.h_max}",
			f"width: min={self.w_min}, opt={self.w_opt}, max={self.w_max}",
			f"token count: min={self.t_min}, opt={self.t_opt}, max={self.t_max}",
		])

	def out(self) -> tuple[int]:
		return (
			self.bs_min, self.bs_opt, self.bs_max,
			self.h_min, self.h_opt, self.h_max,
			self.w_min, self.w_opt, self.w_max,
			self.t_min, self.t_opt, self.t_max,
		)

	def token_to_dim(self) -> None:
		self.t_min = (self.t_min // 75) * 77
		self.t_opt = (self.t_opt // 75) * 77
		self.t_max = (self.t_max // 75) * 77

		if self.static_shapes:
			self.t_min = self.t_max = self.t_opt
			self.bs_min = self.bs_max = self.bs_opt
			self.h_min = self.h_max = self.h_opt
			self.w_min = self.w_max = self.w_opt
			self.static_shapes = True

	def get_latent_dim(self) -> tuple[int]:
		return (
			self.h_min // 8, self.h_opt // 8, self.h_max // 8,
			self.w_min // 8, self.w_opt // 8, self.w_max // 8,
		)

	def get_batch_dim(self) -> tuple[int]:
		is_static_batch = self.bs_min == self.bs_max == self.bs_opt
		if self.t_max <= 77:
			return (self.bs_min * 2, self.bs_opt * 2, self.bs_max * 2)
		elif self.t_max > 77 and is_static_batch:
			return (self.bs_opt, self.bs_opt, self.bs_opt)
		elif self.t_max > 77 and not is_static_batch:
			if self.t_opt > 77:
				return (self.bs_min, self.bs_opt, self.bs_max * 2)
			else:
				return (self.bs_min, self.bs_opt * 2, self.bs_max * 2)
		else:
			raise Exception("Uncovered case in get_batch_dim")
