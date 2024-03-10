# -*- coding: utf-8 -*-

# modified from https://github.com/NVIDIA/Stable-Diffusion-WebUI-TensorRT/blob/main/ui_trt.py
# CHANGE: remove lora, make script as CLI command
# STATUS: ok i guess

import argparse
import sys
import os.path
import gc
import torch

from comfy_trt.exporter import export_onnx, export_trt
from comfy_trt.model_helper import UNetModel
from comfy_trt.model_manager import modelmanager, cc_major
from comfy_trt.datastructures import ProfileSettings

sys.path.append(os.path.join("..", ".."))
from comfy.utils import load_torch_file
from comfy.supported_models import models as LIST_MODELS
from comfy.model_detection import detect_unet_config


def parseArgs():
	parser = argparse.ArgumentParser(description="test: convert Stable Diffusion checkpoint to TensorRT engine")
	parser.add_argument("--ckpt_path", required=True)
	parser.add_argument("--output_name", help=".onnx & .trt file name, default to ckpt file name")
	parser.add_argument("--batch_min", type=int, default=1, help="default 1")
	parser.add_argument("--batch_opt", type=int, default=1, help="default 1")
	parser.add_argument("--batch_max", type=int, default=1, help="limit 16")
	parser.add_argument("--height_min", type=int, help="default 768 if sdxl else 512, limit 256")
	parser.add_argument("--height_opt", type=int, help="default 1024 if sdxl else 512")
	parser.add_argument("--height_max", type=int, help="default 1024 if sdxl else 768, limit 4096")
	parser.add_argument("--width_min", type=int, help="default 768 if sdxl else 512, limit 256")
	parser.add_argument("--width_opt", type=int, help="default 768 if sdxl else 512")
	parser.add_argument("--width_max", type=int, help="default 1024 if sdxl else 768, limit 4096")
	parser.add_argument("--token_count_min", type=int, default=75, help="default 75, cannot go lower")
	parser.add_argument("--token_count_opt", type=int, default=75, help="default 75")
	parser.add_argument("--token_count_max", type=int, default=150, help="default 150, limit 750")
	parser.add_argument("--force_export", action="store_true")
	parser.add_argument("--static_shapes", action="store_true", help="may cause weird error (?) if enable")
	parser.add_argument("--float32", action="store_true")
	return parser.parse_args()


def get_config_from_checkpoint(ckpt_path: str) -> dict:
	"""see comfy/sd.py >>> load_checkpoint_guess_config"""
	tmp0 = "model.diffusion_model."
	sd = load_torch_file(ckpt_path)
	unet_config = detect_unet_config(sd, tmp0)
	for model_config in LIST_MODELS:
		if model_config.matches(unet_config):
			tmp1 = model_config(unet_config)
			model = tmp1.get_model(sd, tmp0, device="cuda")
			model.load_model_weights(sd, tmp0)
			return {
				"model": model.diffusion_model,
				"baseline_model": model_config.__qualname__,
				"prediction_type": str(model.model_type),
				"unet_hidden_dim": unet_config["in_channels"],
				"embedding_dim": unet_config["context_dim"],
			}


if __name__ == "__main__":
	args = parseArgs()

	ckpt_config = get_config_from_checkpoint(args.ckpt_path)
	if cc_major < 7:
		args.float32 = True
		print("FP16 has been disabled because your GPU does not support it.")

	baseline_model = ckpt_config["baseline_model"]
	print(f"\ndetected baseline model version: {baseline_model}")
	is_sdxl = baseline_model in ["SDXL", "SDXLRefiner", "SSD1B", "Segmind_Vega"]  # re-used later

	if is_sdxl:
		if args.height_min is None: args.height_min = 768
		if args.height_opt is None: args.height_opt = 1024
		if args.height_max is None: args.height_max = 1024
		if args.width_min is None: args.width_min = 768
		if args.width_opt is None: args.width_opt = 1024
		if args.width_max is None: args.width_max = 1024
	elif baseline_model in ["SD15", "SD20", "SD21UnclipL", "SD21UnclipH"]:
		if args.height_min is None: args.height_min = 512
		if args.height_opt is None: args.height_opt = 512
		if args.height_max is None: args.height_max = 768
		if args.width_min is None: args.width_min = 512
		if args.width_opt is None: args.width_opt = 512
		if args.width_max is None: args.width_max = 768
	else:  # ["SVD_img2vid", "Stable_Zero123", "SD_X4Upscaler", "Stable_Cascade_B", "Stable_Cascade_C", "KOALA_700M", "KOALA_1B",]

		raise ValueError(f"{baseline_model} not yet supported")

	if args.height_min % 64 != 0 or args.height_opt % 64 != 0 or args.height_max % 64 != 0 or args.width_min % 64 != 0 or args.width_opt % 64 != 0 or args.width_max % 64 != 0:
		raise ValueError("height and width must be divisible by 64")
	if not (args.height_min <= args.height_opt <= args.height_max and args.width_min <= args.width_opt <= args.width_max):
		raise ValueError("need min ≤ opt ≤ max")
	if args.height_min < 256 or args.height_max > 4096 or args.width_min < 256 or args.width_max > 4096:
		raise ValueError("height and width out of limit")

	ckpt_file = os.path.basename(args.ckpt_path)
	if args.output_name is None:  # default to ckpt file name
		args.output_name = os.path.splitext(ckpt_file)[0]
	onnx_filename, onnx_path = modelmanager.get_onnx_path(args.output_name)
	print(f"Exporting {ckpt_file} to TensorRT")
	timing_cache = modelmanager.get_timing_cache()

	profile_settings = ProfileSettings(
		args.batch_min, args.batch_opt, args.batch_max,
		args.height_min, args.height_opt, args.height_max,
		args.width_min, args.width_opt, args.width_max,
		args.token_count_min, args.token_count_opt, args.token_count_max,
		args.static_shapes
	)
	print(profile_settings, end="\n\n")
	profile_settings.token_to_dim()

	modelobj = UNetModel(
		unet=ckpt_config["model"],
		version=baseline_model,
		unet_dim=ckpt_config["unet_hidden_dim"],
		embedding_dim=ckpt_config["embedding_dim"],
		text_minlen=profile_settings.t_min
	)
	profile = modelobj.get_input_profile(profile_settings)

	export_onnx(onnx_path, modelobj, profile_settings, disable_optimizations=is_sdxl)

	trt_engine_filename, trt_path = modelmanager.get_trt_path(args.output_name, profile, args.static_shapes)

	# claim VRAM for TensorRT
	del ckpt_config["model"], modelobj.unet
	gc.collect()
	torch.cuda.empty_cache()

	if not os.path.exists(trt_path) or args.force_export:
		print("Building TensorRT engine… This can take a while.")
		ret = export_trt(trt_path, onnx_path, timing_cache, profile=profile, use_fp16=not args.float32)
		if ret:
			print("Export Failed due to unknown reason.")

		else:
			print("TensorRT engines has been saved to disk.")
			modelmanager.add_entry(
				args.output_name,
				profile,
				args.static_shapes,
				fp32=args.float32,
				baseline_model=baseline_model,  # breaking change incompatible A1111
				prediction_type=ckpt_config["prediction_type"],  # breaking change incompatible A1111
				inpaint=ckpt_config["unet_hidden_dim"] > 4,
				refit=True,
				unet_hidden_dim=ckpt_config["unet_hidden_dim"],
				lora=False,
			)
	else:
		print("TensorRT engine found. Skipping build. You can enable Force Export in the Advanced Settings to force a rebuild if needed.")
