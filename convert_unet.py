# -*- coding: utf-8 -*-

# modified from https://github.com/NVIDIA/Stable-Diffusion-WebUI-TensorRT/blob/main/ui_trt.py
# STATUS: ok i guess

import argparse
import sys
import os.path
import gc
import torch

from comfy_trt.exporter import export_onnx, export_trt
from comfy_trt.models import OAIUNet, OAIUNetXL
from comfy_trt.model_manager import modelmanager, cc_major

sys.path.append(os.path.join("..", ".."))
from comfy.utils import load_torch_file, calculate_parameters
from comfy.supported_models import models as LIST_MODELS
from comfy.model_detection import detect_unet_config
from comfy.model_management import unet_dtype as get_unet_dtype


def parseArgs():
	parser = argparse.ArgumentParser(description="test: convert Stable Diffusion checkpoint to TensorRT engine")
	parser.add_argument("--ckpt_path", required=True)
	parser.add_argument("--output_name", help=".onnx & .trt file name, default to ckpt file name")
	parser.add_argument("--batch_min", type=int, default=1, help="default 1")
	parser.add_argument("--batch_opt", type=int, default=1, help="default 1")
	parser.add_argument("--batch_max", type=int, default=1, help="can go up to 4")
	parser.add_argument("--height_min", type=int, help="default 768 if sdxl else 512")
	parser.add_argument("--height_opt", type=int, help="default 1024 if sdxl else 512")
	parser.add_argument("--height_max", type=int, help="default 1024 if sdxl else 768")
	parser.add_argument("--width_min", type=int, help="default 768 if sdxl else 512")
	parser.add_argument("--width_opt", type=int, help="default 768 if sdxl else 512")
	parser.add_argument("--width_max", type=int, help="default 1024 if sdxl else 768")
	parser.add_argument("--token_count_min", type=int, default=75, help="default 75, cannot go lower")
	parser.add_argument("--token_count_opt", type=int, default=75, help="default 75")
	parser.add_argument("--token_count_max", type=int, default=150, help="default 150")
	parser.add_argument("--force_export", action="store_true")
	parser.add_argument("--static_shapes", action="store_true", help="may cause weird error (?) if enable")
	parser.add_argument("--float32", action="store_true")
	return parser.parse_args()


def get_config_from_checkpoint(ckpt_path: str) -> dict:
	"""see comfy/sd.py >>> load_checkpoint_guess_config"""
	tmp0 = "model.diffusion_model."
	sd = load_torch_file(ckpt_path)
	parameters = calculate_parameters(sd, tmp0)
	unet_dtype = get_unet_dtype(model_params=parameters)
	unet_config = detect_unet_config(sd, tmp0, unet_dtype)
	for model_config in LIST_MODELS:
		if model_config.matches(unet_config):
			tmp1 = model_config(unet_config)
			model = tmp1.get_model(sd, tmp0, device="cuda")
			model.load_model_weights(sd, tmp0)
			return {
				"model": model.diffusion_model,
				"baseline_model": model_config.__qualname__,
				# "comfy_model_type": tmp1.model_type(sd, tmp0),  # TODO: not reliable with SD v2
				"unet_hidden_dim": unet_config["in_channels"],
			}


if __name__ == "__main__":
	args = parseArgs()

	ckpt_config = get_config_from_checkpoint(args.ckpt_path)
	if cc_major < 7:
		args.float32 = True
		print("FP16 has been disabled because your GPU does not support it.")

	baseline_model = ckpt_config["baseline_model"]
	print(f"detected baseline model version: {baseline_model}")
	is_sdxl = baseline_model in ["SDXL", "SDXLRefiner"]

	if baseline_model in ["SD15", "SD20", "SD21UnclipL", "SD21UnclipH"]:
		if args.height_min is None: args.height_min = 512
		if args.height_opt is None: args.height_opt = 512
		if args.height_max is None: args.height_max = 768
		if args.width_min is None: args.width_min = 512
		if args.width_opt is None: args.width_opt = 512
		if args.width_max is None: args.width_max = 768
	elif baseline_model == "SDXL":
		if args.height_min is None: args.height_min = 768
		if args.height_opt is None: args.height_opt = 1024
		if args.height_max is None: args.height_max = 1024
		if args.width_min is None: args.width_min = 768
		if args.width_opt is None: args.width_opt = 1024
		if args.width_max is None: args.width_max = 1024
	elif baseline_model in ["SDXLRefiner", "SSD1B"]:
		raise ValueError(f"{baseline_model} not yet supported")
	else:
		raise ValueError("cannot detect baseline model version from ckpt")

	if args.height_min % 64 != 0 or args.height_opt % 64 != 0 or args.height_max % 64 != 0 or args.width_min % 64 != 0 or args.width_opt % 64 != 0 or args.width_max % 64 != 0:
		raise ValueError("height and width must be divisible by 64")
	if not (args.height_min <= args.height_opt <= args.height_max and args.width_min <= args.width_opt <= args.width_max):
		raise ValueError("need min ≤ opt ≤ max")
	print(
		"[I] size & shape parameters:",
		f"- {args.batch_min=}, {args.batch_opt=}, {args.batch_max=}",
		f"- {args.height_min=}, {args.height_opt=}, {args.height_max=}",
		f"- {args.width_min=}, {args.width_opt=}, {args.width_max=}",
		f"- {args.token_count_min=}, {args.token_count_opt=}, {args.token_count_max=}",
		sep="\n    ", end="\n\n"
	)

	if args.output_name is None:  # default to ckpt file name
		args.output_name = os.path.splitext(os.path.basename(args.ckpt_path))[0]
	onnx_filename, onnx_path = modelmanager.get_onnx_path(args.output_name)
	print(f"Exporting {args.output_name} to TensorRT")
	timing_cache = modelmanager.get_timing_cache()

	min_textlen = (args.token_count_min // 75) * 77
	opt_textlen = (args.token_count_opt // 75) * 77
	max_textlen = (args.token_count_max // 75) * 77
	if args.static_shapes:
		min_textlen = max_textlen = opt_textlen

	if is_sdxl:
		diable_optimizations = True
		modelobj = OAIUNetXL(
			baseline_model,
			fp16=not args.float32,
			device="cuda",
			verbose=False,
			max_batch_size=args.batch_max,
			unet_dim=ckpt_config["unet_hidden_dim"],
			text_optlen=opt_textlen,
			text_maxlen=max_textlen,
		)
	else:
		diable_optimizations = False
		modelobj = OAIUNet(
			baseline_model,
			fp16=not args.float32,
			device="cuda",
			verbose=False,
			max_batch_size=args.batch_max,
			text_optlen=opt_textlen,
			text_maxlen=max_textlen,
			unet_dim=ckpt_config["unet_hidden_dim"],
		)

	profile = modelobj.get_input_profile(
		args.batch_min, args.batch_opt, args.batch_max,
		args.height_min, args.height_opt, args.height_max,
		args.width_min, args.width_opt, args.width_max,
		args.static_shapes,
	)
	print(profile)

	if not os.path.exists(onnx_path):
		print("No ONNX file found. Exporting ONNX …")
		export_onnx(
			ckpt_config["model"],
			onnx_path,
			is_sdxl=is_sdxl,
			modelobj=modelobj,
			profile=profile,
			disable_optimizations=diable_optimizations
		)
		print("Exported to ONNX.")

	trt_engine_filename, trt_path = modelmanager.get_trt_path(args.output_name, profile, args.static_shapes)

	# claim VRAM for TensorRT
	del ckpt_config["model"]
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
				inpaint=ckpt_config["unet_hidden_dim"] > 4,
				refit=True,
				vram=0,
				unet_hidden_dim=ckpt_config["unet_hidden_dim"],
				lora=False,
			)
	else:
		print("TensorRT engine found. Skipping build. You can enable Force Export in the Advanced Settings to force a rebuild if needed.")
