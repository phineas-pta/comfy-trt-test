import sys
import os.path
import gc
import torch
import argparse

from comfy_trt.exporter import export_onnx, export_trt
from comfy_trt.utilities import PIPELINE_TYPE
from comfy_trt.models import make_OAIUNetXL, make_OAIUNet
from comfy_trt.model_manager import modelmanager, cc_major

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from comfy.utils import load_torch_file, calculate_parameters
from comfy.supported_models import models as list_models
from comfy.model_detection import detect_unet_config
from comfy.model_management import unet_dtype


def parseArgs():
	parser = argparse.ArgumentParser(description="test: convert Stable Diffusion checkpoint to TensorRT engine")
	parser.add_argument("--ckpt_path")
	parser.add_argument("--batch_min", type=int, default=1)
	parser.add_argument("--batch_opt", type=int, default=1)
	parser.add_argument("--batch_max", type=int, default=4)
	parser.add_argument("--height_min", type=int, default=512)
	parser.add_argument("--height_opt", type=int, default=768)
	parser.add_argument("--height_max", type=int, default=1024)
	parser.add_argument("--width_min", type=int, default=512)
	parser.add_argument("--width_opt", type=int, default=768)
	parser.add_argument("--width_max", type=int, default=1024)
	parser.add_argument("--token_count_min", type=int, default=75)
	parser.add_argument("--token_count_opt", type=int, default=75)
	parser.add_argument("--token_count_max", type=int, default=150)
	parser.add_argument("--force_export", action="store_true")
	parser.add_argument("--static_shapes", action="store_true")
	parser.add_argument("--float32", action="store_true")
	return parser.parse_args()


def get_config_from_checkpoint(ckpt_path: str) -> dict:
	sd = load_torch_file(ckpt_path)
	parameters = calculate_parameters(sd, "model.diffusion_model.")
	unet_dtype = unet_dtype(model_params=parameters)
	unet_config = detect_unet_config(sd, "model.diffusion_model.", unet_dtype)
	for model_config in list_models:
		if model_config.matches(unet_config):
			return {
				"state_dict": sd,
				"sd_ver": model_config.__class__.__name__,
				"unet_hidden_dim": unet_config["in_channels"]
			}


if __name__ == "__main__":
	args = parseArgs()
	ckpt_config = get_config_from_checkpoint(args.ckpt_path)
	if cc_major < 7:
		args.float32 = True
		print("FP16 has been disabled because your GPU does not support it.")

	model_name = os.path.splitext(os.path.basename(args.ckpt_path))[0]
	onnx_filename, onnx_path = modelmanager.get_onnx_path(model_name)

	print(f"Exporting {model_name} to TensorRT")

	timing_cache = modelmanager.get_timing_cache()

	version = ckpt_config["sd_ver"]

	pipeline = PIPELINE_TYPE.TXT2IMG
	if ckpt_config["unet_hidden_dim"] > 4:
		pipeline = PIPELINE_TYPE.INPAINT

	min_textlen = (args.token_count_min // 75) * 77
	opt_textlen = (args.token_count_opt // 75) * 77
	max_textlen = (args.token_count_max // 75) * 77
	if args.static_shapes:
		min_textlen = max_textlen = opt_textlen

	if version in ["SDXLRefiner", "SDXL"]:
		pipeline = PIPELINE_TYPE.SD_XL_BASE
		modelobj = make_OAIUNetXL(version, pipeline, "cuda", False, args.batch_max, opt_textlen, max_textlen)
		diable_optimizations = True
	else:
		modelobj = make_OAIUNet(
			version,
			pipeline,
			"cuda",
			False,
			args.batch_max,
			opt_textlen,
			max_textlen,
			None,
		)
		diable_optimizations = False

	profile = modelobj.get_input_profile(
		args.batch_min,
		args.batch_opt,
		args.batch_max,
		args.height_min,
		args.height_opt,
		args.height_max,
		args.width_min,
		args.width_opt,
		args.width_max,
		args.static_shapes,
	)
	print(profile)

	if not os.path.exists(onnx_path):
		print("No ONNX file found. Exporting ONNX...")
		export_onnx(
			ckpt_config["state_dict"],
			onnx_path,
			is_sdxl=version in ["SDXLRefiner", "SDXL"],
			modelobj=modelobj,
			profile=profile,
			diable_optimizations=diable_optimizations
		)
		print("Exported to ONNX.")

	trt_engine_filename, trt_path = modelmanager.get_trt_path(model_name, profile, args.static_shapes)

	if not os.path.exists(trt_path) or args.force_export:
		print("Building TensorRT engine... This can take a while, please check the progress in the terminal.")
		gc.collect()
		torch.cuda.empty_cache()
		ret = export_trt(trt_path, onnx_path, timing_cache, profile=profile, use_fp16=not args.float32)
		if ret:
			print("## Export Failed due to unknown reason. See shell for more information.")
			
		else:
			print("TensorRT engines has been saved to disk.")
			modelmanager.add_entry(
				model_name,
				profile,
				args.static_shapes,
				fp32=args.float32,
				inpaint=ckpt_config["unet_hidden_dim"] > 4,
				refit=True,
				vram=0,
				unet_hidden_dim=ckpt_config["unet_hidden_dim"],
				lora=False,
			)
	else:
		print("TensorRT engine found. Skipping build. You can enable Force Export in the Advanced Settings to force a rebuild if needed.")
