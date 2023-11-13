# attempt to use TensorRT with ComfyUI

best suited for RTX 20xx-30xx-40xx

not automatic yet, do not use `ComfyUI-Manager` to install !!!

not beginner-friendly yet, still intended to technical users

i only tested baseline models, need further testing for inpaint or complex workflow

## TODO

- [x] conversion script in CLI
- [x] add new loader node
- [ ] unload model when not in use
- [ ] conversion in GUI (rewrite loader node)
- [ ] re-use engine from A1111 (conversion in GUI)
- [ ] make it more automatic / user-friendly / compatible with `ComfyUI-Manager`
- [ ] lora and/or controlnet: wait upstream update
- [ ] progress bar when conversion (need tensorrt v9)
- [ ] onnx constant folding error (maybe float16 problem)

## 1️⃣ install python dependencies

need ComfyUI version later than commit `f12ec55`

open ComfyUI python env
```
pip install colored onnx
pip install onnx-graphsurgeon polygraphy --extra-index-url https://pypi.ngc.nvidia.com
```

install TensorRT:
- on linux: `pip install tensorrt`
- on windows: follow my guide to install TensorRT & python wheel: https://github.com/phineas-pta/NVIDIA-win/blob/main/NVIDIA-win.md
- alternatively, use the pre-release version: `pip install --pre tensorrt==9.0.1.post11.dev4 --extra-index-url https://pypi.nvidia.com --no-cache-dir`

navigate console to folder `custom_nodes/` and git clone this repo

on windows need additional steps:
```batchfile
cd comfy-trt-test\comfy_trt\timing_caches
git update-index --skip-worktree timing_cache_win_cc75.cache
git update-index --skip-worktree timing_cache_win_cc86.cache
git update-index --skip-worktree timing_cache_win_cc89.cache
```
on windows + cuda < 12.3 also do `set CUDA_MODULE_LOADING=LAZY` can prevent some weird errors

## 2️⃣ convert checkpoint to tensorrt engine

navigate console to folder `comfy-trt-test/`

convert with default profile: `python convert_unet.py --ckpt_path <checkpoint file path>`

for more options: see `python convert_unet.py --help`

example with 8 GB VRAM + SD 1.5 or 2.1: `--batch_max 4 --height_min 256 --width_min 256`

may take roughly 10’ (SD 1.5 & 2.1) up to 30’ (SDXL) — no progress bar like A1111 yet

SDXL need at least 8 GB VRAM to convert (no refiner yet)

for now no LoRA nor ControlNet support yet

⚠️ for now incompatible with TensorRT engines converted in A1111

engine files created in `comfy-trt-test/comfy_trt/Unet-trt/`

## 3️⃣ usage in ComfyUI

add node → “advanced” → “loaders” → “load Unet in TensorRT” → replace “model” in normal checkpoint loader when connect to KSampler

need to get CLIP & VAE from according checkpoint

how to select model type:
- SD 1.5: select `EPS`
- SD 2.1: select `V_PREDICTION`, but if rubbish result image or inpaint then select `EPS`
- SDXL: select `EPS`, but if rubbish result image or inpaint then select `V_PREDICTION`

⚠️ VRAM not released when node not in use, need restart python session to clear

## 🗿 frequently seen error messages

when convert checkpoint to tensorrt engine, those messages below are not critical if engine can be created:
```
[W:onnxruntime:, constant_folding.cc:212 onnxruntime::ConstantFolding::ApplyImpl] Could not find a CPU kernel and hence can't constant fold Sqrt node …
[libprotobuf WARNING ***\externals\protobuf\3.0.0\src\google\protobuf\io\coded_stream.cc:604] Reading dangerously large protocol message. If the message turns out to be larger than 2147483647 bytes, parsing will be halted for security reasons. To increase the limit (or to disable these warnings), see CodedInputStream::SetTotalBytesLimit() in google/protobuf/io/coded_stream.h.
[libprotobuf WARNING ***\externals\protobuf\3.0.0\src\google\protobuf\io\coded_stream.cc:81] The total number of bytes read was …
[E] 2: [virtualMemoryBuffer.cpp::nvinfer1::StdVirtualMemoryBufferImpl::resizePhysical::140] Error Code 2: OutOfMemory (no further information)
```

when using in ComfyUI, if prompt too short (need at least 75-77 tokens) then this message below is shown:
```
[E] 3: [engine.cpp::nvinfer1::rt::Engine::getProfileDimensions::1127] Error Code 3: API Usage Error (Parameter check failed at: engine.cpp::nvinfer1::rt::Engine::getProfileDimensions::1127, condition: bindingIsInput(bindingIndex))
```

if see this message below then restart python session
```
AttributeError: 'TrtUnet' object has no attribute 'engine'
```

## 📑 appendix

original implementation:
- https://github.com/NVIDIA/Stable-Diffusion-WebUI-TensorRT
- https://nvidia.custhelp.com/app/answers/detail/a_id/5487/~/tensorrt-extension-for-stable-diffusion-web-ui
- https://nvidia.custhelp.com/app/answers/detail/a_id/5490/~/system-memory-fallback-for-stable-diffusion

how to write node:
- 1 node with all (unet + sampler) like https://github.com/0xbitches/ComfyUI-LCM/blob/main/nodes.py
- unet CoreML like https://github.com/aszc-dev/ComfyUI-CoreMLSuite/blob/main/coreml_suite/models.py
- unet loader like https://github.com/city96/ComfyUI_ExtraModels/blob/main/DiT/loader.py
- unet AITemplate like https://github.com/FizzleDorf/ComfyUI-AIT/blob/main/ait_load.py

where to download separated CLIP & VAE without checkpoint:
- CLIP:
  - https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/text_encoder/model.fp16.safetensors
  - https://huggingface.co/stabilityai/stable-diffusion-2-1/blob/main/text_encoder/model.fp16.safetensors
  - https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/text_encoder/model.fp16.safetensors (base)
  - https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/text_encoder_2/model.fp16.safetensors (refiner)
- VAE:
  - https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/vae/diffusion_pytorch_model.fp16.safetensors
  - https://huggingface.co/stabilityai/stable-diffusion-2-1/blob/main/vae/diffusion_pytorch_model.fp16.safetensors
  - https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/vae/diffusion_pytorch_model.fp16.safetensors
  - https://huggingface.co/stabilityai/sdxl-vae/blob/main/sdxl_vae.safetensors (float32)
  - https://huggingface.co/stabilityai/sd-vae-ft-ema-original/blob/main/vae-ft-ema-560000-ema-pruned.safetensors (float32)
  - https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors (float32)
