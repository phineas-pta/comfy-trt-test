# attempt to use TensorRT with ComfyUI

best suited for RTX 20xx-30xx-40xx

not automatic yet, do not use `ComfyUI-Manager` to install !!! read below instructions to install

instructions not beginner-friendly yet, still intended for advanced users

i only tested baseline models (SD 1.5 - 2.1 - XL - SSD-1B - Turbo work) with simplest workflow, need further testing for complex workflow

no lora support yet as i wait for better upstream update coz current way of using lora is too cumbersome (only 1 lora at strength 1.0)

very limited usefulness as no additionnal features supported in upstream (controlnet, ipadapter, hypernetwork, freeu, etc.)

this repo is a ComfyUI port from the official A1111 extension, of which the codebase is increasingly intertwined with A1111 so it’s becoming difficult for me to adapt it to ComfyUI, in case i decide to stop maintaining this repo, please check out https://github.com/gameltb/ComfyUI_stable_fast

## TODO

work-in-progress
- [x] conversion script in CLI
- [x] add new loader node
- [ ] support for LCM & SVD & Zero123
- [ ] keep model in VRAM after use
- [ ] conversion in GUI (rewrite loader node)
- [ ] re-use engine from A1111 (conversion in GUI)
- [ ] make it more automatic / user-friendly / compatible with `ComfyUI-Manager`
- [ ] lora and/or controlnet: wait upstream update
- [ ] progress bar when conversion (need tensorrt v9)
- [ ] onnx constant folding error (maybe float16 problem)

## 1️⃣ install python dependencies

i’ll add a proper `requirements.txt` when tensorrt v9 get stable release

need CUDA version ≥ 11 (driver version > 450) & python version ≥ 3.10 & ComfyUI version later than commit `b3b5ddb`

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

build engine with default profile: `python convert_unet.py --ckpt_path <checkpoint file path>`  (SD 1.5 - 2.1 - XL - SSD-1B - Turbo work)

⚠️ LCM & SVD not supported yet

for more options: see `python convert_unet.py --help`

example with 8 GB VRAM + SD 1.5 or 2.1: `--batch_max 4 --height_min 256 --width_min 256`, SDXL batch size 1

1 model can be converted multiple times with different profiles (but take lot more disk space)

may take roughly 10’ (SD 1.5 & 2.1) up to 30’ (SDXL) — no progress bar like A1111 yet

for now no LoRA nor ControlNet support yet

⚠️ for now incompatible with TensorRT engines converted in A1111

engine files created in `comfy-trt-test/comfy_trt/Unet-trt/`

## 3️⃣ usage in ComfyUI

add node → “advanced” → “loaders” → “load Unet in TensorRT” → replace “model” in normal checkpoint loader when connect to KSampler

need to get CLIP & VAE from appropriate checkpoint, or load separately CLIP & VAE

⚠️ this node doesn’t respect any comfy settings about vram/memory

⚠️ model auto unload after use, cannot move between RAM - VRAM like normal model

## 🗿 frequently seen error messages

when convert checkpoint to tensorrt engine, those messages below are not critical if engine can be created:
```
███\comfy\ldm\modules\diffusionmodules\openaimodel.py:849: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  assert y.shape[0] == x.shape[0]
███\comfy\ldm\modules\diffusionmodules\openaimodel.py:125: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  assert x.shape[1] == self.channels
███\comfy\ldm\modules\diffusionmodules\openaimodel.py:83: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  assert x.shape[1] == self.channels
███\Lib\site-packages\torch\onnx\utils.py:737: UserWarning: We detected that you are modifying a dictionary that is an input to your model. Note that dictionaries are allowed as inputs in ONNX but they should be handled with care. Usages of dictionaries is not recommended, and should not be used except for configuration use. Also note that the order and values of the keys must remain the same.
  warnings.warn(warning)

[W:onnxruntime:, constant_folding.cc:212 onnxruntime::ConstantFolding::ApplyImpl] Could not find a CPU kernel and hence can't constant fold Sqrt node …

[libprotobuf WARNING ***\externals\protobuf\3.0.0\src\google\protobuf\io\coded_stream.cc:604] Reading dangerously large protocol message. If the message turns out to be larger than 2147483647 bytes, parsing will be halted for security reasons. To increase the limit (or to disable these warnings), see CodedInputStream::SetTotalBytesLimit() in google/protobuf/io/coded_stream.h.
[libprotobuf WARNING ***\externals\protobuf\3.0.0\src\google\protobuf\io\coded_stream.cc:81] The total number of bytes read was …

[E] 2: [virtualMemoryBuffer.cpp::nvinfer1::StdVirtualMemoryBufferImpl::resizePhysical::140] Error Code 2: OutOfMemory (no further information)

[E] 1: [myelinCache.h::nvinfer1::builder::MyelinCache::deserializeFromBuffer::52] Error Code 1: Myelin (Myelin error from unknown graph)

[E] 1: [stdArchiveReader.cpp::nvinfer1::rt::StdArchiveReader::StdArchiveReaderInitCommon::46] Error Code 1: Serialization (Serialization assertion stdVersionRead == serializationVersion failed.Version tag does not match. Note: Current Version: ███, Serialized Engine Version: ███)

[E] 4: The timing cache will not be used!
```

when using in ComfyUI, if prompt too short (need at least 75-77 tokens) then this error below is shown, but things still work anyway:
```
[E] 3: [engine.cpp::nvinfer1::rt::Engine::getProfileDimensions::1127] Error Code 3: API Usage Error (Parameter check failed at: engine.cpp::nvinfer1::rt::Engine::getProfileDimensions::1127, condition: bindingIsInput(bindingIndex))
```

if see this error below then restart python session
```
AttributeError: 'TrtUnet' object has no attribute 'engine'
```

if error `No valid profile found` meaning engine built with incompatible image resolution (need re-build) or CLIP from checkpoint not correct (select proper checkpoint)

## 📑 appendix

original implementation:
- https://github.com/NVIDIA/Stable-Diffusion-WebUI-TensorRT (upstream)
- https://github.com/NVIDIA/TensorRT/tree/release/9.1/demo/Diffusion
- https://nvidia.custhelp.com/app/answers/detail/a_id/5487/~/tensorrt-extension-for-stable-diffusion-web-ui
- https://nvidia.custhelp.com/app/answers/detail/a_id/5490/~/system-memory-fallback-for-stable-diffusion

how to write node:
- 1 node with all (unet + sampler) like https://github.com/0xbitches/ComfyUI-LCM/blob/main/nodes.py
- unet CoreML like https://github.com/aszc-dev/ComfyUI-CoreMLSuite/blob/main/coreml_suite/models.py
- unet loader like https://github.com/city96/ComfyUI_ExtraModels/blob/main/DiT/loader.py
- unet AITemplate like https://github.com/FizzleDorf/ComfyUI-AIT/blob/main/ait_load.py

original checkpoints:
- SD 1.4: https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/blob/main/sd-v1-4.ckpt
- SD 1.5: https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.safetensors
- SD 2.0: https://huggingface.co/stabilityai/stable-diffusion-2/blob/main/768-v-ema.safetensors
- SD 2.1: https://huggingface.co/stabilityai/stable-diffusion-2-1/blob/main/v2-1_768-ema-pruned.safetensors
- SDXL base: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors
- SDXL refiner: https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/blob/main/sd_xl_refiner_1.0.safetensors
- SVD: https://huggingface.co/stabilityai/stable-video-diffusion-img2vid/blob/main/svd.safetensors
- SVD xt: https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/blob/main/svd_xt.safetensors
- SD turbo: https://huggingface.co/stabilityai/sd-turbo/blob/main/sd_turbo.safetensors
- SDXL turbo: https://huggingface.co/stabilityai/sdxl-turbo/blob/main/sd_xl_turbo_1.0_fp16.safetensors

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
