# comfy-trt-test

failed attempt to use TensorRT with ComfyUI

**NOT WORKING YET**

best optimized for RTX 20xx-30xx-40xx

not automatic yet, do not use `ComfyUI-Manager` to install !!!

not beginner-friendly yet, still intended to technical users

**TODO**:
- [x] conversion script in CLI
- [ ] add new loader node
- [ ] conversion in GUI
- [ ] make it more automatic / user-friendly / compatible with `ComfyUI-Manager`
- [ ] re-use engine from a1111
- [ ] onnx constant folding error (maybe float16 problem)
- [ ] progress bar when conversion (need tensorrt v9)
- [ ] lora & controlnet: lowest priority until they can independently compile without checkpoint

## instructions

### installation

```
pip install colored onnx
pip install onnx-graphsurgeon polygraphy --extra-index-url https://pypi.ngc.nvidia.com
```

- on linux: `pip install tensorrt`
- on windows: follow my guide to install TensorRT & python wheel: https://github.com/phineas-pta/NVIDIA-win/blob/main/NVIDIA-win.md
- alternatively, use the pre-release version: `pip install --pre tensorrt==9.0.1.post11.dev4 --extra-index-url https://pypi.nvidia.com --no-cache-dir`

navigate console to `custom_nodes/` and clone repo `git clone https://github.com/phineas-pta/comfy-trt-test`

on windows need additional steps:
```batchfile
cd comfy-trt-test\comfy_trt\timing_caches
git update-index --skip-worktree timing_cache_win_cc75.cache
git update-index --skip-worktree timing_cache_win_cc86.cache
git update-index --skip-worktree timing_cache_win_cc89.cache
```

### convert checkpoint to tensorrt engine

navigate console back to `comfy-trt-test/`

for options see `python convert_unet.py --help`

may take up to ½ h

### launch in ComfyUI

**NOT WORKING YET**

ISSUES: how to write node: still looking for possibilities:
- 1 node with all (unet + sampler) like https://github.com/0xbitches/ComfyUI-LCM
- unet loader only (need ModelPatcher) like https://github.com/aszc-dev/ComfyUI-CoreMLSuite
- unet loader without patch like https://github.com/city96/ComfyUI_ExtraModels

## appendix

reference: https://github.com/NVIDIA/Stable-Diffusion-WebUI-TensorRT

inspirations for GUI implementation:
- https://github.com/aszc-dev/ComfyUI-CoreMLSuite
- https://github.com/0xbitches/ComfyUI-LCM
- https://github.com/city96/ComfyUI_ExtraModels
