# comfy-trt-test

failed attempt to use TensorRT with ComfyUI

**NOT WORKING YET**

need RTX 20xx-30xx-40xx

**TODO**:
- [ ] conversion script in CLI
- [ ] add new loader node
- [ ] conversion in GUI
- [ ] re-use engine from a1111
- [ ] lora ? controlnet ?

```
pip install onnx
pip install onnx-graphsurgeon polygraphy --extra-index-url https://pypi.ngc.nvidia.com
pip install --pre tensorrt==9.0.1.post11.dev4 --extra-index-url https://pypi.nvidia.com --no-cache-dir

cd custom_nodes/comfy-trt-test
python convert_unet.py --help
```

**ISSUES**: find out how comfy load model, require `.forward()` method for onnx export

reference: https://github.com/NVIDIA/Stable-Diffusion-WebUI-TensorRT

inspirations for implementation:
- https://github.com/aszc-dev/ComfyUI-CoreMLSuite
- https://github.com/0xbitches/ComfyUI-LCM
