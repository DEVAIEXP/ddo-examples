# Qwen Image 2.1 SDNQ INT4

Use `scripts/qwen_image_21/run_dynamic_modular_sdnq.py` with the
[SDNQ optional dependencies](setup.md). No FP8 extra or native allocator is required.

The saved transformer is
[OzzyGT/Qwen_Image_2_1_sdnq_dynamic_4bit](https://huggingface.co/OzzyGT/Qwen_Image_2_1_sdnq_dynamic_4bit).
Auxiliary components come from `Qwen/Qwen-Image-2.1` in BF16.

## Run the model

```powershell
python -B scripts/qwen_image_21/run_dynamic_modular_sdnq.py --variant base --width 512 --height 512 --steps 20 --seed 42 --ddo-preset one_shot_fast --vae-resident --purge-before-transformer --memory-monitor rss --show-metrics --save-metrics
```

Run elevated on Windows for standby purge. To compare official streamed leaf,
change only `--ddo-preset` to `diffusers_leaf_offload_compat`.
The selected Windows INT4 comparison is in
[Qwen benchmarks](../benchmarks/qwen-image-21.md).
Turbo, INT8 and Linux/WSL INT4 are not established by that comparison.

For common arguments, metrics and output paths, see
[shared controls](runner-controls.md). For model controls, see
[Qwen parameters](qwen-image-21-parameters.md).
