"""Conventional Qwen-Image 2.1 control using official streamed leaf offload.

No DDO offload runtime, allocator or custom model blocks are installed.
Shared reporting and optional DDO timing/Windows lifecycle helpers are instrumentation.
The full pipeline keeps its components loaded; it is not phase-staged.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    from scripts.helpers.runner_reporting import add_reporting_arguments

    parser = argparse.ArgumentParser(description=__doc__)
    add_reporting_arguments(parser)
    parser.add_argument("--model-path", default="Qwen/Qwen-Image-2.1")
    parser.add_argument("--prompt", default="A capybara wearing a wizard hat, reading by candlelight, oil painting")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--vae-tiling", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/qwen21_base_diffusers_leaf_20steps")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    import diffusers
    import torch
    from diffusers import QwenImage21Pipeline
    from diffusers.hooks.group_offloading import apply_group_offloading
    from diffusers_dynamic_offloader.timing import DynamicOffloadExecutionTimer

    from scripts.helpers.inference_utils import flush
    from scripts.helpers.runner_reporting import RunnerReporting

    device = torch.device("cuda:0")
    metrics = {
        "runner": Path(__file__).stem,
        "model": args.model_path,
        "backend": "official_diffusers_leaf",
        "ddo_enabled": False,
        "diffusers_source": diffusers.__file__,
        "workload": {
            "prompt": args.prompt,
            "width": args.width,
            "height": args.height,
            "steps_requested": args.steps,
            "seed": args.seed,
            "dtype": "bfloat16",
            "true_cfg_scale": 1.0,
            "use_kv_cache": True,
            "vae_tiling": args.vae_tiling,
        },
        "offload": {
            "offload_type": "leaf_level",
            "use_stream": True,
            "record_stream": False,
            "low_cpu_mem_usage": True,
            "non_blocking": False,
        },
        "timing_scope": "Full conventional pipeline. First callback includes encoding; "
        "CUDA phase also includes decode. Host callbacks are asynchronous.",
        "events": [],
        "steps": [],
        "denoise_callbacks": [],
    }
    with RunnerReporting(args, metrics, device) as run:
        with run.phase("load_pipeline"):
            pipe = QwenImage21Pipeline.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                local_files_only=args.local_files_only,
            )
        with run.phase("attach_leaf"):
            for component in (pipe.text_encoder, pipe.transformer):
                apply_group_offloading(
                    component,
                    onload_device=device,
                    offload_device=torch.device("cpu"),
                    **metrics["offload"],
                )
            del component
            pipe.vae.to(device)
            if args.vae_tiling:
                pipe.vae.enable_tiling()
            metrics["executed_controls"] = {"vae_tiling": bool(pipe.vae.use_tiling)}
        # Do not pretend a before-pipeline purge is a before-transformer purge.
        # The one-shot lifecycle callback runs on the first transformer invocation.
        purge_done = False

        def before_transformer(module, inputs):
            nonlocal purge_done
            if not purge_done:
                run.purge_before_transformer()
                purge_done = True

        purge_handle = pipe.transformer.register_forward_pre_hook(before_transformer)
        try:
            with run.phase("generate"):
                timer = DynamicOffloadExecutionTimer(enabled=args.ddo_gpu_timing, device=device)
                previous_callback = time.perf_counter()

                def on_step_end(pipeline, index, timestep, values, execution_timer=timer):
                    nonlocal previous_callback
                    execution_timer.mark_step()
                    now = time.perf_counter()
                    elapsed = now - previous_callback
                    metrics["denoise_callbacks"].append(
                        {
                            "step": index + 1,
                            "callback_interval_sec": elapsed,
                            "scope": "encode_plus_first_step" if index == 0 else "asynchronous_host_callback",
                        }
                    )
                    previous_callback = now
                    if args.show_denoise_steps:
                        print(f"Step {index + 1}: {elapsed:.3f} s (host callback)", flush=True)
                    return values

                with timer:
                    image = pipe(
                        prompt=args.prompt,
                        height=args.height,
                        width=args.width,
                        num_inference_steps=args.steps,
                        true_cfg_scale=1.0,
                        generator=torch.Generator(device="cpu").manual_seed(args.seed),
                        use_kv_cache=True,
                        callback_on_step_end=on_step_end,
                    ).images[0]
                metrics["execution_timing"] = timer.snapshot(wait=args.ddo_gpu_timing)
                metrics["executed_steps"] = len(metrics["denoise_callbacks"])
        finally:
            purge_handle.remove()
        with run.phase("cleanup_pipeline"):
            del pipe, purge_handle, timer, on_step_end, before_transformer
            flush()
        with run.phase("save_image"):
            args.output_dir.mkdir(parents=True, exist_ok=True)
            image.save(args.output_dir / "image.png")
            del image


if __name__ == "__main__":
    main()
