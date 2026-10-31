"""Prepare a new sealed host, reusing the accepted recipe without executing CUDA."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXAMPLES = ROOT.parent.parent/"ddo-examples"
BASE = EXAMPLES/".internal/candidates/splitzip-qwen-bf16-eligibility-20261030-a"


def prepare(output, native):
    sys.path.insert(0, str(HERE))
    spec=importlib.util.spec_from_file_location("owner_capture", HERE/"capture.py")
    capture=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(capture)
    baseline=json.loads((BASE/"manifest.json").read_text())
    for name, expected in baseline["files"].items():
        if capture.digest(name) != expected:
            raise ValueError("Accepted baseline source changed: " + name)
    output.mkdir(parents=True, exist_ok=False)
    for name in ("observer.py", "capture.py", "runner_tail.txt"):
        shutil.copy2(HERE/name, output/name)
    original=(BASE/"runner.py").read_text(encoding="utf-8")
    anchor="            from sampler import SampleStop, observe\n"
    if original.count(anchor) != 1:
        raise ValueError("Accepted runner splice boundary changed")
    runner=original.split(anchor)[0]+(HERE/"runner_tail.txt").read_text()
    compile(runner, str(output/"runner.py"), "exec")
    (output/"runner.py").write_text(runner, encoding="utf-8")
    package_code="import importlib.metadata as m,json; print(json.dumps({n:m.version(n) for n in "+repr(list(baseline["packages"]))+"}))"
    packages=json.loads(subprocess.check_output([baseline["python"], "-I", "-B", "-c", package_code], text=True))
    if packages != baseline["packages"]:
        raise ValueError("Baseline package versions changed")
    paths={Path(name) for name in baseline["files"]}
    paths.add(BASE/"manifest.json")
    paths.update(path for path in HERE.iterdir() if path.is_file())
    paths.update(path for path in output.iterdir() if path.is_file())
    paths.update(path for path in (ROOT/"native/include").glob("*.h"))
    paths.update(path for path in (ROOT/"native/src").glob("*.cpp"))
    paths.update([ROOT/"CMakeLists.txt", native/"pool_probe.cpp", native/"CMakeLists.txt",
                  native/"source-generation.json", native/"build.cmd", native/"build/ddo_vmm.dll"])
    generated=(native/"pool_probe.cpp").read_bytes()
    source=(ROOT/"native/src/ddo_vmm_torch_pool.cpp").read_bytes()
    if generated != source+b'\n#include "probe_tail.inc"\n':
        raise ValueError("Read-only source-generation contract violated")
    manifest=dict(schema=1, python=baseline["python"], root=baseline["root"], runner=str(output/"runner.py"),
                  args=baseline["args"], packages=packages, model=baseline["model"], custom=baseline["custom"],
                  native=str(native/"build/ddo_vmm.dll"), baseline_native=baseline["native"],
                  limits=capture.LIMITS, files={str(path):capture.digest(path) for path in sorted(paths)},
                  native_features=dict(span_pool=True, joined_spans=True, mapping_trace=False,
                                       source_dispatcher=False, read_only_owner_probe=True),
                  toolchain="MSVC 19.44.35229.0; CUDA Driver toolkit 13.2; Release; no CUDA kernels",
                  installed_artifacts_changed=False, cuda_executed_during_preparation=False)
    capture.write_json(output/"manifest.json", manifest)
    (output/"manifest.sha256").write_text(capture.digest(output/"manifest.json")+"\n")
    print(json.dumps(dict(status="prepared_without_cuda", files=len(manifest["files"]),
                          manifest_sha256=capture.digest(output/"manifest.json"))))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("native", type=Path)
    args=parser.parse_args()
    prepare(args.output.resolve(), args.native.resolve())
