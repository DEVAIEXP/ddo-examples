"""Sealed metadata-only Qwen two-block capture, with no retry or forced cleanup."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import threading
import time
import uuid

HERE = Path(__file__).resolve().parent
LIMITS = dict(seconds=1800, progress_seconds=10, log_bytes=16777216, metadata_bytes=65536,
              rendered_bytes=262144, diagnostic_buffer_bytes=4194304, access_records=64,
              owner_queries=64, operations=65536, labels=256, label_bytes=16384,
              access_row_bytes=128, owner_values=32, binary_bytes=24576,
              isolation_minimum_bytes=33554432, exclusive_owners=64,
              exclusive_backing_bytes=2147483648, padding_bytes=134217728,
              pool_physical_bytes=4294967296, pool_va_bytes=8589934592,
              required_page_bytes=2097152, configuration_calls=2,
              snapshots=8, snapshot_values=32, output_total_bytes=41943040,
              input_bytes=33751040, original_backing_bytes=35651584,
              compact_backing_max_bytes=35651584, workspace_max_bytes=262144,
              local_physical_bytes=71565312, local_va_bytes=71303168,
              encodes=1, decodes=1, releases=1, restores=1, analyses_max=2,
              receipt_reads_max=4, receipt_bytes=16, codec_streams=1,
              additional_live_events_max=4, host_payload_work_bytes=35651584,
              verification_chunk_bytes=65536, verification_reads_max=1030)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    encoded = json.dumps(value, indent=2).encode()
    if len(encoded) > 65536:
        raise ValueError("Controller metadata ceiling")
    Path(path).write_bytes(encoded)


def validate():
    manifest_path = HERE/"manifest.json"
    if manifest_path.stat().st_size > 65536 or digest(manifest_path) != (HERE/"manifest.sha256").read_text().strip():
        raise ValueError("Manifest seal mismatch/limit")
    manifest = json.loads(manifest_path.read_text())
    if manifest["limits"] != LIMITS:
        raise ValueError("Unapproved observation limits")
    for name, expected in manifest["files"].items():
        if digest(name) != expected:
            raise ValueError("Sealed file changed: " + name)
    code = ("import importlib.metadata as m,json,sys; import ddo_vmm_bootstrap as b; "
            "p=b._discover_library(b._default_allocator_root(),b._installed_allocator_root()); "
            "assert 'torch' not in sys.modules; print(json.dumps({'library':str(p.resolve()),'packages':"
            "{n:m.version(n) for n in " + repr(list(manifest["packages"])) + "}}))")
    actual = json.loads(subprocess.check_output([manifest["python"], "-I", "-B", "-c", code], text=True))
    if actual["packages"] != manifest["packages"] or Path(actual["library"]) != Path(manifest["baseline_native"]):
        raise ValueError("Installed baseline environment changed")
    if "torch" in sys.modules:
        raise ValueError("Controller imported Torch")
    return manifest


def validate_observation(sample, access_bytes, owner_bytes, metrics):
    if (sample.get("schema") != 1 or sample.get("status") != "recorded"
            or sample.get("failure") is not None or sample.get("unknown_semantics") is not False):
        raise ValueError("Incomplete observation")
    for key, expected in dict(blocks=[0,1], expected_stop=True, cleanup_returned=True,
                              completed_steps=0, decode_executed=False, compression_executed=True,
                              completion_proven=True, performance_acceptance=False, placement_acceptance=False).items():
        if sample.get(key) != expected:
            raise ValueError("Observation receipt mismatch: " + key)
    from cycle import validate_cycle
    validate_cycle(sample.get("codec"))
    count = sample.get("accesses")
    if type(count) is not int or not 1 <= count <= 64 or sample.get("queries") != count:
        raise ValueError("Access/query count")
    if len(access_bytes) != count*128 or len(owner_bytes) != count*256 or len(access_bytes)+len(owner_bytes) > 24576:
        raise ValueError("Binary extent mismatch")
    accesses = list(struct.iter_unpack("<16Q", access_bytes))
    owners = list(struct.iter_unpack("<32Q", owner_bytes))
    identity = sample["identity"]
    for index, (access, owner) in enumerate(zip(accesses, owners)):
        if owner[:4] != (1,1,sample["origin"]["canonical"],33751040) or list(owner[i] for i in (2,4,5,6,10,11)) != identity:
            raise ValueError("Native owner identity/extent mismatch")
        if owner[23] or owner[30] or access[13] != index or access[12] != 33751040 or access[8] < access[7]:
            raise ValueError("Native failure or access receipt mismatch")
        if index and access[0] < accesses[index-1][0]:
            raise ValueError("Operation ordering")
    if sample["origin"]["shape"] != [1,4120,4096] or sample["origin"]["dtype"] != "torch.bfloat16":
        raise ValueError("Selected activation changed")
    if not 0 < sample["operations"] <= 65536 or len(sample["labels"]) > 256 or sum(len(s.encode()) for s in sample["labels"]) > 16384:
        raise ValueError("Operation/label ceiling")
    if any(row[1] >= len(sample["labels"]) for row in accesses):
        raise ValueError("Unknown operator label")
    for operation_key, label_key in (("park_operation","first_operator"),("restore_operation","next_operator")):
        matching = [row for row in accesses if row[0] == sample["codec"][operation_key] and row[2] == 1]
        if (len(matching) != 1 or matching[0][3] != 1
                or sample["labels"][matching[0][1]] != sample["codec"][label_key]):
            raise ValueError("Cycle/access correlation mismatch")
    if sample.get("final_epoch") != 0 or any(row[3] & (2|16) for row in accesses):
        raise ValueError("Activation version/semantics changed")
    if metrics.get("status") != "completed" or metrics.get("cleanup_errors"):
        raise ValueError("Runner failed cleanup")
    report = metrics.get("dynamic_offload_reports", {}).get("transformer")
    if not report or report.get("configuration", {}).get("dynamic_offload_vmm_execution_backing_reclaim") is not True:
        raise ValueError("Final native/reclaim report unavailable")
    def failures(value):
        if isinstance(value, dict):
            for name, child in value.items():
                if name.endswith(("_failures", "_quarantined_bytes")) and isinstance(child, (int,float)) and child != 0:
                    raise ValueError("Native failure counter: " + name)
                failures(child)
        elif isinstance(value, list):
            for child in value:
                failures(child)
    failures(report)


class Log:
    def __init__(self, pipe, path):
        self.pipe, self.path = pipe, path
        self.bytes = 0
        self.error = None
        self.suspicious = False
        self.thread = threading.Thread(target=self.drain, daemon=True)

    def drain(self):
        carry = b""
        try:
            with self.path.open("xb") as saved:
                while chunk := self.pipe.read1(4096):
                    remaining = LIMITS["log_bytes"]-self.bytes
                    saved.write(chunk[:max(0,remaining)])
                    self.bytes += len(chunk)
                    scanned = carry+chunk.lower()
                    self.suspicious |= any(token in scanned for token in
                        (b"ddo vmm: retaining", b"ddo allocation failure", b"standby cache purge skipped"))
                    carry = scanned[-128:]
                    if self.bytes > LIMITS["log_bytes"]:
                        raise RuntimeError("Log ceiling")
        except BaseException as error:
            self.error = repr(error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--execute-approved-gpu-gate", action="store_true")
    args = parser.parse_args()
    manifest = validate()
    if args.validate_only:
        print(json.dumps(dict(status="validated_without_cuda", files=len(manifest["files"]),
                              manifest_sha256=digest(HERE/"manifest.json"), torch_imported=False)))
        return
    if os.name != "nt" or not ctypes.windll.shell32.IsUserAnAdmin():
        raise RuntimeError("Use elevated PowerShell to preserve the accepted purge condition")
    environment = dict(os.environ)
    unexpected = [key for key in environment if key.upper().startswith("DDO")
                  and key.upper() != "DDO_VMM_EXECUTION_BACKING_RECLAIM"]
    if unexpected:
        raise RuntimeError("Unapproved DDO overrides: " + repr(unexpected))
    environment.update(DDO_VMM_LIBRARY_PATH=manifest["native"], DDO_VMM_EXECUTION_BACKING_RECLAIM="1",
                       DDO_VMM_TORCH_CACHE_MIB="512",
                       SZ_OBSERVATION_STARTED_NS=str(time.monotonic_ns()))
    output = Path(manifest["root"])/".internal/diagnostics"/("splitzip-qwen-exclusive-codec-"+uuid.uuid4().hex)
    output.mkdir(parents=True, exist_ok=False)
    command = [manifest["python"], "-I", "-u", "-B", str(HERE/"bootstrap.py"),
               *manifest["args"], "--output-dir", str(output)]
    record = dict(schema=1, status="failure", command=command, limits=LIMITS,
                  manifest_sha256=digest(HERE/"manifest.json"), retry=False, forced_kill=False,
                  observed_exit=False, observation_acceptance=False, model_acceptance=False,
                  performance_acceptance=False, placement_acceptance=False)
    print("Evidence: " + str(output), flush=True)
    child, logs = None, []
    started = time.monotonic()
    try:
        child = subprocess.Popen(command, cwd=manifest["root"], env=environment,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
        record["pid"] = child.pid
        logs = [Log(child.stdout, output/"stdout.txt"), Log(child.stderr, output/"stderr.txt")]
        for log in logs:
            log.thread.start()
        next_progress = 10
        while child.poll() is None:
            elapsed = time.monotonic()-started
            if elapsed >= 1800 or any(log.error or log.suspicious for log in logs):
                raise RuntimeError("Capture deadline/anomaly; child may run; no retry/kill")
            if elapsed >= next_progress:
                print(json.dumps(dict(event="progress", elapsed_seconds=round(elapsed,1),
                                      deadline_seconds=1800, eta_seconds=None)), flush=True)
                next_progress += 10
            time.sleep(.1)
        record.update(observed_exit=True, exit_code=child.returncode)
        for log in logs:
            log.thread.join(timeout=1)
        if child.returncode or any(log.error or log.suspicious or log.thread.is_alive() for log in logs):
            raise RuntimeError("Model observation failed; inspect preserved stderr; no retry")
        extents = {"isolation.json":65536, "observation.json":262144, "accesses.bin":8192, "owners.bin":16384, "metrics.json":1048576}
        for name, ceiling in extents.items():
            if (output/name).stat().st_size > ceiling:
                raise ValueError("Output ceiling: " + name)
        if sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) > LIMITS["output_total_bytes"]:
            raise ValueError("Total output ceiling")
        sys.path.insert(0, str(HERE))
        from isolation import validate_snapshots
        isolation = json.loads((output/"isolation.json").read_text())
        validate_snapshots(isolation, complete=True)
        sample = json.loads((output/"observation.json").read_text())
        if not sample.get("structural_exclusive_observed"):
            raise ValueError("Exclusive activation not established")
        owner_raw = (output/"owners.bin").read_bytes()
        for row in struct.iter_unpack("<32Q", owner_raw):
            if (row[31] != 1 or row[8] or row[15] != 1 or row[13] != 35651584
                    or row[14] != row[13] or any(row[i] for i in (12,16,17,18)) or row[19] != row[9]):
                raise ValueError("Exclusive owner evidence mismatch")
        record["isolation"] = isolation
        record["codec"] = sample.get("codec")
        validate_observation(sample, (output/"accesses.bin").read_bytes(),
                             (output/"owners.bin").read_bytes(), json.loads((output/"metrics.json").read_text()))
        record.update(status=sample["status"], observation_acceptance=sample["status"] == "recorded",
                      operations=sample["operations"], accesses=sample["accesses"],
                      structural_exclusive_observed=sample["structural_exclusive_observed"],
                      unknown_semantics=sample["unknown_semantics"],
                      observation_sha256=digest(output/"observation.json"))
    except BaseException as error:
        record["error"] = repr(error)
    finally:
        record.update(child_may_be_running=child is not None and child.poll() is None,
                      elapsed_seconds=round(time.monotonic()-started,3),
                      logs=[dict(path=log.path.name, bytes=log.bytes, error=log.error,
                                 suspicious=log.suspicious) for log in logs])
        write_json(output/"controller.json", record)
        print(json.dumps(record), flush=True)
    raise SystemExit(0 if record["status"] in ("recorded","inconclusive") else 1)


if __name__ == "__main__":
    main()
