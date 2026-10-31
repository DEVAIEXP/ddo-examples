"""Bounded metadata-only observation of one declared storage, without retention."""
import ctypes
import json
import struct
import threading
import time
from pathlib import Path
import torch
from torch.utils._python_dispatch import TorchDispatchMode

UNKNOWN = (1 << 64)-1
ACCESS_FIELDS = ["operation", "operator", "phase", "roles", "epoch", "block", "stream",
                 "entry_ns", "return_ns", "aliases", "minimum_offset", "maximum_offset",
                 "storage_bytes", "query_index", "flags", "reserved"]
OWNER_FIELDS = ["schema", "found", "canonical", "query_bytes", "backing", "generation", "lease",
                "requested", "offset", "lease_bytes", "device", "context", "retiring", "backing_va",
                "backing_physical", "live_count", "reserved_count", "pending", "quarantined",
                "leased", "requested_total", "available", "dependencies", "failures", "completion",
                "version", "role", "pool_physical", "pool_va", "pool_live", "pool_quarantined", "reserved"]
READ, WRITE, OUTPUT, UNCERTAIN = 1, 2, 4, 16


class ObservationStop(Exception):
    """Expected stop after block 1; not a completed step."""


def leaves(value, depth=0, counter=None):
    if counter is None:
        counter = [0]
    counter[0] += 1
    if counter[0] > 1024 or depth > 16:
        raise RuntimeError("Operator container extent exceeded")
    if isinstance(value, torch.Tensor):
        yield value
    elif type(value) in (tuple, list):
        for child in value:
            yield from leaves(child, depth+1, counter)
    elif type(value) is dict:
        for child in value.values():
            yield from leaves(child, depth+1, counter)


class Native:
    def __init__(self, expected):
        from ddo_vmm_allocator.torch_allocator import installed_library_path
        actual = installed_library_path()
        if actual is None or actual.resolve() != Path(expected).resolve():
            raise RuntimeError("Unexpected installed allocator path")
        self.library = ctypes.CDLL(str(actual))
        self.call = self.library.sz_owner_access_probe
        self.call.argtypes = [ctypes.c_uint64, ctypes.c_uint64,
                              ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        self.call.restype = ctypes.c_int

    def __call__(self, address, size):
        row = (ctypes.c_uint64 * 32)()
        if self.call(address, size, row, 32):
            raise RuntimeError("Native owner probe refused")
        return list(row)


class Observer(TorchDispatchMode):
    def __init__(self, query, context, *, shape=(1,4120,4096), size=33751040, deadline=None):
        super().__init__()
        self.query, self.context = query, context
        self.shape, self.size = tuple(shape), size
        self.deadline = deadline or time.monotonic()+1800
        self.origin = None
        self.identity = None
        self.execution = None
        self.operations = self.epoch = 0
        self.block = -1
        self.blocks = []
        self.labels = []
        self.records, self.owners = [], []
        self.failure = None
        self.status = "not_selected"
        self.target = None

    def check(self):
        if time.monotonic() >= self.deadline:
            raise RuntimeError("Observation deadline")
        if self.execution is not None and self.context() != self.execution:
            raise RuntimeError("Observed thread/device/stream changed")

    def label(self, operator):
        if operator not in self.labels:
            if len(self.labels) >= 256 or sum(len(s.encode()) for s in self.labels)+len(operator.encode()) > 16384:
                raise RuntimeError("Operator label ceiling")
            self.labels.append(operator)
        return self.labels.index(operator)

    def select(self, output):
        self.check()
        tensors = list(leaves(output))
        if self.target is not None or len(tensors) != 1:
            raise RuntimeError("Exactly one first-block output required")
        tensor = tensors[0]
        storage = tensor.untyped_storage()
        if (tensor.dtype != torch.bfloat16 or not tensor.is_contiguous() or tuple(tensor.shape) != self.shape
                or tensor.storage_offset() != 0 or tensor.numel()*2 != self.size or storage.nbytes() != self.size):
            raise RuntimeError("Selected Qwen activation geometry changed")
        self.execution = self.context()
        self.target = (str(tensor.device), storage.data_ptr(), storage.nbytes())
        self.origin = dict(producer="transformer_blocks.0", dtype=str(tensor.dtype),
                           shape=list(tensor.shape), stride=list(tensor.stride()), storage_bytes=self.size,
                           device=str(tensor.device), canonical=storage.data_ptr())
        self.status = "recorded"
        now = time.monotonic_ns()
        self.record([tensor], self.label("producer.transformer_blocks.0"), 0, OUTPUT, now, now)

    def matching(self, values):
        if self.target is None:
            return []
        result = []
        for tensor in leaves(values):
            if str(tensor.device) != self.target[0]:
                continue
            storage = tensor.untyped_storage()
            if storage.data_ptr() == self.target[1]:
                if storage.nbytes() != self.target[2]:
                    raise RuntimeError("Target storage extent changed")
                result.append(tensor)
        return result

    def record(self, tensors, label, phase, roles, entry, returned):
        if not tensors:
            return
        self.check()
        if len(self.records) >= 64 or len(self.owners) >= 64:
            raise RuntimeError("Target access/query ceiling")
        row = self.query(self.target[1], self.target[2])
        if len(row) != 32:
            raise RuntimeError("Native owner row extent")
        self.owners.append(list(row))
        if row[0] != 1 or row[1] != 1:
            raise RuntimeError("Native owner unavailable")
        if (row[2:4] != list(self.target[1:]) or row[7] < self.size or row[23] or row[30]
                or row[4]+row[8] != row[2] or row[9] < row[7] or row[10] != self.execution[1]):
            raise RuntimeError("Native owner extent/failure mismatch")
        identity = tuple(row[i] for i in (2,4,5,6,10,11))
        if self.identity is not None and identity != self.identity:
            raise RuntimeError("Native owner identity changed")
        self.identity = identity
        offsets = [tensor.storage_offset()*tensor.element_size() for tensor in tensors]
        self.records.append([self.operations,label,phase,roles,self.epoch,self.block,self.execution[2],
                             entry,returned,len(tensors),min(offsets),max(offsets),self.size,
                             len(self.owners)-1,0,0])

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        self.check()
        self.operations += 1
        if self.operations > 65536:
            raise RuntimeError("Operation ceiling")
        label = self.label(str(function))
        kwargs = kwargs or {}
        matched, roles = [], 0
        schema = getattr(function, "_schema", None)
        if schema is None:
            matched = self.matching((args, kwargs))
            roles = UNCERTAIN
        else:
            for index, argument in enumerate(schema.arguments):
                value = args[index] if index < len(args) else kwargs.get(argument.name)
                targets = self.matching(value)
                if targets:
                    matched.extend(targets)
                    roles |= WRITE if argument.alias_info is not None and argument.alias_info.is_write else READ
        entry = time.monotonic_ns()
        self.record(matched, label, 1, roles, entry, entry)
        output = function(*args, **kwargs)
        returned = time.monotonic_ns()
        if matched and roles & WRITE:
            self.epoch += 1
        self.record(self.matching(output), label, 2, OUTPUT | (UNCERTAIN if schema is None else 0), entry, returned)
        # Fill input return timestamps only after successful dispatch.
        for record in reversed(self.records):
            if record[0] != self.operations:
                break
            if record[2] == 1:
                record[8] = returned
        return output

    def summary(self):
        gaps = []
        for left, right in zip(self.records, self.records[1:]):
            if right[0] > left[0]:
                gaps.append(dict(previous_operation=left[0], next_operation=right[0],
                    intervening_operations=max(0,right[0]-left[0]-1),
                    host_dispatch_gap_ns=max(0,right[7]-left[8]),
                    next_operator=self.labels[right[1]], next_roles=right[3],
                    previous_epoch=left[4], next_epoch=right[4]))
        gaps.sort(key=lambda row: row["host_dispatch_gap_ns"], reverse=True)
        owner = self.owners[0] if self.owners else None
        structural = (owner is not None and owner[8] == 0 and owner[15] == 1
                      and not any(owner[i] for i in (12,16,17,18))
                      and owner[19] == owner[9] and owner[13] == owner[14])
        uncertain = any(row[3] & UNCERTAIN for row in self.records)
        return dict(schema=1, status="incomplete" if self.failure else ("inconclusive" if uncertain else self.status),
                    failure=self.failure, blocks=self.blocks, operations=self.operations,
                    queries=len(self.owners), accesses=len(self.records), labels=self.labels,
                    origin=self.origin, identity=list(self.identity) if self.identity else None,
                    final_epoch=self.epoch, unknown_semantics=uncertain,
                    structural_exclusive_observed=structural, completion_proven=False,
                    compression_executed=False, performance_acceptance=False, placement_acceptance=False,
                    cross_step_observed=False, free_observed=False,
                    external_consumer_coverage="unproven", timing_scope="host dispatch only",
                    stop_censors_future_access=True, gaps=gaps,
                    access_fields=ACCESS_FIELDS, owner_fields=OWNER_FIELDS)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        accesses = b"".join(struct.pack("<16Q", *row) for row in self.records)
        owners = b"".join(struct.pack("<32Q", *row) for row in self.owners)
        if len(accesses)+len(owners) > 24576:
            raise RuntimeError("Binary observation ceiling")
        (directory/"accesses.bin").write_bytes(accesses)
        (directory/"owners.bin").write_bytes(owners)
        encoded = json.dumps(self.summary(), indent=2).encode()
        if len(encoded) > 262144:
            raise RuntimeError("Rendered observation ceiling")
        (directory/"observation.json").write_bytes(encoded)
