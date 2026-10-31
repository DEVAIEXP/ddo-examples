"""CPU-only real Torch operations with injected native owner snapshots."""
import gc
import json
from pathlib import Path
import struct
import sys
import tempfile
import threading
import unittest
import weakref

HERE=Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from observer import Observer, UNKNOWN, READ, WRITE, OUTPUT
import capture
import torch


class Query:
    def __init__(self):
        self.generation=1
        self.available=True
        self.calls=0

    def __call__(self, address, size):
        self.calls+=1
        row=[UNKNOWN]*32
        row[:4]=[1,int(self.available),address,size]
        row[4:24]=[address,self.generation,3,size,0,size,0,123,0,size,size,1,0,0,0,size,size,0,0,0]
        row[27:31]=[size,size,1,0]
        return row


class Tests(unittest.TestCase):
    def observer(self, tensor, query=None):
        query=query or Query()
        observer=Observer(query, lambda:(threading.get_ident(),0,77), shape=tensor.shape, size=tensor.numel()*2)
        observer.block=0
        observer.select(tensor)
        return observer,query

    def test_reads_views_and_writes(self):
        tensor=torch.ones(64,dtype=torch.bfloat16)
        observer,query=self.observer(tensor)
        with observer:
            first=tensor.clone()
            view=tensor[2:]
            tensor.add_(1)
            second=view.clone()
        self.assertTrue(torch.all(first==1))
        self.assertTrue(torch.all(second==2))
        self.assertEqual(observer.epoch,1)
        self.assertTrue(any(row[3]&WRITE for row in observer.records))
        self.assertTrue(any(row[10]>0 for row in observer.records))
        self.assertEqual(query.calls,len(observer.records))
        self.assertTrue(observer.summary()["structural_exclusive_observed"])
        self.assertFalse(observer.summary()["completion_proven"])

    def test_no_tensor_retention(self):
        tensor=torch.ones(64,dtype=torch.bfloat16)
        reference=weakref.ref(tensor)
        observer,_=self.observer(tensor)
        with observer:
            output=tensor.clone()
        del tensor,output
        gc.collect()
        self.assertIsNone(reference())

    def test_changed_generation_refused(self):
        tensor=torch.ones(64,dtype=torch.bfloat16)
        observer,query=self.observer(tensor)
        query.generation=2
        with self.assertRaisesRegex(RuntimeError,"identity changed"),observer:
            tensor.clone()

    def test_unavailable_owner_refused(self):
        tensor=torch.ones(64,dtype=torch.bfloat16)
        query=Query(); query.available=False
        observer=Observer(query,lambda:(threading.get_ident(),0,77),shape=tensor.shape,size=128)
        observer.block=0
        with self.assertRaisesRegex(RuntimeError,"unavailable"):
            observer.select(tensor)

    def test_wrong_geometry_refused(self):
        tensor=torch.ones(64,dtype=torch.bfloat16)
        observer=Observer(Query(),lambda:(threading.get_ident(),0,77))
        with self.assertRaisesRegex(RuntimeError,"geometry"):
            observer.select(tensor)

    def test_limits_and_context(self):
        tensor=torch.ones(64,dtype=torch.bfloat16)
        observer,_=self.observer(tensor)
        with self.assertRaisesRegex(RuntimeError,"ceiling"),observer:
            for _ in range(64):
                tensor.clone()
        observer,_=self.observer(tensor)
        observer.operations=65536
        with self.assertRaisesRegex(RuntimeError,"Operation ceiling"),observer:
            tensor.clone()
        observer,_=self.observer(tensor)
        observer.context=lambda:(0,0,77)
        with self.assertRaisesRegex(RuntimeError,"changed"),observer:
            tensor.clone()

    def test_bounded_serialization_and_censored_gaps(self):
        tensor=torch.ones(64,dtype=torch.bfloat16)
        observer,_=self.observer(tensor)
        with observer:
            tensor.clone()
            torch.ones(1)
            tensor.clone()
        observer.blocks=[0,1]
        with tempfile.TemporaryDirectory() as directory:
            observer.save(directory)
            root=Path(directory)
            self.assertEqual((root/"accesses.bin").stat().st_size,len(observer.records)*128)
            self.assertEqual((root/"owners.bin").stat().st_size,len(observer.owners)*256)
            summary=json.loads((root/"observation.json").read_text())
            self.assertTrue(summary["stop_censors_future_access"])
            self.assertTrue(summary["gaps"])
            self.assertLess((root/"observation.json").stat().st_size,262144)

    def test_native_bridge_is_read_only(self):
        source=(HERE/"probe_tail.inc").read_text()
        for forbidden in ("cuEvent", "cuMem", "cuCtx", "reap(", "retire(", ".reserve(", "region_evict("):
            self.assertNotIn(forbidden,source)
        self.assertIn("std::scoped_lock lock(pool_mutex)",source)
        self.assertIn("++queries > 64",source)

    def test_bootstrap_and_controller_without_cuda(self):
        tail=(HERE/"runner_tail.txt").read_text()
        self.assertIn("except ObservationStop",tail)
        self.assertNotIn("synchronize(",tail)
        self.assertNotIn("torch.cuda.Event",tail)
        self.assertFalse(torch.cuda.is_initialized())

    def test_controller_rejects_corrupt_receipts(self):
        tensor=torch.ones(64,dtype=torch.bfloat16)
        observer,_=self.observer(tensor)
        observer.operations=1
        observer.blocks=[0,1]
        summary=observer.summary()
        summary.update(expected_stop=True,cleanup_returned=True,completed_steps=0,decode_executed=False)
        summary["origin"]["shape"]=[1,4120,4096]
        access=list(observer.records[0]); access[12]=33751040
        owner=list(observer.owners[0]); owner[3]=33751040
        metric=dict(status="completed",cleanup_errors=[],dynamic_offload_reports={"transformer":dict(
            configuration=dict(dynamic_offload_vmm_execution_backing_reclaim=True),pool_failures=0)})
        access_raw=struct.pack("<16Q",*access)
        owner_raw=struct.pack("<32Q",*owner)
        capture.validate_observation(summary,access_raw,owner_raw,metric)
        with self.assertRaisesRegex(ValueError,"extent"):
            capture.validate_observation(summary,access_raw[:-1],owner_raw,metric)
        owner[5]+=1
        with self.assertRaisesRegex(ValueError,"identity"):
            capture.validate_observation(summary,access_raw,struct.pack("<32Q",*owner),metric)
        owner[5]-=1; owner[23]=1
        with self.assertRaisesRegex(ValueError,"failure"):
            capture.validate_observation(summary,access_raw,struct.pack("<32Q",*owner),metric)

    def test_source_generation_and_seal_reject_changes(self):
        native=HERE.parents[2]/".internal/candidates/splitzip-qwen-owner-access-native-20261030-a"
        source=HERE.parents[2]/"native/src/ddo_vmm_torch_pool.cpp"
        self.assertEqual((native/"pool_probe.cpp").read_bytes(),source.read_bytes()+b'\n#include "probe_tail.inc"\n')
        old=capture.HERE
        try:
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                payload=root/"payload"; payload.write_text("before")
                manifest=dict(limits=capture.LIMITS,files={str(payload):capture.digest(payload)})
                (root/"manifest.json").write_text(json.dumps(manifest))
                (root/"manifest.sha256").write_text(capture.digest(root/"manifest.json"))
                capture.HERE=root
                payload.write_text("after")
                with self.assertRaisesRegex(ValueError,"Sealed file changed"):
                    capture.validate()
        finally:
            capture.HERE=old


if __name__ == "__main__":
    unittest.main()
