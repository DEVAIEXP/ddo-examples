"""Real CPU Torch dispatch with injected native transitions; no CUDA initialization."""
import copy
import gc
import importlib.util
from pathlib import Path
import struct
import sys
import threading
import time
import unittest
import weakref

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from observer import Observer, READ, WRITE, UNCERTAIN
from cycle import Cycle, validate_cycle
from isolation import FIELDS, LABELS, validate_snapshots
import capture
import torch


class Query:
    def __init__(self):
        self.generation = 1
        self.unavailable = False

    def __call__(self,address,size):
        row = [0]*32
        row[:4] = [1,int(not self.unavailable),address,size]
        for index,value in {4:address,5:self.generation,6:1,7:size,9:size,11:123,
                            13:35651584,14:35651584,15:1,19:size,20:size,
                            24:2**64-1,25:1,27:35651584,28:35651584,29:1,31:1}.items():
            row[index] = value
        return row


class FakeIsolation:
    def __init__(self):
        self.labels = []
        self.fail = None

    def snapshot(self,label):
        self.labels.append(label)
        if self.fail == label:
            raise RuntimeError('Snapshot refused')


def fixture():
    rows = []
    for index,label in enumerate(LABELS):
        r = [0]*32
        r[:3] = [2,1,int(index>0)]
        if index:
            r[3:10] = [1,35651584,1900544,35651584,35651584,1438646272,1438646272]
        if index >= 4:
            r[19:24] = [1,0,1,1,515]
            r[24:30] = [27262976,132112,8256496,3,100,200]
        if index == 4:
            r[12:14] = r[24:26]
            r[16:18] = [1,1]
        if index >= 5:
            r[20:24] = [1,1,2,1030]
            r[27] = 4
            r[30:32] = [300,1]
        rows.append(dict(label=label,values=r))
    return dict(schema=2,fields=FIELDS,snapshots=rows)


class Tests(unittest.TestCase):
    def setup_cycle(self):
        tensor = torch.ones(64,dtype=torch.bfloat16)
        query = Query()
        observer = Observer(query,lambda:(threading.get_ident(),0,77),
                            shape=tensor.shape,size=128)
        observer.block = 0
        observer.select(tensor)
        isolation = FakeIsolation()
        calls = []
        cycle = Cycle(isolation,time.monotonic()+1800,functions=(
            lambda *args: calls.append(('park',args)) or 0,
            lambda *args: calls.append(('restore',args)) or 0, lambda:b'injected refusal'))
        observer.cycle = cycle
        return tensor,observer,query,isolation,calls

    def first_read(self,tensor,observer):
        return observer.__torch_dispatch__(torch.ops.aten.layer_norm.default,(),(tensor,[64]))

    def test_ordering_real_read_and_next_clone(self):
        tensor,observer,_,isolation,calls = self.setup_cycle()
        out = self.first_read(tensor,observer)
        self.assertTrue(torch.equal(out,torch.zeros_like(out)))
        self.assertTrue(observer.cycle.parked)
        with observer:
            unrelated = torch.ones(4)
            output = tensor.clone()
        self.assertTrue(torch.equal(output,tensor))
        self.assertEqual([c[0] for c in calls],['park','restore'])
        self.assertEqual(isolation.labels,['parked','restored'])
        validate_cycle(observer.cycle.summary())
        self.assertTrue(observer.summary()['completion_proven'])

    def test_view_restores_before_dispatch(self):
        tensor,observer,_,_,calls = self.setup_cycle()
        self.first_read(tensor,observer)
        with observer:
            view = tensor[2:]
        self.assertEqual([c[0] for c in calls],['park','restore'])
        self.assertEqual(view.untyped_storage().data_ptr(),tensor.untyped_storage().data_ptr())

    def test_mutation_refused_without_dispatch(self):
        tensor,observer,_,_,calls = self.setup_cycle()
        self.first_read(tensor,observer)
        with self.assertRaisesRegex(RuntimeError,'read-only'),observer:
            tensor.add_(1)
        self.assertTrue(torch.equal(tensor,torch.ones_like(tensor)))
        self.assertEqual(len(calls),1)
        self.assertTrue(observer.cycle.parked)

    def test_unknown_and_wrong_first_read_refused(self):
        tensor,observer,_,_,calls = self.setup_cycle()
        with self.assertRaisesRegex(RuntimeError,'read-only'):
            observer.cycle.before(observer,'unknown',UNCERTAIN)
        with self.assertRaisesRegex(RuntimeError,'First declared'),observer:
            tensor.clone()
        self.assertFalse(calls)

    def test_context_change_refused_while_parked(self):
        tensor,observer,_,_,calls = self.setup_cycle()
        self.first_read(tensor,observer)
        observer.context = lambda:(0,0,77)
        with self.assertRaisesRegex(RuntimeError,'changed'),observer:
            tensor.clone()
        self.assertEqual(len(calls),1)

    def test_native_restore_failure_retains_state(self):
        tensor,observer,_,_,calls = self.setup_cycle()
        self.first_read(tensor,observer)
        observer.cycle.restore = lambda stream: 1
        with self.assertRaisesRegex(RuntimeError,'injected refusal'),observer:
            tensor.clone()
        self.assertTrue(observer.cycle.parked)
        self.assertFalse(observer.cycle.closed)

    def test_native_park_failure_no_success_claim(self):
        tensor,observer,_,_,_ = self.setup_cycle()
        observer.cycle.park = lambda *args: 1
        with self.assertRaisesRegex(RuntimeError,'injected refusal'):
            self.first_read(tensor,observer)
        self.assertFalse(observer.cycle.started)
        self.assertFalse(observer.summary()['completion_proven'])

    def test_changed_generation_after_restore_refused(self):
        tensor,observer,query,_,_ = self.setup_cycle()
        self.first_read(tensor,observer)
        query.generation = 2
        with self.assertRaisesRegex(RuntimeError,'identity changed'),observer:
            tensor.clone()

    def test_no_tensor_retention(self):
        tensor,observer,_,_,_ = self.setup_cycle()
        reference = weakref.ref(tensor)
        output = self.first_read(tensor,observer)
        with observer:
            clone = tensor.clone()
        del tensor,output,clone
        gc.collect()
        self.assertIsNone(reference())

    def test_operation_and_deadline_limits(self):
        tensor,observer,_,_,_ = self.setup_cycle()
        observer.operations = 65536
        with self.assertRaisesRegex(RuntimeError,'Operation ceiling'),observer:
            tensor.clone()
        tensor,observer,_,_,_ = self.setup_cycle()
        observer.cycle.deadline = time.monotonic()-1
        with self.assertRaisesRegex(RuntimeError,'Cycle deadline'):
            self.first_read(tensor,observer)

    def test_snapshot_validation_and_mutations(self):
        data = fixture()
        validate_snapshots(data,complete=True)
        for index,field,value in [(4,26,0),(5,31,0),(5,12,1),(7,18,1),(4,6,2**33),
                                  (4,23,514),(5,23,1029),(4,24,35651584),(0,2,1)]:
            bad = copy.deepcopy(data)
            bad['snapshots'][index]['values'][field] = value
            with self.assertRaises(ValueError):
                validate_snapshots(bad,complete=True)
        with self.assertRaises(ValueError):
            validate_snapshots(dict(data,snapshots=data['snapshots'][:-1]),complete=True)

    def test_receipts_require_actual_cycle(self):
        for data in (None,{},dict(started=True,closed=False)):
            with self.assertRaises(ValueError):
                validate_cycle(data)

    def test_controller_receipt_and_binary_identity(self):
        tensor,observer,_,_,_ = self.setup_cycle()
        self.first_read(tensor,observer)
        with observer:
            tensor.clone()
        observer.blocks = [0,1]
        sample = observer.summary()
        sample.update(expected_stop=True,cleanup_returned=True,completed_steps=0,decode_executed=False,
                      codec=observer.cycle.summary())
        sample['origin']['shape'] = [1,4120,4096]
        accesses = copy.deepcopy(observer.records)
        owners = copy.deepcopy(observer.owners)
        for row in accesses:
            row[12] = 33751040
        for row in owners:
            row[3] = 33751040
        pack = lambda rows,n:b''.join(struct.pack('<'+str(n)+'Q',*row) for row in rows)
        metric = dict(status='completed',cleanup_errors=[],dynamic_offload_reports={'transformer':dict(
            configuration=dict(dynamic_offload_vmm_execution_backing_reclaim=True),pool_failures=0)})
        capture.validate_observation(sample,pack(accesses,16),pack(owners,32),metric)
        bad = copy.deepcopy(sample);bad['codec']['closed'] = False
        with self.assertRaises(ValueError):
            capture.validate_observation(bad,pack(accesses,16),pack(owners,32),metric)
        owners[-1][5] += 1
        with self.assertRaisesRegex(ValueError,'identity'):
            capture.validate_observation(sample,pack(accesses,16),pack(owners,32),metric)

    def test_no_cuda_and_bootstrap_order(self):
        self.assertFalse(torch.cuda.is_initialized())
        source = (HERE/'runner.py').read_text()
        self.assertLess(source.index('ensure_vmm_fast_process()'),source.index('from scripts.helpers'))
        for source in ((HERE/'cycle.py').read_text(),(HERE/'observer.py').read_text()):
            self.assertNotIn('torch.cuda.synchronize',source)
            self.assertNotIn('torch.cuda.Event',source)
        bootstrap = (HERE/'bootstrap.py').read_text()
        self.assertLess(bootstrap.index('os.add_dll_directory'),bootstrap.index('install_pytorch_vmm_allocator('))
        self.assertLess(bootstrap.index('install_pytorch_vmm_allocator('),bootstrap.index('runpy.run_path('))
        self.assertIn('DDO_VMM_TORCH_CACHE_MIB="512"',(HERE/'capture.py').read_text())


if __name__ == '__main__':
    unittest.main()
