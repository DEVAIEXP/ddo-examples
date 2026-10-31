"""Cold fixed limits and eight bounded native cycle snapshots."""
import ctypes
import json
import os
from pathlib import Path

FIELDS = ['schema','enabled','frozen','owners','exclusive_backing','padding',
    'physical','va','physical_peak','va_peak','pending_owners','pending_bytes',
    'compact','workspace','failures','quarantined','held','parked','poisoned',
    'encodes','decodes','analyses','receipts','reads','compact_backing',
    'workspace_size','saved_bytes','stage','encode_host_ns','park_host_ns',
    'restore_host_ns','verified']
LABELS = ['cold_configured','after_text_encoder_cleanup','before_observed_blocks',
          'after_block_0','parked','restored','after_block_1','after_transformer_cleanup']


def validate_snapshots(data, complete=False):
    if data.get('schema') != 2 or data.get('fields') != FIELDS:
        raise ValueError('Cycle snapshot schema mismatch')
    rows = data['snapshots']
    if not 1 <= len(rows) <= 8 or [r['label'] for r in rows] != LABELS[:len(rows)]:
        raise ValueError('Cycle snapshot coverage/ceiling')
    if complete and len(rows) != 8:
        raise ValueError('Incomplete cycle snapshots')
    for index, snapshot in enumerate(rows):
        row = snapshot['values']
        if (len(row) != 32 or any(type(n) is not int or not 0 <= n < 2**64 for n in row)
                or row[:2] != [2,1] or row[3] > 64 or row[4] > 2147483648
                or row[5] > min(row[4],134217728)
                or row[6] > 4294967296 or row[7] > 8589934592
                or row[8] > 4294967296 or row[9] > 8589934592
                or row[6] > row[8] or row[7] > row[9]
                or any(row[i] for i in (14,15,18))
                or row[19] > 1 or row[20] > 1 or row[21] > 2
                or row[22] > 4 or row[23] > 1030
                or row[24] > 35651584 or row[25] > 262144
                or row[12] > row[24] or row[13] > row[25]):
            raise ValueError('Cycle charge/counter/failure mismatch')
        if index == 0 and (row[2] or any(row[3:])):
            raise ValueError('Configuration was not cold')
        if index and row[2] != 1:
            raise ValueError('Configuration did not freeze')
        if index < 4 and any(row[12:14]+row[16:]):
            raise ValueError('Cycle ran before the selected read')
        if index >= 4:
            if (not 0 < row[24] < 35651584 or row[24] % 2097152
                    or not 0 < row[25] <= 262144
                    or row[26] != 35651584-row[24]-row[25] or row[26] <= 0
                    or 35651584+row[24]+row[25] > 71565312
                    or 35651584+row[24] > 71303168
                    or row[19] != 1 or row[21] != 1 or row[28] <= 0 or row[29] <= 0):
                raise ValueError('Measured codec geometry/saving mismatch')
        if index == 4 and (row[12:14] != row[24:26] or row[16:18] != [1,1]
                or row[20:24] != [0,1,1,515] or row[27] != 3 or any(row[30:])):
            raise ValueError('Park receipt mismatch')
        if index >= 5 and (any(row[12:14]+row[16:18]) or row[20:24] != [1,1,2,1030]
                or row[27] != 4 or row[30] <= 0 or row[31] != 1):
            raise ValueError('Restore/cleanup receipt mismatch')


class Isolation:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        self.data = dict(schema=2,fields=FIELDS,snapshots=[],
            scope='Pool plus codec charges; not physical local VRAM placement')
        self.library = ctypes.CDLL(os.environ['DDO_VMM_LIBRARY_PATH'])
        configure = self.library.sz_isolation_configure
        configure.argtypes = [ctypes.c_uint64]*4
        configure.restype = ctypes.c_int
        combined = self.library.sz_combined_configure
        combined.argtypes = [ctypes.c_uint64]*2
        combined.restype = ctypes.c_int
        if configure(33554432,64,2147483648,134217728) or combined(4294967296,8589934592):
            raise RuntimeError('Native cold configuration refused; no allocation authorized')
        self.call = self.library.sz_isolation_observation_snapshot
        self.call.argtypes = [ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]
        self.call.restype = ctypes.c_int
        self.snapshot('cold_configured')

    def snapshot(self,label):
        if len(self.data['snapshots']) >= 8:
            raise RuntimeError('Native snapshot ceiling')
        row = (ctypes.c_uint64*32)()
        if self.call(row,32):
            raise RuntimeError('Native cycle metadata unavailable')
        self.data['snapshots'].append(dict(label=label,values=list(row)))
        raw = json.dumps(self.data,indent=2).encode()
        if len(raw) > 65536:
            raise RuntimeError('Cycle metadata ceiling')
        (self.directory/'isolation.json').write_bytes(raw)
        validate_snapshots(self.data)
