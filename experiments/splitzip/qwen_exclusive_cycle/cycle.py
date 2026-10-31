"""One bounded native cycle, with no retained tensor or Python residency policy."""
import ctypes
import time


def validate_cycle(data):
    if not isinstance(data, dict):
        raise ValueError('Cycle receipt unavailable')
    for key in ('started','closed','restore_before_access','native_payload_verified'):
        if data.get(key) is not True:
            raise ValueError('Cycle receipt mismatch: '+key)
    if (data.get('first_operator') != 'aten.layer_norm.default'
            or not isinstance(data.get('next_operator'),str) or not data['next_operator']
            or type(data.get('park_operation')) is not int
            or type(data.get('restore_operation')) is not int
            or data['restore_operation'] <= data['park_operation']
            or type(data.get('parked_interval_host_ns')) is not int
            or data['parked_interval_host_ns'] <= 0):
        raise ValueError('Cycle declared access ordering mismatch')


class Cycle:
    def __init__(self, isolation, deadline, *, functions=None):
        self.isolation, self.deadline = isolation, deadline
        if functions is None:
            library = isolation.library
            park, restore, error = library.sz_cycle_park, library.sz_cycle_restore, library.sz_cycle_error
            park.argtypes = [ctypes.c_void_p,ctypes.c_void_p,ctypes.c_uint64]
            restore.argtypes = [ctypes.c_void_p]
            error.argtypes = []
            park.restype = restore.restype = ctypes.c_int
            error.restype = ctypes.c_char_p
            functions = park,restore,error
        self.park, self.restore, self.error = functions
        self.started = self.parked = self.closed = False
        self.first_operator = self.next_operator = None
        self.park_operation = self.restore_operation = None
        self.parked_at = None
        self.parked_interval_host_ns = None

    def require(self, status):
        if status:
            message = self.error()
            raise RuntimeError('Native cycle refused; no retry: '+
                               (message.decode('utf-8',errors='replace') if message else 'diagnostic unavailable'))

    def before(self, observer, operator, roles):
        # Reject mutation and unknown semantics before allowing any dispatch.
        if roles != 1:
            raise RuntimeError('Selected activation requires declared read-only semantics')
        if not self.started and operator != 'aten.layer_norm.default':
            raise RuntimeError('First declared activation read changed')
        if self.parked:
            interval = time.perf_counter_ns()-self.parked_at
            self.require(self.restore(observer.execution[2]))
            self.isolation.snapshot('restored')
            self.parked = False
            self.closed = True
            self.next_operator = operator
            self.restore_operation = observer.operations
            self.parked_interval_host_ns = interval

    def after(self, observer, operator, roles):
        if self.started:
            return
        if roles != 1 or operator != 'aten.layer_norm.default':
            raise RuntimeError('First completed activation read changed')
        remaining = int((self.deadline-time.monotonic())*1000)
        if not 0 < remaining <= 1800000:
            raise RuntimeError('Cycle deadline')
        self.require(self.park(observer.target[1],observer.execution[2],remaining))
        self.parked_at = time.perf_counter_ns()
        self.started = self.parked = True
        self.first_operator = operator
        self.park_operation = observer.operations
        self.isolation.snapshot('parked')

    def summary(self):
        return dict(started=self.started,closed=self.closed,
            restore_before_access=self.closed,native_payload_verified=self.closed,
            first_operator=self.first_operator,next_operator=self.next_operator,
            park_operation=self.park_operation,restore_operation=self.restore_operation,
            parked_interval_host_ns=self.parked_interval_host_ns,
            scope='One enrolled activation; declared serial consumers only; no performance acceptance')
