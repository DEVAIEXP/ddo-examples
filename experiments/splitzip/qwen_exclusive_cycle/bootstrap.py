"""Fresh child: resolve the isolated codec dependency and install before model imports."""
import os
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent
DLL_DIRECTORY = os.add_dll_directory(str(Path(os.environ['DDO_VMM_LIBRARY_PATH']).parent))
from ddo_vmm_allocator.torch_allocator import install_pytorch_vmm_allocator
install_pytorch_vmm_allocator(os.environ['DDO_VMM_LIBRARY_PATH'])
os.environ['DDO_VMM_FAST_CHILD'] = '1'
sys.path.insert(0,str(HERE))
sys.argv = [str(HERE/'runner.py'),*sys.argv[1:]]
runpy.run_path(str(HERE/'runner.py'),run_name='__main__')
