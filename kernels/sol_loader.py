"""Load the reviewed Sol source without environment-specific search paths."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from threading import RLock

ROOT = Path(__file__).resolve().parent
REVISION = '781e8341fb3b29d3aea5d50265772e9619340b66'
_lock = RLock()


def source_identity():
    """Validate bundled text content without importing Torch or initializing CUDA."""
    manifest = json.loads((ROOT / 'sol_attn_source.json').read_text(encoding='utf-8'))
    if manifest.get('schema') != 1 or manifest.get('revision') != REVISION:
        raise ValueError('The bundled Sol-Attn provenance differs from the reviewed revision.')
    package = ROOT / 'sol_attn'
    expected = manifest.get('files', {})
    if not expected or '__init__.py' not in expected or 'interface.py' not in expected:
        raise ValueError('The bundled Sol-Attn source manifest is incomplete.')
    files = {}
    for name, digest in expected.items():
        path = package / name
        if not path.resolve().is_relative_to(package.resolve()) or not path.is_file():
            raise ValueError(f'Missing or invalid bundled Sol-Attn file: {name}')
        data = path.read_bytes()
        if hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest() != digest:
            raise ValueError(f'The bundled Sol-Attn source was modified: {name}')
        files[str(path.resolve())] = hashlib.sha256(data).hexdigest()
    actual = {str(path.relative_to(package)).replace('\\', '/') for path in package.rglob('*')
              if path.is_file() and (path.suffix == '.py' or 'LICENSE' in path.name or 'NOTICE' in path.name)}
    if actual != set(expected):
        raise ValueError('The bundled Sol-Attn source contains unexpected files.')
    return {'revision': REVISION, 'interface': str((package / 'interface.py').resolve()),
            'source_kind': 'bundled', 'files': files}


def load_interface():
    """Load upstream's package name lazily and refuse a mixed external installation."""
    with _lock:
        source_identity()
        package = ROOT / 'sol_attn'
        existing = sys.modules.get('sol_attn')
        if existing is not None:
            origin = getattr(existing, '__file__', None)
            if origin is None or Path(origin).resolve() != (package / '__init__.py').resolve():
                raise ImportError('An external sol_attn package is already loaded; start a fresh runner process.')
        else:
            before = set(sys.modules)
            spec = importlib.util.spec_from_file_location(
                'sol_attn', package / '__init__.py', submodule_search_locations=[str(package)])
            module = importlib.util.module_from_spec(spec)
            sys.modules['sol_attn'] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                for name in set(sys.modules) - before:
                    if name == 'sol_attn' or name.startswith('sol_attn.'):
                        sys.modules.pop(name, None)
                raise
        interface = importlib.import_module('sol_attn.interface')
        if Path(interface.__file__).resolve() != (package / 'interface.py').resolve():
            raise ImportError('The loaded Sol-Attn interface differs from the bundled source.')
        return interface
