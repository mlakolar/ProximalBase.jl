"""Record generated-harness code hashes and preserve inherited control metadata."""
import hashlib,json
from pathlib import Path

def record_control(harness,variant):
    harness=Path(harness);file=harness/'control.json';parent=None
    if file.exists():parent=json.loads(file.read_text())
    hashes={name:hashlib.sha256((harness/name).read_bytes()).hexdigest()
            for name in ('adapters.jl','cases.jl','worker.jl','spec.py','Project.toml','Manifest.toml')}
    file.write_text(json.dumps({'variant':variant,'code_sha256':hashes,'inherited_control':parent},indent=2)+'\n')
