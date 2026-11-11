"""Record generated-harness code hashes and preserve inherited control metadata."""
import hashlib,json
from pathlib import Path

def record_control(harness,variant):
    harness=Path(harness);file=harness/'control.json';parent=None
    if file.exists():parent=json.loads(file.read_text())
    hashes={str(file.relative_to(harness)):hashlib.sha256(file.read_bytes()).hexdigest()
            for file in sorted(harness.rglob('*'))
            if file.is_file() and file.suffix in ('.jl','.py','.toml')}
    file.write_text(json.dumps({'variant':variant,'code_sha256':hashes,'inherited_control':parent},indent=2)+'\n')
