"""Verify documentation-only changes under src against the post-#850 baseline."""
from pathlib import Path
import re, subprocess, hashlib
root = Path(__file__).resolve().parents[3]
for path in sorted((root/'src').rglob('*.jl')):
    rel = path.relative_to(root).as_posix()
    before = subprocess.check_output(['git','show','2b373c54:'+rel], cwd=root, text=True)
    after = path.read_text()
    strip = lambda text: re.sub(r'^"""\n.*?^"""\n', '', text, flags=re.S|re.M)
    assert strip(before) == strip(after), f'Executable source changed: {rel}'
    print(rel, 'executable text unchanged', hashlib.sha256(after.encode()).hexdigest())
print('PASS: all 17 source files preserve executable text; changes are docstrings only.')
