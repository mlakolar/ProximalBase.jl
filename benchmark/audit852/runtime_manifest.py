#!/usr/bin/env python3
"""Hash the complete installed Julia distribution, including its system image."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('julia',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
binary=a.julia.resolve();root=binary.parent.parent;files={}
for file in sorted(root.rglob('*')):
    if file.is_file():
        with file.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        files[str(file.relative_to(root))]={'sha256':digest,'bytes':file.stat().st_size}
a.output.write_text(json.dumps({'julia_root':str(root),'executable':str(binary),'files':files},indent=2)+'\n')
print(len(files),'Julia distribution files hashed, including shared libraries/system image')
