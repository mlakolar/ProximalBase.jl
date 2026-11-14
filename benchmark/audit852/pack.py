#!/usr/bin/env python3
"""Archive explicit campaign/log paths deterministically, preserving every byte."""
import argparse,gzip,hashlib,io,tarfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('paths',nargs='+',type=Path);a=p.parse_args()
assert not a.output.exists(),a.output
files={}
for root in a.paths:
    assert root.exists(),root
    if root.is_file():files[root.name]=root
    else:
        for f in root.rglob('*'):
            if f.is_file():files[root.name+'/'+f.relative_to(root).as_posix()]=f
with a.output.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',filename='',mtime=0,compresslevel=9) as zipped,tarfile.open(fileobj=zipped,mode='w|',format=tarfile.GNU_FORMAT) as tf:
    for name,f in sorted(files.items()):
        data=f.read_bytes();info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644;info.mtime=0
        tf.addfile(info,io.BytesIO(data))
print(hashlib.sha256(a.output.read_bytes()).hexdigest(),a.output.name,len(files),'members',a.output.stat().st_size,'bytes')
