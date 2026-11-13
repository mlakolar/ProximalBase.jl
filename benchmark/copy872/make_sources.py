#!/usr/bin/env python3
"""Build benchmark-only source trees outside the repository: the base commit's src/ with
_copy_active! replaced by each candidate in candidates/ (A and AA are the base itself), plus
the original implementation at bd0740d0 used by #852 (role O).
usage: python3 make_sources.py BASE_SHA OUTDIR"""
import pathlib,subprocess,sys
here=pathlib.Path(__file__).resolve().parent
root=here.parent.parent
base,out=sys.argv[1],pathlib.Path(sys.argv[2]).resolve()
out.mkdir(parents=True,exist_ok=False)
start='# The layouts and nonthrowing conversion have already been checked.'
endmark='  dest.nactive = src.nactive\n  dest\nend\n'
def extract(sha,dst):
    dst.mkdir(parents=True)
    tar=subprocess.run(['git','-C',str(root),'archive',sha,'src'],check=True,capture_output=True).stdout
    subprocess.run(['tar','-x','-C',str(dst)],input=tar,check=True)
for name in ['A','AA']+sorted(p.stem for p in (here/'candidates').glob('*.jl')):
    extract(base,out/name)
    if name in ('A','AA'):continue
    f=out/name/'src'/'block_iterate.jl';s=f.read_text()
    i=s.index(start);j=s.index(endmark,i)+len(endmark)
    f.write_text(s[:i]+(here/'candidates'/f'{name}.jl').read_text()+s[j:])
extract('bd0740d0ebbebcc76fe208e4318fe6fc5451d34b',out/'O')
print('built',sorted(p.name for p in out.iterdir()),'from',base)
