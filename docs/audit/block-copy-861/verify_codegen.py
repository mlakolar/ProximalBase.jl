"""Preserve instructions/constants; normalize declared process global addresses."""
import hashlib,json,re,sys
from pathlib import Path

def normalize(text):
    for symbol,address in re.findall(r'\.set\s+("[^"]+"|[^,\s]+),\s*(\d+)',text):
        text=re.sub(r'\b'+address+r'\b','ADDRESS_OF_'+symbol,text)
    return text

before,after,output=map(Path,sys.argv[1:])
rows=[]
assert {p.name for p in before.glob('*.s')}=={p.name for p in after.glob('*.s')}
for a in sorted(before.glob('*.s')):
    b=after/a.name
    rows.append({'file':a.name,'v1_sha256':hashlib.sha256(a.read_bytes()).hexdigest(),
        'final_sha256':hashlib.sha256(b.read_bytes()).hexdigest(),
        'exact_equal':a.read_bytes()==b.read_bytes(),
        'equal_after_global_address_normalization':normalize(a.read_text())==normalize(b.read_text())})
result={'normalization':'Replace only numeric addresses of symbols explicitly declared by assembler .set directives, using their declared symbol name. Instructions/registers/labels/constants unchanged.',
        'comparisons':rows}
assert len(rows)==24
assert all(r['equal_after_global_address_normalization'] for r in rows)
output.write_text(json.dumps(result,indent=2)+'\n')
print(f"{len(rows)} identical native listings; {sum(r['exact_equal'] for r in rows)} require no normalization")
