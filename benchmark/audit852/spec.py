"""The complete, version-independent case matrix. No timing or package imports."""
SPARSE_OPS = ('construct_empty','construct_populated','activate_batch','write_active',
 'repeat_update','active_sweep','read_random','read_sequential','read_scalar','write_scalar',
 'dense_import','copy','copyto','clear','zero_active','dropzeros','dot','matvec',
 'cd_fixed','cd_expand','prune_rebuild')
SYM_OPS = ('construct_empty','construct_populated','activate_batch','write_active',
 'repeat_update','active_sweep','read_diagonal','read_offdiagonal','copy','copyto',
 'clear','zero_active','dropzeros','bilinear','matmat','symmetric_updates')
BLOCK_OPS = ('construct_empty','construct_populated','activate_batch','write_active',
 'block_read','block_sweep','copy','copyto','clear','dropzeros','dot','matvec','block_cd')
def scenarios():
    out=[]
    for dtype in ('Float32','Float64'):
        for n in (10000,1000000):
            for s in (100,1000,10000):
                out.append(dict(family='sparse',dtype=dtype,n=n,s=s,b=1,layout='scalar',ops=SPARSE_OPS))
        for n in (144,1414):
            for s in (100,1000,10000):
                out.append(dict(family='symmetric',dtype=dtype,n=n,s=s,b=1,layout='triangle',ops=SYM_OPS))
        for b in (1,8,64):
            for layout in ('contiguous','strided','arbitrary'):
                for pct in (1,10):
                    n=8192;s=max(1,(n//b)*pct//100)
                    out.append(dict(family='block',dtype=dtype,n=n,s=s,b=b,layout=layout,ops=BLOCK_OPS))
    # Ambient/group-count scaling at fixed block size; all layouts and activity levels.
    for layout in ('contiguous','strided','arbitrary'):
        for pct in (1,10):
            out.append(dict(family='block',dtype='Float64',n=65536,s=(65536//8)*pct//100,b=8,layout=layout,
                            ops=('construct_empty','copyto','clear','matvec','block_cd')))
    for x in out:
        x['key']=f"{x['family']}/{x['dtype']}/n{x['n']}/s{x['s']}/b{x['b']}/{x['layout']}"
    return out
VERSIONS={'original':'bd0740d0ebbebcc76fe208e4318fe6fc5451d34b',
 'historical05':'def81ec9d021c39ed1727cfeb9598357291dd60d',
 'post848':'eff21b6d7d33c622313122f46c6ee71693c101ee',
 'post850':'2b373c546e247781d9514a7000ef3e010343ca4c'}
if __name__=='__main__':
    import json
    print(json.dumps(scenarios(),indent=2))
