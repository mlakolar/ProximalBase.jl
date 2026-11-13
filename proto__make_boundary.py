# Build the disposable real-package pairs for the compile/precompile measurements:
# pkgs/bnd/<V>/ProximalBase (PB master + the variant's PB-side code as a PB source file) and
# pkgs/bnd/<V>/ProximalSolvers (the variant's PS sources calling it through the package boundary).
import pathlib, re, shutil, sys
run = pathlib.Path(sys.argv[1]); proto = run / 'proto'; out = run / 'pkgs' / 'bnd'
if out.exists(): shutil.rmtree(out)
pbsrc = run / 'pkgs' / 'pb-master'
ps_root = run.parents[2]
WORKLOAD = '''      # PB #911 prototype: the cache-aware oracle in the workload
      zz = X * x; dd = similar(b); gg = similar(x); rr = b .- zz
      for ff in (LeastSquaresLoss(b, X), Scaled(LeastSquaresLoss(b, X), T(0.2)))
        __CALLS__
      end
'''
CALLS = {
 'B1': 'value_at_image(ff, zz); value_and_gradient_at_image!(gg, dd, ff, zz); value_at_residual(ff, rr)',
 'B2': 'value_at_image(ff, zz); value_and_gradient_at_image!(gg, dd, ff, zz); value_at_residual(ff, rr)',
 'B3': 'value_at_image(ff, zz); value_and_gradient_at_image!(gg, dd, ff, zz); value_at_residual(ff, rr)',
 'H': 'value_at_image(ff, zz); value_and_gradient_at_image!(gg, dd, ff, zz); value_at_residual(ff, rr)',
 'C':  'half_sumsq_diff(zz, b); half_sumsq(rr)',
}
def pb_side(v):
    s = (proto / 'variants' / v / 'src' / 'pb_oracle.jl').read_text()
    s = re.sub(r'^module PBOracle\n', '', s, flags=re.M)
    s = re.sub(r'\nend\s*$', '\n', s)
    s = re.sub(r'^(using LinearAlgebra|import ProximalBase|const PB = ProximalBase|using ProximalBase:.*)\n', '', s, flags=re.M)
    s = s.replace('PB.', '')
    s = s.replace('_apply_scale(', '_image_scale(').replace('_len_error(', '_image_len_error(').replace('_alias_error(', '_image_alias_error(')
    if v == 'B3':
        s = re.sub(r'^const UNCHECKED = .*\n', '', s, flags=re.M)
        s = s.replace('@inline _val(h, z) = UNCHECKED ? _value_unchecked(h, z) : h(z)', '@inline _image_val(h, z) = h(z)')
        s = s.replace('@inline _vag!(d, h, z) = UNCHECKED ? _value_and_gradient_unchecked!(d, h, z) : value_and_gradient!(d, h, z)',
                      '@inline _image_vag!(d, h, z) = value_and_gradient!(d, h, z)')
        s = s.replace('_val(', '_image_val(').replace('_vag!(', '_image_vag!(')
        s = s.replace('_image_image_val(', '_image_val(').replace('_image_image_vag!(', '_image_vag!(')
    return s
for v in ['K', 'B1', 'B2', 'B3', 'H', 'C']:
    d = out / v
    pb = d / 'ProximalBase'
    if v == 'H':   # revision 2: the disposable PB refactor (shared kernel, LS methods rerouted, oracle)
        shutil.copytree(run / 'pkgs' / 'pb-H', pb)
    else:
        shutil.copytree(pbsrc, pb)
    if v not in ('K', 'H'):
        (pb / 'src' / 'functions' / 'leastsquares_image.jl').write_text('# PB #911 prototype ' + v + ' (disposable)\n' + pb_side(v))
        m = (pb / 'src' / 'ProximalBase.jl').read_text()
        a = 'include("functions/scaled.jl")\n'
        assert m.count(a) == 1
        (pb / 'src' / 'ProximalBase.jl').write_text(m.replace(a, a + 'include("functions/leastsquares_image.jl")\n'))
        w = (pb / 'src' / 'precompile.jl').read_text()
        a = '      prox_block(NormL1(T(0.1)), x[1], 1, one(T))\n'
        assert w.count(a) == 1
        (pb / 'src' / 'precompile.jl').write_text(w.replace(a, WORKLOAD.replace('__CALLS__', CALLS[v]) + a))
    ps = d / 'ProximalSolvers'; ps.mkdir()
    shutil.copytree(proto / 'variants' / v / 'src', ps / 'src')
    if v != 'K':
        (ps / 'src' / 'pb_oracle.jl').unlink()
        m = (ps / 'src' / 'ProximalSolvers.jl').read_text()
        m = m.replace('include("pb_oracle.jl")\n', ''); (ps / 'src' / 'ProximalSolvers.jl').write_text(m)
        for f in (ps / 'src').rglob('*.jl'):
            t = f.read_text(); f.write_text(t.replace('PBOracle.', 'ProximalBase.'))
        m = (ps / 'src' / 'ProximalSolvers.jl').read_text()
        assert 'import ProximalBase' not in m
        m = m.replace('using LinearAlgebra\n', 'using LinearAlgebra\nimport ProximalBase\n', 1)
        (ps / 'src' / 'ProximalSolvers.jl').write_text(m)
    proj = (ps_root / 'Project.toml').read_text()
    proj = re.sub(r'\[sources\]\n[^\[]*', '', proj)
    (ps / 'Project.toml').write_text(proj)
print('built', sorted(p.name for p in out.iterdir()))
