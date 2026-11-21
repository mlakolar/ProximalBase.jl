#!/usr/bin/env python3
"""Prepare an instrumented temporary copy for traversal counts, never for timing."""
import shutil
import sys
from pathlib import Path

FIELDS = ('sum', 'classify', 'condat', 'selector_excess', 'selector_max', 'shrink', 'copy', 'scalar_store')


def hook(name, n='length(x)'):
    return f'_record_scan!({FIELDS.index(name) + 1}, {n})'


def replace_once(body, old, new):
    if body.count(old) != 1:
        raise RuntimeError(f'Expected one profiling anchor for {old!r}')
    return body.replace(old, new)


def prefix(body, line, method):
    return replace_once(body, line, ' ' * (len(line)-len(line.lstrip())) + method + '\n' + line)


def profile_linf(body):
    line = 'function _linf_sum(x, rd, ::Type{T}) where {T}\n'
    structure = f'''const _trace_n = Ref(0)
const _trace_scans = zeros(Int, 3, {len(FIELDS)})
@noinline function _record_scan!(i, n)
  _trace_scans[1,i] += 1
  _trace_scans[2,i] += n == _trace_n[]
  _trace_scans[3,i] += n
  nothing
end

'''
    body = replace_once(body, line, structure + line + f'  {hook("sum")}\n')
    line = 'function _linf_classify(x, rd, ::Type{T}) where {T<:Union{Float32, Float64}}\n'
    body = replace_once(body, line, line + f'  {hook("classify")}\n')
    body = prefix(body, '  many && return _l1ball_handover', hook('condat', 'j - 1'))
    body = prefix(body, '    r, k, hi = _l1ball_excess(x, n, τ, rd)', hook('selector_excess', 'n'))
    line = '@noinline function _l1ball_top(x, n::Int, rd::F, ::Type{T}) where {T, F}\n'
    return replace_once(body, line, line + f'  {hook("selector_max", "n")}\n')


def profile_ball(body):
    line = '  _ball_l1_inside(a, s, k) && return _ball_l2_copy!(y, x)'
    replacement = f'''  if _ball_l1_inside(a, s, k)
    y !== x && {hook("copy")}
    return _ball_l2_copy!(y, x)
  end'''
    body = replace_once(body, line, replacement)
    line = '    @inbounds y[1] = copysign(a, convert(T, x[1]))'
    body = prefix(body, line, hook('scalar_store', '1'))
    line = '@inline function _ball_l1_shrink_loop!(y::_DenseOutput, x, τ::T) where {T}\n'
    return replace_once(body, line, line + f'  {hook("shrink")}\n')


def main():
    source, staged = map(Path,sys.argv[1:])
    shutil.copytree(source, staged)
    for name, transform in (('normlinf.jl', profile_linf), ('indballl1.jl', profile_ball)):
        target = staged / 'functions' / name
        target.write_text(transform(target.read_text()))
    (staged/'scan_fields.txt').write_text('\n'.join(FIELDS)+'\n')


if __name__ == '__main__':
    main()
