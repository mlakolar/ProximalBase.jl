# Summarize one harness output (time + diag in the same process): per solver, the distribution of
# per-configuration paired-median ratios to K, split into configurations whose trajectory equals K's
# (same iteration count and bitwise iterate) and the rest; for the rest also the ratio normalized by
# iterations (cost per iteration/pass). Usage: python3 analyze_time.py out.txt [--table]
import sys, collections
path = sys.argv[1]
rows = [l.rstrip('\n').split(',', 8) for l in open(path) if l.startswith(('time,', 'diag,'))]
T = collections.defaultdict(dict); D = collections.defaultdict(dict)
for r in rows:
    k = (r[2], r[3], r[4])
    if r[0] == 'time' and r[6] in ('ratio_median_paired', 'ratio_ci_lo', 'ratio_ci_hi', 'median_ns', 'min_ns'):
        T[k].setdefault(r[5], {})[r[6]] = float(r[8])
    elif r[0] == 'diag':
        D[k].setdefault(r[5], {})[r[6]] = r[8]
VARS = ['K2', 'K3', 'B1', 'B2', 'B3', 'B3u', 'H', 'C']
def med(xs):
    s = sorted(xs); n = len(s); return (s[(n - 1) // 2] + s[n // 2]) / 2
print('file', path, 'timed configs', len(T))
for solver in ['cd', 'cds', 'pg', 'apg']:
    keys = [k for k in T if k[1] == solver]
    if not keys: continue
    print('\n== %s (%d configurations)' % (solver, len(keys)))
    print('variant | same-trajectory: n, min/median/max ratio [CI envelope] | changed trajectory: n, raw min/max, per-iteration min/median/max')
    for v in VARS:
        same, diff = [], []
        for k in keys:
            if v not in T[k]: continue
            t = T[k][v]; dk, dv = D.get(k, {}).get('K', {}), D.get(k, {}).get(v, {})
            st = dv.get('bitwise_x_equal_K') == 'true' and dv.get('iterations') == dk.get('iterations')
            if st: same.append((t['ratio_median_paired'], t['ratio_ci_lo'], t['ratio_ci_hi'], k))
            else:
                itK, itV = float(dk.get('iterations', 'nan')), float(dv.get('iterations', 'nan'))
                diff.append((t['ratio_median_paired'], t['ratio_median_paired'] * itK / itV, k, itK, itV))
        s = ''
        if same:
            r = [x[0] for x in same]
            s += '%2d  %.3f/%.3f/%.3f [%.3f, %.3f]' % (len(same), min(r), med(r), max(r), min(x[1] for x in same), max(x[2] for x in same))
            w = max(same); s += '  (max: %s %s %s)' % w[3]
        if diff:
            r = [x[0] for x in diff]; q = [x[1] for x in diff]
            s += '  ||  %2d  raw %.3f..%.3f  per-iter %.3f/%.3f/%.3f' % (len(diff), min(r), max(r), min(q), med(q), max(q))
        print('  %-4s %s' % (v, s))
if '--worst' in sys.argv:
    print('\nconfigurations where a same-trajectory variant exceeds the A/A envelope of its solver:')
    for solver in ['cd', 'cds', 'pg', 'apg']:
        keys = [k for k in T if k[1] == solver]
        if not keys: continue
        aa = max(max(T[k][a]['ratio_median_paired'] for a in ('K2', 'K3')) for k in keys)
        for v in ['B1', 'B2', 'B3', 'H', 'C']:
            for k in keys:
                dk, dv = D.get(k, {}).get('K', {}), D.get(k, {}).get(v, {})
                if dv.get('bitwise_x_equal_K') == 'true' and dv.get('iterations') == dk.get('iterations'):
                    t = T[k][v]
                    if t['ratio_median_paired'] > aa:
                        print('  %s %s ratio %.4f [%.4f, %.4f] > A/A max %.4f; same config K2 %.4f K3 %.4f' % (v, k, t['ratio_median_paired'], t['ratio_ci_lo'], t['ratio_ci_hi'], aa, T[k]['K2']['ratio_median_paired'], T[k]['K3']['ratio_median_paired']))
