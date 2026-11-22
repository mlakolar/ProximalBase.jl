#!/usr/bin/env python3
"""Make the two variant trees of the GroupSCAD diagnosis from a candidate tree (a folder with src/).

    python3 benchmark/groupscad/make_variants.py CANDIDATE_TREE OUT_DIR

OUT_DIR/vstored/src: the ordinary value path (the value of f(x) and the two ordinary terms of the fused value) calls the existing
branch-based _scad_stored_value instead of _scad_value. OUT_DIR/vinplace/src: the ordinary scaling of a group uses a loop over the
one array when y === x (so that the compiler can vectorize it without an alias test) and _l2_scale! otherwise. Each edit must match
exactly once, otherwise the script stops. The variants are measurement trees; they are not part of the package.
"""
import pathlib
import shutil
import sys


def patch(path, edits):
    text = path.read_text()
    for old, new in edits:
        if text.count(old) != 1:
            sys.exit(f"{path}: the pattern occurs {text.count(old)} times, expected once: {old[:70]!r}")
        text = text.replace(old, new)
    path.write_text(text)


def main(candidate, out):
    candidate, out = pathlib.Path(candidate), pathlib.Path(out)
    for name in ("vstored", "vinplace"):
        if (out / name).exists():
            shutil.rmtree(out / name)
        shutil.copytree(candidate / "src", out / name / "src")
    patch(out / "vstored" / "src" / "functions" / "groupscad.jl", [
        ("return (_scad_value(λ, a, sqrt(s)), true)", "return (_scad_stored_value(λ, a, sqrt(s)), true)"),
        ("return V ? _scad_value(λ, a, r) : T(_L2_RECOMPUTE)", "return V ? _scad_stored_value(λ, a, r) : T(_L2_RECOMPUTE)"),
        ("return _scad_value(λ, a, sqrt(sy))", "return _scad_stored_value(λ, a, sqrt(sy))"),
    ])
    helper = """# y[G] = q x[G]; for y === x a loop over the one array, which the compiler can vectorize without an alias test
@inline function _gscad_scale!(y, x, G, q)
  if y === x
    @inbounds @simd for i in G
      x[i] = q * x[i]
    end
  else
    _l2_scale!(y, x, G, q)
  end
  nothing
end

"""
    anchor = "# y[G] = q x[G] with q in the recovery type R"
    patch(out / "vinplace" / "src" / "functions" / "groupscad.jl", [
        ("V || (_l2_scale!(y, x, G, q); return T(_L2_RECOMPUTE))", "V || (_gscad_scale!(y, x, G, q); return T(_L2_RECOMPUTE))"),
        (anchor, helper + anchor),
    ])
    print(f"variants in {out}: vstored, vinplace")


if len(sys.argv) != 3:
    sys.exit(__doc__)
main(*sys.argv[1:])
