#!/usr/bin/env python3
"""Make measurement variants of the GroupSCAD source for the diagnosis, from a tree with a src/ folder.

    python3 benchmark/groupscad/make_variants.py TREE OUT_DIR [vstored] [vinplace]

With no name both are made.

OUT_DIR/vstored/src: the ordinary value path (the value of f(x) and the two ordinary terms of the fused value) calls the existing
branch-based _scad_stored_value instead of _scad_value. The three edited lines are in the tree of 9d2fefc71 and in the head
of the pull request.

OUT_DIR/vinplace/src: the ordinary scaling of a group uses a loop over the one array when y === x (so that the compiler can
vectorize it without an alias test) and _l2_scale! otherwise. The edit applies to the source of 9d2fefc71, where the scaling is
`V || (_l2_scale!(y, x, G, q); return T(_L2_RECOMPUTE))`. A tree whose scaling already has the in-place flag (the head of the
pull request: `::Val{I}`) is left alone: the variant is not made, and the script says so.

Each edit must match exactly once, otherwise the script stops. The variants are measurement trees; they are not part of the
package.
"""
import pathlib
import shutil
import sys

GSCAD = pathlib.Path("src") / "functions" / "groupscad.jl"


def patch(path, edits):
    text = path.read_text()
    for old, new in edits:
        if text.count(old) != 1:
            sys.exit(f"{path}: the pattern occurs {text.count(old)} times, expected once: {old[:70]!r}")
        text = text.replace(old, new)
    path.write_text(text)


def copy_tree(tree, out, name):
    if (out / name).exists():
        shutil.rmtree(out / name)
    shutil.copytree(tree / "src", out / name / "src")
    return out / name / GSCAD


def vstored(tree, out):
    patch(copy_tree(tree, out, "vstored"), [
        ("return (_scad_value(λ, a, sqrt(s)), true)", "return (_scad_stored_value(λ, a, sqrt(s)), true)"),
        ("return V ? _scad_value(λ, a, r) : T(_L2_RECOMPUTE)", "return V ? _scad_stored_value(λ, a, r) : T(_L2_RECOMPUTE)"),
        ("return _scad_value(λ, a, sqrt(sy))", "return _scad_stored_value(λ, a, sqrt(sy))"),
    ])
    print(f"made {out / 'vstored'}")


def vinplace(tree, out):
    source = (tree / GSCAD).read_text()
    if "::Val{I}" in source:
        print("vinplace not made: the tree already has the in-place flag of the head")
        return
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
    patch(copy_tree(tree, out, "vinplace"), [
        ("V || (_l2_scale!(y, x, G, q); return T(_L2_RECOMPUTE))", "V || (_gscad_scale!(y, x, G, q); return T(_L2_RECOMPUTE))"),
        (anchor, helper + anchor),
    ])
    print(f"made {out / 'vinplace'}")


def main(args):
    if len(args) < 2:
        sys.exit(__doc__)
    tree, out = pathlib.Path(args[0]), pathlib.Path(args[1])
    names = args[2:] or ["vstored", "vinplace"]
    for name in names:
        if name not in ("vstored", "vinplace"):
            sys.exit(f"unknown variant {name!r}")
    out.mkdir(parents=True, exist_ok=True)
    for name in names:
        {"vstored": vstored, "vinplace": vinplace}[name](tree, out)


main(sys.argv[1:])
