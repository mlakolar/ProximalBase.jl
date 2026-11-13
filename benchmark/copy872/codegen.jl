# Native code of the replacement path for each source tree, one tree per process.
# usage: julia -O2 --check-bounds=no codegen.jl SOURCE_ROOT OUTDIR
# Writes the full listings to OUTDIR and prints, per function, element type and layout,
# the instruction, push and call counts of the whole function, the number of
# instructions up to and including its first jump, and whether it reads an fs- or gs-based
# address.
using InteractiveUtils
const ROOT, OUT = abspath(ARGS[1]), abspath(ARGS[2])
mkpath(OUT)
Base.include(Main, joinpath(ROOT, "src", "ProximalBase.jl"))
const PB = Main.ProximalBase
const TREE = basename(ROOT)

layouts(n, b) = (contiguous = [(k-1)*b+1:k*b for k in 1:n÷b],
                 strided = [k:n÷b:n for k in 1:n÷b],
                 arbitrary = [collect((k-1)*b+1:k*b) for k in 1:n÷b])

for T in (Float64, Float32), (lname, groups) in pairs(layouts(64, 8))
  x = PB.BlockIterate(zeros(T, 64), groups)
  tt = Tuple{typeof(x), typeof(x)}
  for name in (:copyto!, :_copy_active!, :_copy_all_active!, :_try_copy_all_active!, :_copy_some_active!)
    f = name === :copyto! ? copyto! : isdefined(PB, name) ? getfield(PB, name) : continue
    hasmethod(f, tt) || continue
    nat = sprint(io -> code_native(io, f, tt; debuginfo = :none, syntax = :intel))
    write(joinpath(OUT, "$TREE-$T-$lname-$name.s"), nat)
    ins = [strip(l) for l in split(nat, '\n') if startswith(l, r"\s+[a-z]") && !startswith(strip(l), ".")]
    head = ins[1:something(findfirst(l -> startswith(l, "j"), ins), length(ins))]
    println(TREE, " ", T, " ", lname, " ", name, ": instructions=", length(ins),
            " pushes=", count(l -> startswith(l, "push"), ins),
            " calls=", count(l -> startswith(l, "call"), ins),
            " prologue_until_first_jump=", length(head),
            " fs_or_gs_access=", any(l -> occursin(r"[fg]s:", l), ins))
  end
end
