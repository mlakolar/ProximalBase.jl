using InteractiveUtils
function loadtree(name, root)
  m = Core.eval(Main, :(module $(Symbol("Tree_", name)) end))
  Base.include(m, joinpath(abspath(root), "src", "ProximalBase.jl"))
  getfield(m, :ProximalBase)
end
M = loadtree("t", ARGS[1])
f(M, T) = M.NormL1(T(0.3))
g = Base.invokelatest(f, M, Float64)
io = IOBuffer()
Base.invokelatest(code_native, io, M.prox_block, Tuple{typeof(g), Float64, Int, Float64}; debuginfo = :none, syntax = :intel)
s = String(take!(io))
ls = [strip(l) for l in split(s, '\n') if !isempty(strip(l)) && !startswith(strip(l), ".") && !startswith(strip(l), "#")]
println(length(ls), " instructions; conditional jumps: ", count(l -> occursin(r"^j[a-z]+\s", l) && !startswith(l, "jmp"), ls))
println(join(ls[1:min(end, 45)], "\n"))
