# Compare stdout from this script under the pinned pre-844 and candidate environments.
# This intentionally names retired APIs only to exercise the immutable baseline.
using ProximalBase
const PB = ProximalBase
const modern = isdefined(PB, :prox_block!)
const scalar_prox = getfield(PB, modern ? :prox_block : :prox_entry)
const retired_step! = modern ? nothing : getfield(PB, :prox_coordinate!)

function record(io, label, x)
    println(io, label, '\t', join((isnan(v) ? "NaN" : repr(v) for v in x), ','))
end

function differential(io)
    for T in (Float32, Float64)
        for f in (NormL1(T(.3)), ElasticNet(T(.3),T(.2)), MCP(T(.3)), SCAD(T(.3)),
                  IndBox(T(-1),T(2)), NegEntropy(T(.2)))
            for u in (T(0), T(2), T(-3), T(NaN), T(Inf), T(floatmax(T)/2)), γ in (T(0),T(.7),T(Inf))
                record(io, "$T/$(nameof(typeof(f)))/$u/$γ", (scalar_prox(f,u,1,γ),))
            end
        end
        gs = [[1,3], [4,2]]
        for λ in (T(0),T(.3),T(Inf)), w in (T(0),T(1),T(Inf))
            for f in (GroupNormL2(λ,gs,T[w,1]), GroupNormLinf(λ,gs,T[w,1]),
                      SparseGroupNormL2(T(.2),λ,gs,T[w,1]),
                      GroupMCP(λ,gs,T[w,1]), GroupSCAD(λ,gs,T[w,1]))
                for h in (f,Scaled(f,T(2)),Regularize(f,T(.2)),Regularize(f,T(.2),T[1,2,3,4]))
                    for u in (T[0,0],T[3,-4],T[NaN,2],T[Inf,0],T[floatmax(T)/2,0]), γ in (T(0),T(.7),T(Inf))
                        x = AtomIterate(T,4,gs)
                        out = if modern
                            PB.prox_block(h,u,1,γ)
                        else
                            collect(retired_step!(x,h,1,u,γ))
                        end
                        record(io, "$T/$(typeof(h))/$λ/$w/$u/$γ", out)
                    end
                end
            end
        end
    end
end

differential(stdout)
