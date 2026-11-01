# Case definitions are independent of the scheduler and of measurement settings.
# Random data use stable scenario keys, so adding a scenario does not change others.
struct PerfCase{F,S,V,M}
    meta::M
    operation::F
    prepare::S
    validate::V
    evals::Union{Int,Nothing}
    profile::Bool
end

function scenario_rng(rep, key)
    seed = UInt64(0xcbf29ce484222325) ⊻ UInt64(rep)
    for b in codeunits(key)
        seed = (seed ⊻ UInt64(b)) * UInt64(0x100000001b3)
    end
    Xoshiro(seed)
end

noop() = nothing
tolerance(::Type{Float32}) = 5e-4
tolerance(::Type{Float64}) = 2e-8
dense(x) = Array(x)

function close_enough(x, y)
    t = tolerance(float(eltype(x)))
    isapprox(dense(x), dense(y); atol=t, rtol=t) || error("output differs from dense reference")
end

function addcase!(cases, family, name, op, validate; T=Float64, shape="", layout="dense",
                  phase="warm", prepare=noop, evals=prepare===noop ? nothing : 1, profile=false, work=0,
                  zero_alloc=phase == "warm")
    id = join((family, name, string(T), shape, layout, phase), "/")
    meta = (;id, family, name, dtype=string(T), shape, layout, phase, work, zero_alloc)
    push!(cases, PerfCase(meta, op, prepare, validate, evals, profile))
end

function addprox!(cases, family, name, f, x; γ=eltype(x)(0.7), layout="dense",
                  prepare=noop, profile=false, alias=false, value=true, reference_function=f)
    T = eltype(x)
    y = similar(x)
    reference = Ref{Any}(nothing)
    # Reference construction and checks are deliberately outside all measurements.
    function check!()
        r = prox(reference_function, dense(x), γ)
        reference[] = r
        prepare(); prox!(y, f, x, γ)
        close_enough(y, r)
        all(isfinite, y) || error("nonfinite proximal output")
        nothing
    end
    addcase!(cases, family, name * "/prox!", () -> prox!(y, f, x, γ), check!;
             T, shape=join(size(x), "x"), layout, prepare, profile, work=length(x))
    if value
        function checkvalue!()
            check!()
            prepare(); v = value_and_prox!(y, f, x, γ)
            close_enough(y, reference[])
            expected = f(y)
            t = tolerance(T)
            (v == expected || isapprox(v, expected; atol=t, rtol=t)) ||
                error("returned value differs from f(prox): $v versus $expected")
        end
        addcase!(cases, family, name * "/value_and_prox!", () -> value_and_prox!(y, f, x, γ), checkvalue!;
                 T, shape=join(size(x), "x"), layout, prepare, work=length(x))
    end
    if alias
        z = copy(x)
        prep_alias!() = copyto!(z, x)
        check_alias!() = (prep_alias!(); prox!(z, f, z, γ); close_enough(z, prox(f, dense(x), γ)))
        addcase!(cases, family, name * "/prox!_alias", () -> prox!(z, f, z, γ), check_alias!;
                 T, shape=join(size(x), "x"), layout, prepare=prep_alias!, work=length(x))
    end
end

function addcold!(cases, family, name, makef, x; γ=eltype(x)(0.7))
    y = similar(x)
    op() = prox!(y, makef(), x, γ)
    check!() = (op(); close_enough(y, prox(makef(), dense(x), γ)))
    addcase!(cases, family, name, op, check!; T=eltype(x), shape=join(size(x), "x"),
             phase="construct_and_first_prox", profile=eltype(x)==Float64, work=length(x))
end

function addgrad!(cases, family, name, f, x; profile=false, second=true)
    T = eltype(x)
    g = zeros(T, size(x))
    v = fill(T(inv(sqrt(length(x)))), size(x))
    hv = similar(g)
    function checkgrad!()
        val = value_and_gradient!(g, f, x)
        t = tolerance(T)
        isapprox(val, f(x); atol=t, rtol=t) || error("value/gradient objective mismatch")
        all(isfinite, g) || error("nonfinite gradient")
        if T == Float64
            δ = 1e-5
            fd = (f(dense(x) .+ δ .* v) - f(dense(x) .- δ .* v)) / (2δ)
            isapprox(dot(g, v), fd; atol=1e-5, rtol=1e-4) || error("directional gradient check failed")
        end
    end
    shape = join(size(x), "x")
    addcase!(cases, family, name * "/value", () -> f(x), checkgrad!; T, shape, work=length(x))
    addcase!(cases, family, name * "/gradient!", () -> gradient!(g, f, x), checkgrad!; T, shape, profile, work=length(x))
    addcase!(cases, family, name * "/value_and_gradient!", () -> value_and_gradient!(g, f, x), checkgrad!; T, shape, work=length(x))
    if second && hashessian(f)
        function checkhess!()
            hessian_vec!(hv, f, x, v)
            δ = T == Float32 ? T(0.01) : T(1e-5)
            if f isa GLMLoss && f.loss isa ProximalBase.Huber
                η=f.X*dense(x)
                f.offset===nothing || (η .+= f.offset)
                weights=T.(abs.(η .- f.y).<=f.loss.δ)
                f.weights===nothing || (weights .*= f.weights)
                reference=f.X'*(weights.*(f.X*v))
                close_enough(hv,reference)
                return
            end
            gp = gradient(f, dense(x) .+ δ .* v)
            gm = gradient(f, dense(x) .- δ .* v)
            isapprox(hv, (gp .- gm) ./ (2δ); atol=T==Float32 ? 0.02 : 1e-5,
                     rtol=T==Float32 ? 0.02 : 1e-4) || error("Hessian direction check failed")
        end
        addcase!(cases, family, name * "/hessian_vec!", () -> hessian_vec!(hv, f, x, v), checkhess!;
                 T, shape, work=length(x))
        q = quadratic_model(f, x)
        checkmodel!() = (quadratic_model!(q, f, x); close_enough(gradient(q, x), gradient(f, x)))
        addcase!(cases, family, name * "/quadratic_model!", () -> quadratic_model!(q, f, x), checkmodel!;
                 T, shape, work=length(x))
    end
end

# Tuple references descend to array blocks instead of calling tuple oracles.
function close_enough(x::Tuple, y::Tuple)
    length(x)==length(y) || error("block counts differ")
    foreach(close_enough,x,y)
    nothing
end

block_similar(x::Tuple) = map(block_similar,x)
block_similar(x) = similar(x)
block_type(x::Tuple) = block_type(first(x))
block_type(x) = eltype(x)
block_count(x::Tuple) = sum(block_count,x)
block_count(x) = 1
block_work(x::Tuple) = sum(block_work,x)
block_work(x) = length(x)
block_finite(x::Tuple) = all(block_finite,x)
block_finite(x) = all(isfinite,x)

block_functions(f) = f
block_functions(f::SeparableSum) = map(block_functions,f.fs)
block_wrap(w, fs::Tuple) = map(f -> block_wrap(w,f),fs)
block_wrap(w, f) = w(f)
block_functions(f::Scaled) = block_wrap(g -> Scaled(g,f.α),block_functions(f.f))
block_functions(f::Conjugate) = block_wrap(Conjugate,block_functions(f.f))
block_prox(fs::Tuple, x::Tuple, γ) = map((f,z) -> block_prox(f,z,γ),fs,x)
block_prox(f, x, γ) = prox(f,x,γ)
block_prox_value(fs::Tuple, x::Tuple, γ) = sum(map((f,z) -> block_prox_value(f,z,γ),fs,x))
# The leaf oracle also supplies values for conjugates without an explicit value method.
block_prox_value(f, x, γ) = value_and_prox!(similar(x),f,x,γ)
block_value(fs::Tuple, x::Tuple) = sum(map(block_value,fs,x))
block_value(f, x) = f(x)
block_gradient(fs::Tuple, x::Tuple) = map(block_gradient,fs,x)
block_gradient(f, x) = gradient(f,x)
block_hessian(fs::Tuple, x::Tuple, v::Tuple) = map(block_hessian,fs,x,v)
block_hessian(f, x, v) = hessian_vec(f,x,v)

function block_close_value(v, expected, T)
    t=tolerance(T)
    (v==expected || isapprox(v,expected;atol=t,rtol=t)) || error("block objective mismatch: $v versus $expected")
end

function addblockprox!(cases, name, f, x; value=true, operation="prox!")
    T=block_type(x); γ=T(0.7); y=block_similar(x)
    shape="$(block_count(x))x$(length(first_leaf(x)))"; work=block_work(x)
    fs=block_functions(f)
    function check!()
        reference=block_prox(fs,x,γ)
        prox!(y,f,x,γ)
        close_enough(y,reference)
        block_finite(y) || error("nonfinite proximal output")
    end
    addcase!(cases,"blocks",name*"/"*operation,()->prox!(y,f,x,γ),check!;
             T,shape,work)
    if value
        function checkvalue!()
            reference=block_prox(fs,x,γ)
            expected=block_prox_value(fs,x,γ)
            val=value_and_prox!(y,f,x,γ)
            close_enough(y,reference)
            block_finite(y) || error("nonfinite proximal output")
            block_close_value(val,expected,T)
        end
        addcase!(cases,"blocks",name*"/value_and_prox!",()->value_and_prox!(y,f,x,γ),checkvalue!;
                 T,shape,work)
    end
end

first_leaf(x::Tuple) = first_leaf(first(x))
first_leaf(x) = x

function addblockgrad!(cases, name, f, x, v; full=true)
    T=block_type(x); g=block_similar(x); hv=block_similar(x)
    shape="$(block_count(x))x$(length(first_leaf(x)))"; work=block_work(x)
    fs=block_functions(f)
    checkvalue!() = block_close_value(f(x),block_value(fs,x),T)
    function checkgrad!()
        gradient!(g,f,x)
        close_enough(g,block_gradient(fs,x))
    end
    function checkvaluegrad!()
        val=value_and_gradient!(g,f,x)
        close_enough(g,block_gradient(fs,x))
        block_close_value(val,block_value(fs,x),T)
    end
    function checkhess!()
        hessian_vec!(hv,f,x,v)
        close_enough(hv,block_hessian(fs,x,v))
    end
    if full
        addcase!(cases,"blocks",name*"/value",()->f(x),checkvalue!;T,shape,work)
        addcase!(cases,"blocks",name*"/gradient!",()->gradient!(g,f,x),checkgrad!;T,shape,work)
    end
    addcase!(cases,"blocks",name*"/value_and_gradient!",()->value_and_gradient!(g,f,x),checkvaluegrad!;
             T,shape,work)
    addcase!(cases,"blocks",name*"/hessian_vec!",()->hessian_vec!(hv,f,x,v),checkhess!;
             T,shape,work)
    if full
        q=quadratic_model(f,x)
        function checkmodel!()
            quadratic_model!(q,f,x)
            block_close_value(q(x),block_value(fs,x),T)
            close_enough(gradient(q,x),block_gradient(fs,x))
            close_enough(hessian_vec(q,x,v),block_hessian(fs,x,v))
        end
        addcase!(cases,"blocks",name*"/quadratic_model!",()->quadratic_model!(q,f,x),checkmodel!;
                 T,shape,work)
    end
end

function block_cases!(cases, rep; quick=false)
    # Both suites retain the same small-block overhead and long-block work cases.
    for T in (Float64,Float32), n in (2,1000)
        l1=NormL1(T(0.1)); l2=SqrNormL2(); l22=SqrNormL2(T(2))
        flat=SeparableSum(l1,l2)
        fixtures=(("flat",flat,:flat,false),
                  ("flat_smooth",SeparableSum(l2,l22),:flat,true),
                  ("nested",SeparableSum(flat,SeparableSum(l22,NormL1(T(0.05)))),:nested,false),
                  ("mixed",SeparableSum(l1,SeparableSum(l2,l1)),:mixed,false),
                  ("nested_smooth",SeparableSum(SeparableSum(l2,l22),SeparableSum(SqrNormL2(T(3)),SqrNormL2(T(4)))),:nested,true),
                  ("mixed_smooth",SeparableSum(l2,SeparableSum(l22,SqrNormL2(T(3)))),:mixed,true),
                  ("scaled",Scaled(flat,T(2)),:flat,false),
                  ("conjugate_known",Conjugate(flat),:flat,false),
                  ("conjugate_fallback",Conjugate(SeparableSum(ElasticNet(T(0.1),T(0.2)),ElasticNet(T(0.3),T(0.1)))),:flat,false),
                  ("conjugate_indicator",Conjugate(SeparableSum(IndBox(T(-1),T(1)),IndBox(T(-0.5),T(2)))),:flat,false),
                  ("conjugate_scaled",Conjugate(Scaled(flat,T(2))),:flat,false))
        for (name,f,shape,smooth) in fixtures
            rng=scenario_rng(rep,"blocks/$name/$T/$n")
            makeblocks() = shape==:nested ? ((randn(rng,T,n),randn(rng,T,n)),(randn(rng,T,n),randn(rng,T,n))) :
                           shape==:mixed ? (randn(rng,T,n),(randn(rng,T,n),randn(rng,T,n))) :
                           (randn(rng,T,n),randn(rng,T,n))
            x=makeblocks()
            if smooth
                addblockgrad!(cases,name,f,x,makeblocks())
            else
                addblockprox!(cases,name,f,x)
            end
        end
    end
    for T in (Float64,Float32), k in (8,31,32,64,128), n in (k in (8,32) ? (2,1000) : (2,))
        name="n$k"
        rng=scenario_rng(rep,"blocks/$name/$T/$n")
        x=ntuple(_ -> randn(rng,T,n),k)
        f=SeparableSum(ntuple(i -> isodd(i) ? NormL1(T(0.1)) : SqrNormL2(),k))
        addblockprox!(cases,name,f,x)
        addblockprox!(cases,name,Conjugate(f),x;value=false,operation="conjugate_prox!")
        rngsmooth=scenario_rng(rep,"blocks/$(name)_smooth/$T/$n")
        xs=ntuple(_ -> randn(rngsmooth,T,n),k)
        v=ntuple(_ -> randn(rngsmooth,T,n),k)
        smooth=SeparableSum(ntuple(i -> SqrNormL2(T(isodd(i) ? 1 : 2)),k))
        addblockgrad!(cases,name,smooth,xs,v;full=false)
    end
end

function separable_cases!(cases, rep; quick=false)
    for T in (quick ? (Float64,) : (Float32, Float64)), n in (quick ? (256,) : (256, 100_000))
        rng = scenario_rng(rep, "separable/$T/$n")
        x, w = randn(rng, T, n), rand(rng, T, n) .+ T(0.2)
        fs = (NormL1(T(0.3)), NormL1(T(0.3), w), NormL1Nonneg(T(0.3)),
              ElasticNet(T(0.3), T(0.4)), IndBox(T(-1), T(1)), IndNonnegative(),
              NormL0(T(0.1)), MCP(T(0.3)), SCAD(T(0.3)), NegEntropy(T(0.2)),
              KLDivergence(w; λ=T(0.2)), NegLog(T(0.2)), Berhu(T(0.3), T(1)),
              CheckLoss(T(0.3)), HingeLoss(), EpsInsensitiveLoss(T(0.2)),
              CappedL1(T(0.3), T(1)), LogPenalty(T(0.3), T(0.5)), NormLq(T(0.3), T(0.5)),
              NormL1Nonneg(T(0.3), w))
        for (k, f) in enumerate(fs)
            quick && !(k in (1, 4)) && continue
            weighted = f isa Union{NormL1,NormL1Nonneg} && f.w !== nothing
            name = string(nameof(typeof(f)), weighted ? "_weighted" : "")
            addprox!(cases, "separable", name, f, x; profile=T==Float64 && n==100_000 && k in (1,9,10,19), alias=k in (1,4) || weighted)
            if weighted
                z = f isa NormL1Nonneg ? abs.(x) : x
                checkvalue() = isapprox(f(z), sum(value_entry(f,z[i],i) for i in eachindex(z));
                                       rtol=tolerance(T), atol=tolerance(T)) || error("weighted value differs from entrywise sum")
                addcase!(cases, "separable", name * "/value", () -> f(z), checkvalue;
                         T, shape=string(n), work=n)
            end
        end
        if !quick
            wz = copy(w)
            wz[1:10:end] .= zero(T)
            for F in (NormL1, NormL1Nonneg)
                z = F === NormL1Nonneg ? abs.(x) : x
                for (suffix, level, weights) in (("zero_weights", T(0.3), wz), ("zero_level", zero(T), w))
                    f = F(level, weights)
                    checkzero() = isapprox(f(z), sum(value_entry(f,z[i],i) for i in eachindex(z));
                                          rtol=tolerance(T), atol=tolerance(T)) || error("zero-factor value differs from entrywise sum")
                    addcase!(cases, "separable", string(nameof(F), "_", suffix, "/value"), () -> f(z), checkzero;
                             T, shape=string(n), work=n)
                end
            end
        end
        sx = SparseIterate(sprandn(rng, T, n, 0.01))
        for f in (NormL1(T(0.3)), ElasticNet(T(0.3), T(0.4)), NormL0(T(0.1)), MCP(T(0.3)))
            quick && !(f isa NormL1) && continue
            addprox!(cases, "sparse_penalties", string(nameof(typeof(f))), f, sx; layout="sparse_1pct", profile=T==Float64 && n==100_000)
        end
        steps = T(0.2) .+ rand(rng, T, n)
        for f in (NormL1(T(0.3)), ElasticNet(T(0.3), T(0.4)))
            addprox!(cases, "metric", string(nameof(typeof(f))), f, x; γ=steps, layout="diagonal_step")
        end
        if !quick
            wi = copy(w)
            wi[1], wi[2] = zero(T), T(Inf)
            failed_steps = copy(steps)
            failed_steps[1] = zero(T)
            for (suffix, weights, γ) in (("ordinary", w, steps), ("zero_infinite_weights", wi, steps),
                                         ("uncertified_step", w, failed_steps))
                for f in (ElasticNet(T(0.3), T(0.4), weights), Berhu(T(0.3), one(T), weights),
                          CheckLoss(T(0.3), weights), EpsInsensitiveLoss(T(0.2), weights),
                          HingeLoss(nothing, weights), CappedL1(T(0.3), one(T), weights))
                    y = similar(x)
                    function checksteps!()
                        reference = [prox_entry(f, x[i], i, γ[i]) for i in eachindex(x)]
                        prox!(y, f, x, γ)
                        isequal(y, reference) || error("weighted steps differ from entrywise reference")
                    end
                    addcase!(cases, "metric", string(nameof(typeof(f)), "_weighted_", suffix, "/prox!"),
                             () -> prox!(y, f, x, γ), checksteps!;
                             T, shape=string(n), layout="diagonal_step", work=n)
                end
            end
        end
        xv = view(randn(rng, T, 2n), 1:2:2n)
        addprox!(cases, "layouts", "NormL1", NormL1(T(0.3)), xv; layout="stride2_input", alias=true)
        for f in (NormL2(T(0.3)), NormLinf(T(0.3)), IndBallL0(max(1,n÷10)),
                  IndBallL1(T(n/10)), IndBallL2(T(1)), IndSphereL2(T(1)), IndSimplex(T(1)),
                  IndHyperplane(one(T), zero(T)), IndHalfspace(one(T), zero(T)))
            quick && !(f isa Union{NormL2,IndSimplex}) && continue
            addprox!(cases, "norms_sets", string(nameof(typeof(f))), f, x;
                     profile=T==Float64 && n==100_000 && f isa Union{NormLinf,IndSimplex})
        end
        for f in (TotalVariation1D(T(0.4)), FusedLasso(T(0.2),T(0.4)), IndIsotonic(),
                  SLOPE(collect(range(T(0.5),T(0.1);length=n))))
            if f isa Union{TotalVariation1D,FusedLasso}
                function checkvalue!()
                    tv = sum(abs, diff(x))
                    reference = f isa TotalVariation1D ? f.λ * tv : f.λ1 * sum(abs,x) + f.λ2 * tv
                    isapprox(f(x),reference;atol=tolerance(T),rtol=tolerance(T)) ||
                        error("ordered value differs from direct sum")
                end
                addcase!(cases,"ordered",string(nameof(typeof(f)),"/value"),()->f(x),checkvalue!;
                         T,shape=string(n),profile=T==Float64 && n==100_000,work=n)
            end
            quick && !(f isa Union{TotalVariation1D,SLOPE}) && continue
            addprox!(cases, "ordered", string(nameof(typeof(f))), f, x; profile=T==Float64 && n==100_000)
        end
        groups = [j:min(j+7,n) for j in 1:8:n]
        weights = ones(T, length(groups))
        for f in (GroupNormL2(T(0.3),groups,weights), GroupNormLinf(T(0.3),groups,weights),
                  SparseGroupNormL2(T(0.2),T(0.3),groups,weights),
                  GroupMCP(T(0.3),groups,weights), GroupSCAD(T(0.3),groups,weights))
            quick && !(f isa GroupNormL2) && continue
            addprox!(cases, "groups", string(nameof(typeof(f))), f, x; profile=T==Float64 && n==100_000 && f isa SparseGroupNormL2)
            if !quick
                steps = T(0.2) .+ rand(rng, T, length(groups))
                addprox!(cases, "groups", string(nameof(typeof(f))), f, x; γ=steps, layout="group_steps")
            end
            xa = copy(x); xa[1:n÷2] .= 0
            atomx = AtomIterate(xa,f)
            addprox!(cases, "groups", string(nameof(typeof(f))), f, atomx; layout="atom_half_active")
            z = copy(atomx)
            step = T(0.7)
            group_index = length(groups)
            prep!() = copyto!(z,atomx)
            check!() = (prep!(); prox_coordinate!(z,f,group_index,step); all(isfinite,z) || error("nonfinite group update"))
            addcase!(cases, "coordinates", string(nameof(typeof(f))), () -> prox_coordinate!(z,f,group_index,step), check!;
                     T, shape=string(n), layout="atom", prepare=prep!, work=length(last(groups)))
        end
    end
    if !quick
        # SLOPE value at sizes beyond the (256, 100_000) loop above; inputs come from their own
        # seeded streams so the draws of every other case stay unchanged.
        for T in (Float32, Float64), n in (256, 512, 1024, 4096, 8192, 16384)
            rng = scenario_rng(rep, "ordered/SLOPE/value/$T/$n")
            x = randn(rng, T, n)
            f = SLOPE(collect(range(T(0.5), T(0.1); length = n)))
            function checkslope!()
                m = sort(abs.(x); rev = true)
                reference = sum(k -> f.λ[k] * m[k], 1:n)
                isapprox(f(x), reference; atol = tolerance(T), rtol = tolerance(T)) ||
                    error("SLOPE value differs from sorted-magnitude sum")
            end
            addcase!(cases, "ordered", "SLOPE/value", () -> f(x), checkslope!;
                     T, shape = string(n), work = n)
        end
    end
end

function group_layout_cases!(cases, rep)
  for T in (Float32, Float64), K in (1_000, 10_000), indices in (false, true)
    ranges = [8k-7:8k for k in 1:K]
    groups = indices ? collect.(ranges) : ranges
    p = 8K
    rng = scenario_rng(rep, "group_layouts/$T/$K/$indices")
    v = zeros(T, p)
    v[1:64] .= T(2) .* randn(rng, T, 64)
    grad = ones(T, p)
    step, k = T(0.7), 4
    for F in (GroupNormL2, GroupNormLinf, SparseGroupNormL2)
      f = F === SparseGroupNormL2 ? F(T(0.2), T(0.3), groups) : F(T(0.3), groups)
      name = string(nameof(typeof(f)))
      reference = prox(f, v, step)
      reference_value = f(v)
      reference_prox_value = f(reference)
      reference_kkt = kkt_violation(f, grad, zeros(T, p), first(groups[k]))
      coordinate_reference = copy(v)
      coordinate_reference[groups[k]] = reference[groups[k]]
      for shared in (true, false)
        layout = string(indices ? "indices" : "ranges", shared ? "_shared" : "_distinct")
        source = shared ? f : groups
        x = AtomIterate(v, source)
        y = AtomIterate(T, p, source)
        z = AtomIterate(v, source)
        f(x); f(y); f(z)
        prepare!() = copyto!(z, x)
        function checkcoordinate!()
          prepare!()
          prox_coordinate!(z, f, k, step)
          close_enough(z, coordinate_reference)
        end
        checkkkt!() = isapprox(kkt_violation(f, grad, x, k), reference_kkt;
                              atol=tolerance(T), rtol=tolerance(T)) || error("KKT differs from dense reference")
        checkvalue!() = isapprox(f(x), reference_value; atol=tolerance(T), rtol=tolerance(T)) ||
          error("value differs from dense reference")
        checkprox!() = (prox!(y, f, x, step); close_enough(y, reference))
        function checkvprox!()
          value = value_and_prox!(y, f, x, step)
          close_enough(y, reference)
          isapprox(value, reference_prox_value; atol=tolerance(T), rtol=tolerance(T)) ||
            error("proximal value differs from dense reference")
        end
        addcase!(cases, "group_layouts", name * "/coordinate", () -> prox_coordinate!(z, f, k, step), checkcoordinate!;
                 T, shape=string(K), layout, prepare=prepare!, work=8)
        addcase!(cases, "group_layouts", name * "/kkt", () -> kkt_violation(f, grad, x, k), checkkkt!;
                 T, shape=string(K), layout, work=8)
        addcase!(cases, "group_layouts", name * "/value", () -> f(x), checkvalue!;
                 T, shape=string(K), layout, work=64)
        addcase!(cases, "group_layouts", name * "/prox!", () -> prox!(y, f, x, step), checkprox!;
                 T, shape=string(K), layout, work=64)
        addcase!(cases, "group_layouts", name * "/value_and_prox!", () -> value_and_prox!(y, f, x, step), checkvprox!;
                 T, shape=string(K), layout, work=64)
        if !shared
          other = F === SparseGroupNormL2 ? F(T(0.2), T(0.3), groups) : F(T(0.3), groups)
          function checkalternating!()
            checkvalue!()
            isapprox(other(x), other(v); atol=tolerance(T), rtol=tolerance(T)) ||
              error("alternating value differs from dense reference")
          end
          addcase!(cases, "group_layouts", name * "/value_alternating", () -> (f(x), other(x)), checkalternating!;
                   T, shape=string(K), layout, work=128)
        end
      end
    end
  end
end

function smooth_cases!(cases, rep; quick=false)
    for T in (quick ? (Float64,) : (Float32,Float64)), (n,p) in (quick ? ((128,32),) : ((128,32),(1024,256)))
        rng = scenario_rng(rep,"smooth/$T/$n/$p")
        X,y,x = randn(rng,T,n,p),randn(rng,T,n),T(0.03).*randn(rng,T,p)
        for (name,f) in (("least_squares",LeastSquaresLoss(y,X)), ("logistic",LogisticLoss(rand(rng,0:1,n),X)),
                  ("poisson",PoissonLoss(rand(rng,0:3,n),X)), ("huber",HuberLoss(y,X,T(0.5))),
                  ("logistic_weighted_offset",LogisticLoss(rand(rng,0:1,n),X;weights=rand(rng,T,n),offset=randn(rng,T,n))))
            quick && !(name in ("least_squares", "logistic")) && continue
            addgrad!(cases,"smooth",name,f,x; profile=T==Float64 && n==1024)
        end
        ls = LeastSquaresLoss(y,X)
        check_global_curvature!() = begin
            bound = lipschitz(ls)
            reference = opnorm(X)^2
            isfinite(bound) && bound >= reference * (1 - tolerance(T)) ||
                error("invalid global curvature bound")
        end
        addcase!(cases,"setup","least_squares_lipschitz",() -> lipschitz(ls),check_global_curvature!;
                 T,shape="$(n)x$p",phase="setup",zero_alloc=false,work=n*p)
        addprox!(cases,"least_squares","tall",ls,x;profile=T==Float64 && n==1024)
        addcold!(cases,"least_squares","tall",() -> LeastSquaresLoss(y,X),x)
        B = randn(rng,T,p,3); Y = randn(rng,T,n,3)
        addgrad!(cases,"least_squares_multi","three_responses",LeastSquaresLoss(Y,X),B)
        L = zeros(T,p)
        checkL!() = (coordinate_lipschitz!(L,ls); close_enough(L,vec(sum(abs2,X;dims=1))))
        addcase!(cases,"coordinates","least_squares_curvatures",() -> coordinate_lipschitz!(L,ls),checkL!;T,shape="$(n)x$p",work=n*p)
        G = zeros(T,p); penalty=NormL1(T(0.3)); step=T(0.1)/sum(abs2,X)
        checkG!() = (gradient_mapping!(G,ls,penalty,x,step);all(isfinite,G) || error("nonfinite gradient mapping"))
        addcase!(cases,"calculus","gradient_mapping!",() -> gradient_mapping!(G,ls,penalty,x,step),checkG!;T,shape="$(n)x$p",work=n*p)
        H=Matrix(Symmetric(X'X+I)); q=QuadraticFunction(H,randn(rng,T,p),zero(T))
        addgrad!(cases,"quadratic","dense",q,x)
        addprox!(cases,"quadratic","dense",q,x)
        addcold!(cases,"quadratic","dense",() -> QuadraticFunction(H,zeros(T,p),zero(T)),x)
        for f in (Regularize(ls,T(0.3)), Precompose(SqrNormL2(T(0.3)),T(-2)),
                  Translate(SqrNormL2(T(0.3)),ones(T,p)), Sum(ls,SqrNormL2(T(0.3))))
            addgrad!(cases,"compositions",string(nameof(typeof(f))),f,x)
        end
        # An actual fixed-work solver trace, with reset outside the measured block.
        β=zeros(T,p); grad=similar(β); r=copy(y); a=vec(sum(abs2,X;dims=1))
        prep_cd!() = (fill!(β,0);copyto!(r,y))
        function cdtrace!()
            for sweep in 1:5, j in 1:p
                old=β[j]; col=view(X,:,j)
                new=prox_entry(penalty,old+dot(col,r)/a[j],j,inv(a[j]))
                β[j]=new; axpy!(old-new,col,r)
            end
            β
        end
        checkcd!() = (prep_cd!();cdtrace!(); close_enough(r,y-X*β); ls(β)+penalty(β)<=sum(abs2,y)/2+1e-3 || error("CD objective increased"))
        addcase!(cases,"solver_traces","lasso_5_sweeps",cdtrace!,checkcd!;T,shape="$(n)x$p",prepare=prep_cd!,profile=T==Float64 && n==1024,work=5n*p)
    end
    for T in (quick ? (Float64,) : (Float32,Float64)), (n,p) in (quick ? ((32,128),) : ((32,128),(128,1024)))
        rng=scenario_rng(rep,"wide/$T/$n/$p")
        X,y,x=randn(rng,T,n,p),randn(rng,T,n),randn(rng,T,p)
        f=LeastSquaresLoss(y,X)
        addprox!(cases,"least_squares","wide",f,x;profile=T==Float64 && p==1024)
        addcold!(cases,"least_squares","wide",() -> LeastSquaresLoss(y,X),x)
        sparsex=SparseIterate(sprandn(rng,T,p,0.01))
        addgrad!(cases,"sparse_smooth","least_squares",f,sparsex;second=false)
        A=randn(rng,T,4,p); b=zeros(T,4); C=IndAffine(A,b)
        constrained=Constrained(f,C)
        addprox!(cases,"affine_constrained","least_squares",constrained,x)
        addcold!(cases,"affine_constrained","least_squares",() -> Constrained(LeastSquaresLoss(y,X),C),x)
    end
end

function matrix_cases!(cases, rep; quick=false)
    for T in (quick ? (Float64,) : (Float32,Float64)), p in (quick ? (32,) : (32,256))
        rng=scenario_rng(rep,"matrices/$T/$p")
        M=randn(rng,T,p,p÷2); A=Matrix(Symmetric(randn(rng,T,p,p)))
        for f in (NuclearNorm(T(0.5)), IndBallRank(4), SingularValue(NormL1(T(0.5))))
            addprox!(cases,"svd",string(nameof(typeof(f))),f,M;profile=T==Float64 && p==256)
        end
        for f in (IndPSD(),IndPSD(rank=4),TracePSD(T(0.2)),Spectral(NormL1(T(0.2))))
            addprox!(cases,"eigen",string(nameof(typeof(f)),f isa IndPSD && f.rank !== nothing ? "_rank4" : ""),f,A;
                     profile=T==Float64 && p==256)
        end
        for f in (IndPattern(rand(rng,p,p).>0.8), IndBanded(3,p), IndToeplitz(symmetric=true))
            addprox!(cases,"matrix_sets",string(nameof(typeof(f))),f,A;profile=T==Float64 && p==256)
        end
        B=randn(rng,T,2p,p); S=Matrix(Symmetric(B'B/(2p)+I)); P=Matrix(T(1)I,p,p)
        addgrad!(cases,"logdet","positive_definite",LogDetLoss(S),P;profile=T==Float64 && p==256)
        addprox!(cases,"logdet","positive_definite",LogDetLoss(S),P)
        kron=KronQuadratic(S,Matrix(T(1)I,4,4),randn(rng,T,p,4))
        addgrad!(cases,"quadratic","kron",kron,randn(rng,T,p,4);profile=T==Float64 && p==256)
        sp=SymmetricSparseIterate(A .* (abs.(A).>T(1.5)))
        addprox!(cases,"sparse_penalties","symmetric_NormL1",NormL1(T(0.3)),sp;layout="symmetric_sparse")
    end
    for T in (quick ? (Float64,) : (Float32,Float64)), p in (quick ? (128,) : (128,512))
        rng=scenario_rng(rep,"partial/$T/$p")
        U=Matrix(qr(randn(rng,T,p,4)).Q)[:,1:4]
        V=Matrix(qr(randn(rng,T,p÷2,4)).Q)[:,1:4]
        M=U*Diagonal(T[20,16,13,10])*V' + T(0.001).*randn(rng,T,p,p÷2)
        A=U*Diagonal(T[20,16,13,10])*U' + T(0.001).*Matrix(Symmetric(randn(rng,T,p,p)))
        for (name,f,x) in (("nuclear",NuclearNorm(T(0.5);partial=true),M),
                           ("rank4",IndBallRank(4;partial=true),M),
                           ("psd_rank4",IndPSD(rank=4,partial=true),A),
                           ("psd",IndPSD(partial=true),A - T(0.5)*I),
                           ("trace",TracePSD(T(0.5);partial=true),A))
            full = name=="nuclear" ? NuclearNorm(T(0.5)) : name=="rank4" ? IndBallRank(4) :
                   name=="psd_rank4" ? IndPSD(rank=4) : name=="psd" ? IndPSD() : TracePSD(T(0.5))
            previous=x .+ T(1e-5).*randn(rng,T,size(x)); scratch=similar(x)
            # The warm call always starts from a nearby point, not its own answer.
            prep!() = prox!(scratch,f,previous,T(0.7))
            addprox!(cases,"partial",name,f,x;prepare=prep!,profile=T==Float64 && p==512,reference_function=full)
            addprox!(cases,"partial",name*"_full",full,x;value=false)
            makef = name=="nuclear" ? () -> NuclearNorm(T(0.5);partial=true) :
                    name=="rank4" ? () -> IndBallRank(4;partial=true) :
                    name=="psd_rank4" ? () -> IndPSD(rank=4,partial=true) :
                    name=="psd" ? () -> IndPSD(partial=true) : () -> TracePSD(T(0.5);partial=true)
            addcold!(cases,"partial",name,makef,x)
        end
        if !quick && p == 128
            # A high-rank input exercises the full-decomposition fallback.
            fallback_input = randn(rng,T,64,32)
            fallback_function = NuclearNorm(T(0.01);partial=true)
            fallback_scratch = similar(fallback_input)
            addprox!(cases,"partial","high_rank_fallback",fallback_function,fallback_input;
                     prepare=() -> prox!(fallback_scratch,fallback_function,fallback_input,T(0.7)),
                     reference_function=NuclearNorm(T(0.01)))
        end
    end
end

function constrained_cases!(cases,rep; quick=false)
    for T in (quick ? (Float64,) : (Float32,Float64)), n in (quick ? (256,) : (256,100_000))
        rng=scenario_rng(rep,"constrained/$T/$n"); x=randn(rng,T,n)
        fs=(Constrained(NormL1(T(0.3)),IndBox(T(-1),T(1))),
            Constrained(SCAD(T(0.3)),IndBox(T(-1),T(1))),
            Constrained(NormL1(T(0.3)),IndHyperplane(one(T),zero(T))),
            Constrained(IndBox(zero(T),one(T)),IndHyperplane(one(T),T(n÷2))),
            Constrained(ElasticNet(T(0.3),T(0.4)),IndSimplex(T(n/10))),
            Constrained(NormL1(T(0.3)),IndBallL2(one(T))),
            Constrained(NormL1(T(0.3)),IndSphereL2(one(T))))
        for (name,f) in zip(("l1_box","scad_box","zero_sum_lasso","capped_simplex","simplex_elastic_net","l1_ball","l1_sphere"),fs)
            previous=x .+ T(0.001).*randn(rng,T,n); scratch=similar(x)
            prep!() = prox!(scratch,f,previous,T(0.7))
            addprox!(cases,"constrained",name,f,x;prepare=prep!,profile=T==Float64 && n==100_000)
            # Reset the multiplier, retaining allocated workspace, before a cold call.
            if name in ("zero_sum_lasso", "capped_simplex", "simplex_elastic_net")
                cold!() = (f.last[] = T(NaN); nothing)
                addprox!(cases,"constrained",name*"_cold",f,x;prepare=cold!)
            end
        end
        for (name,f) in (("scaled",T(2)*NormL1(T(0.3))),
                         ("translated",Translate(NormL1(T(0.3)),ones(T,n))),
                         ("tilted",Tilt(NormL1(T(0.3)),fill(T(0.1),n))),
                         ("conjugate",Conjugate(NormL2(T(0.3)))),
                         ("regularize",Regularize(NormL1(T(0.3)),T(0.4))),
                         ("precompose",Precompose(NormL1(T(0.3)),T(-2))),
                         ("envelope",MoreauEnvelope(NormL1(T(0.3)),T(0.5))))
            addprox!(cases,"compositions",name,f,x;profile=T==Float64 && n==100_000 && name=="envelope")
        end
        # Dykstra has an independent specialized reference for this intersection.
        if n == 256
            C=IndIntersection(IndNonnegative(),IndHyperplane(one(T),T(10));warmstart=false)
            out=similar(x)
            step=T(0.7)
            check!() = (prox!(out,C,x,step);C.converged || error("Dykstra failed to converge");close_enough(out,prox(IndSimplex(T(10)),x)))
            addcase!(cases,"intersection","simplex_cold_corrections",()->prox!(out,C,x,step),check!;
                     T,shape=string(n),profile=T==Float64,work=n)
        end
    end
end

function tensor_cases!(cases,rep; quick=false)
    for T in (quick ? (Float64,) : (Float32,Float64)), p in (quick ? (8,) : (8,48))
        rng=scenario_rng(rep,"tensors/$T/$p"); X=randn(rng,T,p,p,4)
        X=(X+permutedims(X,(2,1,3)))./2
        for (name,f) in (("fiber_l2",FiberSum(NormL2(T(0.3)))),
                         ("fiber_tv",FiberSum(TotalVariation1D(T(0.3)))),
                         ("fiber_symmetric_l2",FiberSum(NormL2(T(0.3));symmetric=true,diagonal=nothing)),
                         ("slice_l1",SliceSum([NormL1(T(0.3)) for _ in 1:4])))
            addprox!(cases,"tensors",name,f,X;profile=T==Float64 && p==48)
        end
    end
end

function product_cases!(cases,rep; quick=false)
    for T in (quick ? (Float64,) : (Float32,Float64)), p in (quick ? (256,) : (256,1024)), density in (quick ? (0.01,) : (0.01,0.2)), triangle in (quick ? (:U,) : (:U,:L))
        rng=scenario_rng(rep,"products/$T/$p/$density/$triangle")
        A=Symmetric(randn(rng,T,p,p),triangle)
        xs=SparseIterate(sprandn(rng,T,p,density))
        groups=[j:min(j+7,p) for j in 1:8:p]
        xa=AtomIterate(dense(xs),GroupNormL2(T(0.3),groups))
        for (layout,x) in (("sparse",xs),("atom",xa))
            y=zeros(T,p); ref=A*dense(x)
            check!()=(mul!(y,A,x);close_enough(y,ref))
            addcase!(cases,"products","symmetric_$triangle",()->mul!(y,A,x),check!;
                     T,shape=string(p),layout="$(layout)_$density",evals=10,
                     profile=T==Float64 && p==1024 && density==0.01 && triangle==:U,work=nnz(xs))
        end
    end
    n = 512
    for T in (quick ? (Float64,) : (Float32,Float64)), p in (quick ? (256,) : (256,1024)), density in (quick ? (0.01,) : (0.01,0.2))
        rng=scenario_rng(rep,"products/strided/$T/$p/$density")
        A=randn(rng,T,n,p)
        xs=SparseIterate(sprandn(rng,T,p,density))
        xm=SparseIterate(sprandn(rng,T,p,4,density))
        groups=[j:min(j+7,p) for j in 1:8:p]
        xa=AtomIterate(dense(xs),GroupNormL2(T(0.3),groups))
        xt=SparseIterate(sprandn(rng,T,n,density))
        for (name,B,x) in (("strided_vector",A,xs),("strided_matrix",A,xm),
                           ("strided_atom",A,xa),("transposed_vector",transpose(A),xt))
            shape="$(n)x$p" * (x === xm ? "x4" : "")
            layout="$(x isa AtomIterate ? "atom" : "sparse")_$density"
            xd=dense(x)
            out=x isa AbstractVector ? zeros(T,size(B,1)) : zeros(T,size(B,1),size(x,2))
            reference=B*xd
            work=size(B,1)*count(!iszero,xd)
            check!()=(mul!(out,B,x);close_enough(out,reference))
            addcase!(cases,"products",name*"/mul!",()->mul!(out,B,x),check!;
                     T,shape,layout,evals=10,work)
            original=fill(T(0.25),size(out))
            α,β=T(1.5),T(0.5)
            prepare!()=copyto!(out,original)
            function checkscaled!()
                expected=copy(original)
                mul!(expected,B,xd,α,β)
                prepare!(); mul!(out,B,x,α,β)
                close_enough(out,expected)
            end
            addcase!(cases,"products",name*"/mul!_scaled",()->mul!(out,B,x,α,β),checkscaled!;
                     T,shape,layout,prepare=prepare!,evals=1,work)
        end
    end
end

const CASE_GROUPS = ("vectors", "smooth", "matrices", "constraints", "tensors", "products", "blocks")

function build_cases(rep; suite="quick", groups=collect(CASE_GROUPS))
    suite in ("quick", "extended") || error("PB_SUITE must be quick or extended")
    isempty(groups) && error("Select at least one benchmark group")
    all(in(CASE_GROUPS), groups) || error("PB_GROUPS must select from $(join(CASE_GROUPS, ','))")
    length(unique(groups)) == length(groups) || error("Duplicate benchmark group")
    cases=PerfCase[]
    builders=(separable_cases!,smooth_cases!,matrix_cases!,constrained_cases!,tensor_cases!,product_cases!,block_cases!)
    # Select groups and sizes before allocating data or constructing function objects.
    for (group,builder) in zip(CASE_GROUPS,builders)
        group in groups && builder(cases,rep;quick=suite=="quick")
    end
    suite == "extended" && "vectors" in groups && group_layout_cases!(cases, rep)
    ids=[c.meta.id for c in cases]
    length(unique(ids))==length(ids) || error("duplicate case IDs")
    cases
end
