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
    # Matrix coefficients with entries stored in random order, at densities on both
    # sides of the cutoff of the five-argument product (1/32), for short and long columns.
    quick && return
    for T in (Float32,Float64), m in (64,2048), density in (0.02,0.05,0.1)
        rng=scenario_rng(rep,"products/switch/$T/$m/$density")
        p,q=1024,4
        A=randn(rng,T,m,p)
        I,J,V=findnz(sprandn(rng,T,p,q,density))
        perm=randperm(rng,length(I))
        x=SparseIterate(T,p,q)
        for k in perm; x[I[k],J[k]]=V[k]; end
        xd=dense(x)
        out=zeros(T,m,q)
        reference=A*xd
        work=m*count(!iszero,xd)
        check!()=(mul!(out,A,x);close_enough(out,reference))
        addcase!(cases,"products","switch_matrix/mul!",()->mul!(out,A,x),check!;
                 T,shape="$(m)x$(p)x$q",layout="random_$density",evals=10,work)
    end
end

# SymmetricSparseIterate owner cases (issue #391, Phase 0; acceptance guide section 4).
# They use only calls that exist both in 0.5.x and in 0.6 (X[i, j], activecoordinates,
# prox_coordinate! with a lower-triangle coordinate ID, mul! and Matrix on the matrix
# view when there is one), so one harness run on each revision gives the before/after
# comparison. No case here depends on the owner being an AbstractMatrix.
owner_matrix(X) = isdefined(ProximalBase, :matrixview) ? ProximalBase.matrixview(X) : X

function owner_fixture(rng, T, p, s)
    q = p * (p + 1) ÷ 2
    ids = randperm(rng, q)[1:s]               # distinct lower-triangle coordinate IDs
    vals = T.(randn(rng, s)) .+ T(2)          # bounded away from zero
    X = SymmetricSparseIterate(T, p)
    for (k, v) in zip(ids, vals)
        r, c = ProximalBase.tril_cartesian(p, k)
        X[r, c] = v
    end
    I = Int[]; J = Int[]; V = T[]
    for (k, v) in zip(ids, vals)
        r, c = ProximalBase.tril_cartesian(p, k)
        push!(I, r); push!(J, c); push!(V, v)
        r == c || (push!(I, c); push!(J, r); push!(V, v))
    end
    (; X, ids, vals, S=sparse(I, J, V, p, p))
end

function owner_cases!(cases, rep; quick=false)
    T = Float64
    # (p, s): s = 100 with p growing, then p = 1000 with s growing
    grid = quick ? [(500, 100), (1000, 100), (1000, 1000)] :
           [(p, 100) for p in (500, 1000, 2000, 4000)] ∪ [(1000, s) for s in (10^3, 10^4, 10^5)]
    npos = 1024
    for (p, s) in grid
        rng = scenario_rng(rep, "owner/$T/$p/$s")
        fx = owner_fixture(rng, T, p, s)
        X, S = fx.X, fx.S
        shape = "p$(p)_s$(s)"
        q = p * (p + 1) ÷ 2
        active = [ProximalBase.tril_cartesian(p, fx.ids[1 + (i - 1) % s]) for i in 1:npos]   # cycles when s < npos
        activeset = Set(fx.ids)
        # distinct, initially inactive coordinate IDs (never repeated)
        absent_ids = Int[]; seen = Set{Int}()
        while length(absent_ids) < npos
            k = rand(rng, 1:q)
            (k in activeset || k in seen) && continue
            push!(seen, k); push!(absent_ids, k)
        end
        absent = [ProximalBase.tril_cartesian(p, k) for k in absent_ids]
        # four read cases of nread reads each: active and absent coordinates, each read through
        # X[r, c] with r >= c and through X[c, r]; each is checked against the CSC reference
        nread = 256
        for (kind, base) in (("active", active), ("absent", absent)), (side, flip) in (("lower", false), ("upper", true))
            pos = [flip ? (c, r) : (r, c) for (r, c) in base[1:nread]]
            expected_read = sum(S[r, c] for (r, c) in pos)
            read_op() = (acc = zero(T); @inbounds for (r, c) in pos; acc += X[r, c]; end; acc)
            check_read!() = isapprox(read_op(), expected_read; rtol=1e-12) || error("owner reads differ from the CSC reference")
            addcase!(cases, "owner", "read_X[i,j]", read_op, check_read!; T, shape, layout="$(kind)_$(side)", work=nread)
        end
        # writes to already active coordinates, with a changing value
        wpos = active
        Xw = copy(fx.X)                      # this case's own owner
        wstate = Ref(zero(T))
        write_op() = (val = wstate[] += one(T); @inbounds for (r, c) in wpos; Xw[r, c] = val; end; nothing)
        function check_write!()
            write_op()
            val = wstate[]
            all(Xw[r, c] == val for (r, c) in wpos) || error("owner writes were not stored")
            length(collect(activecoordinates(Xw))) == s || error("a write to an active coordinate changed the working set")
        end
        addcase!(cases, "owner", "write_X[i,j]_active", write_op, check_write!; T, shape, layout="active", work=length(wpos))
        # first activation: always an absent coordinate; restored outside the timed region
        Xa = copy(fx.X)
        # a batch of 256 distinct, initially inactive coordinates per call: one activation is below the timer floor
        batch_ids = absent_ids[1:256]
        batch = absent[1:256]
        activate_op() = (@inbounds for (r, c) in batch; Xa[r, c] = one(T); end; nothing)
        function restore!()               # outside the timed region
            for (r, c) in batch; Xa[r, c] = zero(T); end
            dropzeros!(Xa)
            any(k -> isactive(Xa, k), batch_ids) && error("a batch coordinate is still active: not a first activation")
            nothing
        end
        function check_activate!()
            restore!()
            length(unique(batch_ids)) == length(batch_ids) || error("batch coordinates are not distinct")
            activate_op()
            all(k -> isactive(Xa, k), batch_ids) || error("activation did not add every coordinate")
            length(collect(activecoordinates(Xa))) == s + length(batch_ids) || error("unexpected working set")
            restore!()
            length(collect(activecoordinates(Xa))) == s || error("restore did not recover the working set")
        end
        addcase!(cases, "owner", "first_activation", activate_op, check_activate!; T, shape, layout="absent_distinct_batch",
                 phase="first_activation", prepare=restore!, evals=1, zero_alloc=true, work=length(batch))
        # sweeps over the active coordinates
        Xs = copy(fx.X)
        sweep_ids() = (acc = 0; for k in activecoordinates(Xs); acc += k; end; acc)
        expected_sum = sum(fx.ids)
        addcase!(cases, "owner", "sweep_activecoordinates", sweep_ids,
                 () -> sweep_ids() == expected_sum || error("sweep visited the wrong coordinates");
                 T, shape, layout="trivial_update", work=s)
        f = NormL1(T(0.1))
        step = T(0.5)
        prox_sweep() = (for k in activecoordinates(Xs); prox_coordinate!(Xs, f, k, step); end; nothing)
        vstate = copy(fx.vals)
        prep_sweep!() = (for (k, v) in zip(fx.ids, fx.vals); r, c = ProximalBase.tril_cartesian(p, k); Xs[r, c] = v; end; nothing)
        function check_sweep!()
            prep_sweep!(); prox_sweep()
            ok = all(zip(fx.ids, fx.vals)) do (k, v)
                r, c = ProximalBase.tril_cartesian(p, k)
                isapprox(Xs[r, c], sign(v) * max(abs(v) - step * T(0.1), zero(T)); atol=1e-12)
            end
            ok || error("coordinate sweep gave the wrong soft threshold")
            prep_sweep!()
        end
        addcase!(cases, "owner", "sweep_prox_coordinate!", prox_sweep, check_sweep!; T, shape, layout="NormL1",
                 prepare=prep_sweep!, evals=1, work=s)
        # matrix product with a dense vector

        v = randn(rng, T, p); y = zeros(T, p); ref = S * v
        A = owner_matrix(X)
        mul_op() = (mul!(y, A, v); nothing)
        check_mul!() = (mul_op(); isapprox(y, ref; rtol=1e-10, atol=1e-10) || error("owner product differs from the CSC product"))
        addcase!(cases, "owner", "mul!", mul_op, check_mul!; T, shape, layout="owner", evals=10, work=s)
        # baselines in the same session
        ycsc = zeros(T, p)
        addcase!(cases, "owner", "mul!", () -> (mul!(ycsc, S, v); nothing),
                 () -> (mul!(ycsc, S, v); isapprox(ycsc, ref; rtol=1e-10, atol=1e-10) || error("CSC product mismatch"));
                 T, shape, layout="csc_baseline", evals=10, work=s)
        if p <= 2000
            D = Symmetric(Matrix(S), :L); yd = zeros(T, p)
            addcase!(cases, "owner", "mul!", () -> (mul!(yd, D, v); nothing),
                     () -> (mul!(yd, D, v); isapprox(yd, ref; rtol=1e-10, atol=1e-10) || error("dense product mismatch"));
                     T, shape, layout="dense_symmetric_baseline", evals=10, work=p * p)
        end
        # conversion to a dense matrix
        if p <= 2000 || s == 100
            addcase!(cases, "owner", "Matrix", () -> (Matrix(A); nothing),
                     () -> (M = Matrix(A); sparse(M) == S || error("Matrix(owner) differs from the CSC reference"));
                     T, shape, layout="owner", phase="allocating", zero_alloc=false, evals=1, work=p * p)
        end
    end
    # construction (empty owner), quadratic in p
    for p in (quick ? (500, 1000) : (500, 1000, 2000, 4000))
        addcase!(cases, "owner", "construct", () -> (SymmetricSparseIterate(T, p); nothing),
                 function ()
                     X = SymmetricSparseIterate(T, p); q = p * (p + 1) ÷ 2
                     size(X, 1) == p || error("wrong size")
                     # retained memory: one T array and two Int arrays of length q, plus headers
                     abs(Base.summarysize(X) - (sizeof(T) + 2 * sizeof(Int)) * q) <= 1024 || error("unexpected retained size")
                 end;
                 T, shape="p$p", layout="empty", phase="construct", zero_alloc=false, evals=1, work=p * (p + 1) ÷ 2)
    end
    handle_sweep_cases!(cases, rep; quick)
end

# Sweep over activecoordinates(x) for SparseIterate and AtomIterate (issue #405): the handle
# must cost nothing over the view it replaced. Uses only calls that exist before and after.
function handle_sweep_cases!(cases, rep; quick=false)
    T = Float64
    for s in (quick ? (1000, 100_000) : (100, 1000, 10_000, 100_000))
        n = 4 * s
        rng = scenario_rng(rep, "handle/$T/$s")
        xs = SparseIterate(T, n)
        ids = randperm(rng, n)[1:s]
        for k in ids; xs[k] = one(T); end
        sweep_sparse() = (acc = 0; for k in activecoordinates(xs); acc += k; end; acc)
        addcase!(cases, "iterates", "sweep_activecoordinates", sweep_sparse,
                 () -> sweep_sparse() == sum(ids) || error("sweep visited the wrong coordinates");
                 T, shape="n$n/s$s", layout="SparseIterate", work=s)
        groups = [(2k - 1):(2k) for k in 1:(2 * s)]
        xa = AtomIterate(T, 4 * s, groups)
        gids = randperm(rng, length(groups))[1:s]
        for k in gids; xa[first(groups[k])] = one(T); end
        sweep_atom() = (acc = 0; for k in activecoordinates(xa); acc += k; end; acc)
        addcase!(cases, "iterates", "sweep_activecoordinates", sweep_atom,
                 () -> sweep_atom() == sum(gids) || error("sweep visited the wrong coordinates");
                 T, shape="n$(4 * s)/s$s", layout="AtomIterate", work=s)
    end
end

# ---- helpers shared by the sparse and atom groups ----
#
# One working iterate and one immutable snapshot serve all the cases of a scenario; the cases run
# one at a time. Every validation starts and ends by restoring the working iterate with a state copy
# (copyto! from the snapshot, which is a state copy on 0.5.x and on 0.6), so that it does not matter
# what the case that ran before it left behind. A case that changes state restores it in `prepare`.

# k values ±(1 + u), u ~ U[0, 1), with a random sign: bounded away from zero.
function fixture_values(rng, T, k)
    T[(rand(rng, Bool) ? one(T) : -one(T)) * T(1 + rand(rng)) for _ in 1:k]
end

# k items of `src` in order, starting again from the first when `src` has fewer
cycled(src, k) = [src[1 + (j - 1) % length(src)] for j in 1:k]

# the values with those slots set to exact zeros
function with_zeros(vals, slots)
    v = copy(vals)
    v[slots] .= zero(eltype(v))
    v
end

# the slots of a fraction of the stored values, at random places in storage order
zero_slots(rng, s, fraction) = randperm(rng, s)[1:round(Int, fraction * s)]

# an iterate with `vals` stored at the linear indices `ids`, in this slot order; an exact zero is
# stored with activate!, as zeros are not stored by assignment
function build_iterate(T, dims, ids, vals)
    x = SparseIterate(T, dims...)
    for (i, v) in zip(ids, vals)
        iszero(v) ? activate!(x, i) : (x[i] = v)
    end
    x
end

# the dense array the iterate must equal: the reference of every check
function dense_array(T, dims, ids, vals)
    D = zeros(T, dims...)
    D[ids] = vals
    D
end

# a validation that starts and ends with the shared iterate restored
restored(restore!, check) = function ()
    restore!()
    check()
    restore!()
    nothing
end

# |a - b| <= tolerance(T) * scale, for sums whose rounding depends on the order of the terms
function close_scalar(a, b, scale, T)
    abs(a - b) <= tolerance(T) * scale || error("scalar differs from the dense reference: $a versus $b")
    nothing
end

# Version-dependent spelling is resolved once, when the case is built, with the 0.5.x spelling as the
# fallback, and handed to a function that builds the timed closures: they see a concrete function
# type, and nothing is tested inside the timed call. The catalog of case IDs does not depend on it.
private_nzval(x) = x.nzval
stored_values_by_field(x) = hasfield(typeof(x), :nzval) ? private_nzval : activevalues

# a repetition count for an operation of `units` steps, so that a call takes about 2 µs on the pinned source
repeats_for(units, target) = max(1, cld(target, units))

# ---- the sparse group: SparseIterate vectors and matrices ----
#
# Fixtures: vectors (n, s) with s distinct random coordinates; matrices with exactly round(d·L)
# stored entries (so that a repetition never crosses the 1/32 switch of the matrix mul!); dense
# operands built once per (T, shape) and shared. Only calls that exist on 0.5.x and on 0.6 are used.

const SPARSE_DENSITIES = ((0.005, "0.5"), (0.03, "3"), (0.2, "20"))

# the loops of the cases are functions of their arguments, so that the timed closures are thin
function sum_reads(x, ids, ::Type{T}) where {T}
    acc = zero(T)
    @inbounds for k in ids
        acc += x[k]
    end
    acc
end
function write_values!(x, ids, vals)
    @inbounds for j in eachindex(ids)
        x[ids[j]] = vals[j]
    end
    nothing
end
function activate_all!(x, ids)
    @inbounds for k in ids
        activate!(x, k)
    end
    nothing
end
function count_active(x, ids)
    c = 0
    @inbounds for k in ids
        c += isactive(x, k)
    end
    c
end
# The trivial update of the sweeps adds a zero: `x[i] = x[i]` alone is removed by the compiler (the
# store of a value just loaded; a 10⁴-entry sweep took 10 ns on Julia 1.13), and the sweep would
# measure nothing. Adding +0.0 is not an identity for -0.0, so the compiler keeps it; no stored value
# is -0.0, so the iterate is unchanged.
function full_sweep!(x)
    z = zero(eltype(x))
    @inbounds for i in 1:ncoordinates(x)
        x[i] = x[i] + z
    end
    nothing
end
function active_sweep!(x)
    z = zero(eltype(x))
    @inbounds for k in activecoordinates(x)
        x[k] = x[k] + z
    end
    nothing
end
function sum_active_ids(x)
    acc = 0
    for k in activecoordinates(x)
        acc += k
    end
    acc
end
function sum_active_values(x, ::Type{T}) where {T}
    acc = zero(T)
    for v in activevalues(x)
        acc += v
    end
    acc
end
function coordinate_sweep!(x)
    c = coordinates(x)
    z = zero(eltype(c))
    @inbounds for k in 1:length(c)
        c[k] = c[k] + z
    end
    nothing
end
function index_sweep!(x)
    z = zero(eltype(x))
    @inbounds for k in 1:length(x)
        x[k] = x[k] + z
    end
    nothing
end
function rmul_repeat!(x, a, r)
    for _ in 1:r
        rmul!(x, a)
    end
    nothing
end
function prox_batch!(x, f, ids, us, γ)
    @inbounds for j in eachindex(ids)
        prox_coordinate!(x, f, ids[j], us[j], γ)
    end
    nothing
end

# the three value loops of the momentum history (acceptance guide §3.4), over the stored values
# through `getvals`: the accessor of 0.6, or the private array that the downstream code reads today
momentum_values!(getvals::G, buf, x, n) where {G} = copyto!(buf, 1, getvals(x), 1, n)
momentum_setvalues!(getvals::G, x, buf, n) where {G} = copyto!(getvals(x), 1, buf, 1, n)
function momentum_dot(getvals::G, x, y, xcur, xprev, n) where {G}
    av = getvals(x)
    d = zero(eltype(y))
    @inbounds for j in 1:n
        d += (av[j] - y[j]) * (xcur[j] - xprev[j])
    end
    d
end
# the function barrier: the closures are built where `getvals` has a concrete type
function momentum_ops(getvals::G, x, buf, w, y, xcur, xprev, n) where {G}
    (() -> momentum_values!(getvals, buf, x, n),
     () -> momentum_setvalues!(getvals, x, w, n),
     () -> momentum_dot(getvals, x, y, xcur, xprev, n))
end

function replace_cheap!(x, A)
    copyto!(dropzeros!(fill!(x, 0)), A)
    nothing
end
function replace_staged!(x, A)
    t = similar(x)
    for i in eachindex(t)
        t[i] = A[i]
    end
    copyto!(x, t)
    nothing
end

# FW-4: a state copy with a conversion, on dedicated iterates. The source holds integral values and is
# built from scratch, so that its whole `nzval` array (the stale tail included, which the 0.5.x dense
# branch converts too) converts to Int; it is never changed. Only the success path is measured.
function sparse_convert_cases!(cases, rep, dims, ids, key, shape)
    k = length(ids)
    rng = scenario_rng(rep, key * "/integral")
    iv = Float64[(rand(rng, Bool) ? 1 : -1) * rand(rng, 1:7) for _ in 1:k]
    src = build_iterate(Float64, dims, ids, iv)
    dsrc = dense_array(Float64, dims, ids, iv)
    # the branch of the copy kernel once source and destination hold the same entries (4(nnz + nnz) < length)
    branch = 4 * (2 * k) < prod(dims) ? "sparse_branch" : "dense_branch"
    for Td in (Float64, Float32, Int)
        dest = SparseIterate(Td, dims...)
        op = () -> (copyto!(dest, src); nothing)
        check = function ()
            for _ in 1:2        # the first call starts from an empty destination, the second from a full one
                copyto!(dest, src)
                Array(dest) == Td.(dsrc) || error("converted copy differs from the dense reference")
            end
            nothing
        end
        addcase!(cases, "sparse", "copyto!_convert", op, check; T=Float64, shape,
                 layout="to_$(Td)_$(branch)", zero_alloc=false, work=branch == "sparse_branch" ? k : prod(dims))
    end
end

function sparse_vector_cases!(cases, rep, T, n, s, vcache; rows=())
    key = "sparse/vector/$T/$n/$s"
    shape = "n$(n)_s$(s)"
    # one permutation: the active coordinates, a pool of absent ones (the batches of first activations,
    # the block trial) and the extra coordinates of the second iterate are disjoint slices of it
    perm = randperm(scenario_rng(rep, key * "/ids"), n)
    ids = perm[1:s]
    pool = perm[s+1:s+4096]
    ids_y = vcat(ids[1:s÷2], perm[s+4097:s+4096+(s-s÷2)])
    vals = fixture_values(scenario_rng(rep, key * "/values"), T, s)
    vals_y = fixture_values(scenario_rng(rep, key * "/values_y"), T, s)
    slots = zero_slots(scenario_rng(rep, key * "/zeros"), s, 0.5)
    slots_y = zero_slots(scenario_rng(rep, key * "/zeros_y"), s, 0.5)
    x = build_iterate(T, (n,), ids, vals)
    snap = copy(x)
    d = dense_array(T, (n,), ids, vals)
    y = build_iterate(T, (n,), ids_y, vals_y)
    dy = dense_array(T, (n,), ids_y, vals_y)
    vals50 = with_zeros(vals, slots)
    vals_y50 = with_zeros(vals_y, slots_y)
    x50 = build_iterate(T, (n,), ids, vals50)
    y50 = build_iterate(T, (n,), ids_y, vals_y50)
    v = get!(() -> randn(scenario_rng(rep, "sparse/v/$T/$n"), T, n), vcache, (T, n))::Vector{T}
    restore_state!() = (copyto!(x, snap); nothing)
    guarded(check) = restored(restore_state!, check)

    # scalar reads: 256 reads of active and, separately, of absent coordinates
    nread = 256
    for (kind, src) in (("active", ids), ("absent", pool))
        rids = cycled(src, nread)
        expected = sum_reads(d, rids, T)
        addcase!(cases, "sparse", "read_x[i]", () -> sum_reads(x, rids, T),
                 guarded(() -> sum_reads(x, rids, T) == expected || error("reads differ from the dense vector"));
                 T, shape, layout=kind, zero_alloc=true, work=nread)
    end
    # writes to 1024 active coordinates, with the values they hold: the state is unchanged, and the
    # check first overwrites them
    wids = cycled(ids, 1024)
    wvals = d[wids]
    function check_write()
        write_values!(x, wids, -wvals)
        write_values!(x, wids, wvals)
        Array(x) == d || error("writes were not stored")
        nactive(x) == s || error("a write to an active coordinate changed the working set")
        nothing
    end
    addcase!(cases, "sparse", "write_x[i]_active", () -> write_values!(x, wids, wvals), guarded(check_write);
             T, shape, layout="active", zero_alloc=true, work=length(wids))
    # isactive: 256 queries, half of them active
    qids = shuffle(scenario_rng(rep, key * "/queries"), vcat(cycled(ids, 128), pool[1:128]))
    expected_active = count(!iszero, d[qids])
    addcase!(cases, "sparse", "isactive", () -> count_active(x, qids),
             guarded(() -> count_active(x, qids) == expected_active || error("isactive differs from the dense vector"));
             T, shape, layout="half_active", zero_alloc=true, work=length(qids))

    # first activation: a batch of distinct absent coordinates per call, by assignment and by activate!;
    # the companion has up to 16 disjoint batches in one call, which a timer can resolve
    nbatch = min(16, length(pool) ÷ 256)
    for (suffix, bids) in (("", pool[1:256]), ("_x16", pool[1:256*nbatch]))
        bvals = fixture_values(scenario_rng(rep, key * "/batch_values$suffix"), T, length(bids))
        for (how, op, written) in (("absent_distinct_batch", () -> write_values!(x, bids, bvals), bvals),
                                   ("activate_distinct_batch", () -> activate_all!(x, bids), zeros(T, length(bids))))
            function restore_batch!()      # outside the timed region
                @inbounds for k in bids
                    x[k] = zero(T)
                end
                dropzeros!(x)
                any(k -> isactive(x, k), bids) && error("a batch coordinate is still active: not a first activation")
                nothing
            end
            function check_activation()
                copyto!(x, snap)
                restore_batch!()
                length(unique(bids)) == length(bids) || error("batch coordinates are not distinct")
                op()
                all(k -> isactive(x, k), bids) || error("activation did not add every coordinate")
                nactive(x) == s + length(bids) || error("unexpected working set after the activations")
                expected = copy(d)
                expected[bids] = written
                Array(x) == expected || error("activated values differ from the dense reference")
                restore_batch!()
                nactive(x) == s && Array(x) == d || error("restore did not recover the iterate")
                copyto!(x, snap)
                nothing
            end
            addcase!(cases, "sparse", "first_activation", op, check_activation; T, shape, layout=how * suffix,
                     phase="first_activation", prepare=restore_batch!, evals=1, zero_alloc=true, work=length(bids))
        end
    end

    # a full sweep and an active sweep with a trivial update: the iterate is unchanged
    unchanged() = Array(x) == d && nactive(x) == s || error("the sweep changed the iterate")
    addcase!(cases, "sparse", "sweep_full", () -> full_sweep!(x),
             guarded(() -> (full_sweep!(x); unchanged())); T, shape, layout="vector", zero_alloc=true, work=n)
    addcase!(cases, "sparse", "sweep_active_update", () -> active_sweep!(x),
             guarded(() -> (active_sweep!(x); unchanged())); T, shape, layout="trivial_update", zero_alloc=true, work=s)

    # FD-7: sweeps through the handles created inside the call
    addcase!(cases, "sparse", "sweep_activecoordinates", () -> sum_active_ids(x),
             guarded(() -> sum_active_ids(x) == sum(ids) || error("sweep visited the wrong coordinates"));
             T, shape, layout="vector", zero_alloc=true, work=s)
    absvals = sum(abs, vals)
    addcase!(cases, "sparse", "sweep_activevalues", () -> sum_active_values(x, T),
             guarded(() -> close_scalar(sum_active_values(x, T), sum(vals), absvals, T));
             T, shape, layout="vector", zero_alloc=true, work=s)

    # FD-5: a read-write sweep and a product through coordinates(x), against the same through x
    addcase!(cases, "sparse", "sweep_via_coordinates", () -> coordinate_sweep!(x),
             guarded(() -> (coordinate_sweep!(x); unchanged())); T, shape, layout="rank1", zero_alloc=true, work=n)
    addcase!(cases, "sparse", "sweep_via_x", () -> index_sweep!(x),
             guarded(() -> (index_sweep!(x); unchanged())); T, shape, layout="rank1", zero_alloc=true, work=n)
    absdot = dot(abs.(v), abs.(d))
    addcase!(cases, "sparse", "dot_via_coordinates", () -> dot(v, coordinates(x)),
             guarded(() -> close_scalar(dot(v, coordinates(x)), dot(v, d), absdot, T));
             T, shape, layout="rank1", zero_alloc=false, work=s)
    addcase!(cases, "sparse", "dot_x", () -> dot(v, x),
             guarded(() -> close_scalar(dot(v, x), dot(v, d), absdot, T));
             T, shape, layout="rank1", zero_alloc=true, work=s)

    # FD-9: products, with 0% and 50% of the stored values exactly zero at random slots
    for (zlabel, xz, yz, valsz, vals_yz) in (("z0", x, y, vals, vals_y), ("z50", x50, y50, vals50, vals_y50))
        dz = zlabel == "z0" ? d : dense_array(T, (n,), ids, valsz)
        dyz = zlabel == "z0" ? dy : dense_array(T, (n,), ids_y, vals_yz)
        addcase!(cases, "sparse", "dot_vx", () -> dot(v, xz),
                 guarded(() -> close_scalar(dot(v, xz), dot(v, dz), dot(abs.(v), abs.(dz)), T));
                 T, shape, layout=zlabel, zero_alloc=true, work=s)
        addcase!(cases, "sparse", "dot_xy", () -> dot(xz, yz),
                 guarded(() -> close_scalar(dot(xz, yz), dot(dz, dyz), dot(abs.(dz), abs.(dyz)), T));
                 T, shape, layout=zlabel, zero_alloc=false, work=s)
    end
    for m in rows        # dense operands, only at (n, s) = (10⁴, 10²)
        A = randn(scenario_rng(rep, "sparse/A/$T/$n/$m"), T, m, n)
        tB = transpose(randn(scenario_rng(rep, "sparse/B/$T/$n/$m"), T, n, m))
        out = zeros(T, m)
        for (name, M) in (("strided", A), ("transposed", tB)), (zlabel, xz, valsz) in (("z0", x, vals), ("z50", x50, vals50))
            function check_mul()
                dz = dense_array(T, (n,), ids, valsz)
                mul!(out, M, xz)
                close_enough(out, M * dz)
            end
            addcase!(cases, "sparse", "mul!", () -> mul!(out, M, xz), guarded(check_mul);
                     T, shape=shape * "_m$m", layout="$(name)_$zlabel", zero_alloc=true, work=m * s)
        end
    end

    # FN-9: scaling by a number; a call repeats rmul! so that it takes a few µs at small s (the scale is a
    # power of two, and the repetitions stop before the values overflow, so the result is exact). For Inf
    # only the active entries are checked: the inactive ones stay 0 on 0.5.x and become NaN on 0.6.
    r = min(exponent(floatmax(T)) - 4, repeats_for(s, 16384))
    rscale = ldexp(one(T), r)
    check_finite() = (rmul_repeat!(x, T(2), r); Array(x) == d .* rscale || error("rmul! differs from the dense reference"))
    check_inf() = (rmul_repeat!(x, T(Inf), r);
                   all(x[k] == sign(d[k]) * T(Inf) for k in ids) || error("rmul!(x, Inf) differs at the active entries"))
    addcase!(cases, "sparse", "rmul!_finite", () -> rmul_repeat!(x, T(2), r), guarded(check_finite);
             T, shape, layout="x$r", prepare=restore_state!, zero_alloc=false, work=r * s)
    addcase!(cases, "sparse", "rmul!_inf", () -> rmul_repeat!(x, T(Inf), r), guarded(check_inf);
             T, shape, layout="x$r", prepare=restore_state!, zero_alloc=false, work=r * s)

    # FP-7: fill!(coordinates(x), 0); only the values are checked (0.6 keeps the working set)
    addcase!(cases, "sparse", "fill!_coordinates_zero", () -> fill!(coordinates(x), 0),
             guarded(() -> (fill!(coordinates(x), 0); iszero(Array(x)) || error("an entry is not zero after the fill")));
             T, shape, layout="vector", prepare=restore_state!, zero_alloc=true, work=s)

    # FD-12: the five-argument prox_coordinate! with a scalar u. One step takes about 1.5 ns on the pinned
    # source, so a call takes 1536 distinct coordinates: the active ones (at most 768), moved by a step
    # that keeps them nonzero, and absent ones that mostly end at zero and stay inactive, as in a solver.
    f, γ = NormL1(T(0.1)), T(0.5)
    nprox = 1536
    na = min(s, 768)
    kb = vcat(ids[1:na], pool[257:256+nprox-na])
    rng_u = scenario_rng(rep, key * "/steps")
    ub = T[(rand(rng_u, Bool) ? 1 : -1) * T(j <= na ? 0.1 + 0.4 * rand(rng_u) : 0.055 * rand(rng_u)) for j in 1:nprox]
    function check_prox()
        prox_batch!(x, f, kb, ub, γ)
        expected = copy(d)
        for j in eachindex(kb)
            expected[kb[j]] = sign(ub[j]) * max(abs(ub[j]) - γ * T(0.1), zero(T))
        end
        close_enough(Array(x), expected)
    end
    addcase!(cases, "sparse", "prox_coordinate!_5arg", () -> prox_batch!(x, f, kb, ub, γ), guarded(check_prox);
             T, shape, layout="NormL1_scalar_u", prepare=restore_state!, zero_alloc=false, work=length(kb))

    # FD-8: the momentum value loops, through the accessor and through the private array
    ybuf = randn(scenario_rng(rep, key * "/momentum_y"), T, s)
    xcur = randn(scenario_rng(rep, key * "/momentum_cur"), T, s)
    xprev = randn(scenario_rng(rep, key * "/momentum_prev"), T, s)
    buf = zeros(T, s)
    for (layout, getvals) in (("accessors", activevalues), ("private_fields", stored_values_by_field(x)))
        values_op, setvalues_op, dot_op = momentum_ops(getvals, x, buf, copy(vals), ybuf, xcur, xprev, s)
        function check_values()
            fill!(buf, zero(T))
            values_op()
            buf == d[collect(activecoordinates(x))] || error("_values! differs from the dense gather")
        end
        function check_setvalues()
            fill!(activevalues(x), zero(T))
            setvalues_op()
            Array(x) == d || error("_setvalues! differs from the dense reference")
        end
        function check_restartdot()
            ac = collect(activecoordinates(x))
            total, magnitude = zero(T), zero(T)
            for j in 1:s
                term = (d[ac[j]] - ybuf[j]) * (xcur[j] - xprev[j])
                total += term
                magnitude += abs(term)
            end
            close_scalar(dot_op(), total, magnitude, T)
        end
        for (name, op, check) in (("_values!", values_op, check_values), ("_setvalues!", setvalues_op, check_setvalues),
                                  ("_restartdot", dot_op, check_restartdot))
            addcase!(cases, "sparse", name, op, guarded(check); T, shape, layout, zero_alloc=true, work=s)
        end
    end

    # FD-14: aliasing and self-reference
    function check_unrelated()
        write_values!(x, ids, -vals)
        copyto!(x, d)
        Array(x) == d && nactive(x) == s || error("copyto! from a dense vector differs from the reference")
    end
    addcase!(cases, "sparse", "copyto!_unrelated", () -> (copyto!(x, d); nothing), guarded(check_unrelated);
             T, shape, layout="dense_same_nonzeros", zero_alloc=false, work=n)
    addcase!(cases, "sparse", "broadcast_self", () -> (x .= abs.(x); nothing),
             guarded(() -> (x .= abs.(x); Array(x) == abs.(d) || error("x .= abs.(x) differs from the dense reference")));
             T, shape, layout="abs", prepare=restore_state!, zero_alloc=false, work=n)
    # the aliasing copy on its own iterate whose active coordinates are 1:s in slot order, so that the
    # result is the identity with or without the aliasing defect of 0.5.x
    va = fixture_values(scenario_rng(rep, key * "/alias_values"), T, s)
    xa = build_iterate(T, (n,), 1:s, va)
    da = dense_array(T, (n,), 1:s, va)
    function check_alias()
        collect(activecoordinates(xa)) == collect(1:s) || error("the aliasing fixture is not in slot order")
        copyto!(xa, activevalues(xa))
        Array(xa) == da && nactive(xa) == s || error("copyto!(x, activevalues(x)) changed the values")
    end
    addcase!(cases, "sparse", "copyto!_activevalues", () -> (copyto!(xa, activevalues(xa)); nothing), check_alias;
             T, shape, layout="slot_order", zero_alloc=false, work=s)

    # FD-2: clearing and assigning, against the state copy; values only
    addcase!(cases, "sparse", "fill!_zero", () -> (fill!(x, 0); nothing),
             guarded(() -> (fill!(x, 0); iszero(Array(x)) || error("fill! left a nonzero entry")));
             T, shape, layout="vector", prepare=restore_state!, zero_alloc=false, work=s)
    addcase!(cases, "sparse", "fill!_zero_dropzeros", () -> (dropzeros!(fill!(x, 0)); nothing),
             guarded(() -> (dropzeros!(fill!(x, 0)); iszero(Array(x)) || error("fill! and dropzeros! left a nonzero entry")));
             T, shape, layout="vector", prepare=restore_state!, zero_alloc=false, work=s)
    addcase!(cases, "sparse", "broadcast_assign", () -> (x .= y; nothing),
             guarded(() -> (x .= y; Array(x) == dy || error("x .= y differs from y")));
             T, shape, layout="vector", prepare=restore_state!, zero_alloc=false, work=n)
    addcase!(cases, "sparse", "copyto!_state", () -> (copyto!(x, y); nothing),
             guarded(() -> (copyto!(x, y); Array(x) == dy || error("copyto!(x, y) differs from y")));
             T, shape, layout="vector", prepare=restore_state!, zero_alloc=false, work=2 * s)

    # copy and import
    addcase!(cases, "sparse", "copy", () -> (copy(x); nothing),
             guarded(() -> (c = copy(x); c !== x && Array(c) == d || error("copy differs from the iterate")));
             T, shape, layout="vector", phase="allocating", evals=1, zero_alloc=false, work=n)
    # a state copy onto an iterate that already holds the same entries (the state is unchanged)
    addcase!(cases, "sparse", "copyto!_same_kind", () -> (copyto!(x, snap); nothing),
             guarded(() -> (copyto!(x, snap); Array(x) == d || error("state copy differs from the source")));
             T, shape, layout="same_entries", zero_alloc=false, work=2 * s)
    addcase!(cases, "sparse", "copyto!_dense", () -> (copyto!(x, dy); nothing),
             guarded(() -> (copyto!(x, dy); Array(x) == dy || error("import differs from the dense array")));
             T, shape, layout="dense_other_nonzeros", prepare=restore_state!, zero_alloc=false, work=n)

    # FW-4: conversions on dedicated iterates; the sources are Float64
    T == Float64 && sparse_convert_cases!(cases, rep, (n,), ids, key, shape)

    # FW-6: the cheap replace against the staged one
    addcase!(cases, "sparse", "replace_cheap", () -> replace_cheap!(x, dy),
             guarded(() -> (replace_cheap!(x, dy); Array(x) == dy || error("replace differs from the dense array")));
             T, shape, layout="vector", prepare=restore_state!, zero_alloc=false, work=n)
    addcase!(cases, "sparse", "replace_staged", () -> replace_staged!(x, dy),
             guarded(() -> (replace_staged!(x, dy); Array(x) == dy || error("staged replace differs from the dense array")));
             T, shape, layout="vector", prepare=restore_state!, zero_alloc=false, work=n)

    # compaction: dropzeros! with 10% and 50% of the active coordinates holding an explicit zero
    for (zlabel, fraction) in (("zeros10", 0.1), ("zeros50", 0.5))
        zslots = zero_slots(scenario_rng(rep, key * "/compaction_$zlabel"), s, fraction)
        expected = with_zeros(vals, zslots)
        dz = dense_array(T, (n,), ids, expected)
        function prepare_zeros!()          # the snapshot with explicit zeros at those slots
            copyto!(x, snap)
            av = activevalues(x)
            @inbounds for p in zslots
                av[p] = zero(T)
            end
            nothing
        end
        function check_compaction()
            prepare_zeros!()
            nactive(x) == s || error("the zeros were not stored")
            dropzeros!(x)
            nactive(x) == s - length(zslots) || error("dropzeros! left the wrong number of coordinates")
            all(!iszero, activevalues(x)) || error("dropzeros! left a stored zero")
            Array(x) == dz || error("dropzeros! changed the values")
        end
        addcase!(cases, "sparse", "dropzeros!", () -> dropzeros!(x), guarded(check_compaction);
                 T, shape, layout=zlabel, prepare=prepare_zeros!, zero_alloc=true, work=s)
    end
    nothing
end

# rank 2: the read-write sweeps of FD-5 and the products of FD-9
function sparse_matrix_cases!(cases, rep, T, dims, density, label, rows, acache)
    m, ncol = dims
    L = m * ncol
    k = round(Int, density * L)               # exactly round(d·L) distinct entries
    key = "sparse/matrix/$T/$(m)x$(ncol)/$label"
    shape = "$(m)x$(ncol)_d$label"
    ids = randperm(scenario_rng(rep, key * "/ids"), L)[1:k]
    vals = fixture_values(scenario_rng(rep, key * "/values"), T, k)
    vals50 = with_zeros(vals, zero_slots(scenario_rng(rep, key * "/zeros"), k, 0.5))
    X = build_iterate(T, dims, ids, vals)
    snap = copy(X)
    D = dense_array(T, dims, ids, vals)
    X50 = build_iterate(T, dims, ids, vals50)
    restore_state!() = (copyto!(X, snap); nothing)
    guarded(check) = restored(restore_state!, check)
    unchanged() = Array(X) == D && nactive(X) == k || error("the sweep changed the iterate")
    addcase!(cases, "sparse", "sweep_via_coordinates", () -> coordinate_sweep!(X),
             guarded(() -> (coordinate_sweep!(X); unchanged())); T, shape, layout="rank2", zero_alloc=true, work=L)
    addcase!(cases, "sparse", "sweep_via_x", () -> index_sweep!(X),
             guarded(() -> (index_sweep!(X); unchanged())); T, shape, layout="rank2", zero_alloc=true, work=L)
    # the matrix mul! walks the stored entries when 32 nnz <= length and the columns otherwise
    function check_walk(Z)
        (32 * nnz(Z) <= length(Z)) == (density < 0.1) || error("the density $density changed the walk of mul!")
    end
    for mrows in rows
        A = get!(() -> randn(scenario_rng(rep, "sparse/A/$T/$(m)x$(ncol)/$mrows"), T, mrows, m), acache,
                 (T, dims, mrows))::Matrix{T}
        out = zeros(T, mrows, ncol)
        for (zlabel, Z, valsz) in (("z0", X, vals), ("z50", X50, vals50))
            function check_mul()
                check_walk(Z)
                mul!(out, A, Z)
                close_enough(out, A * dense_array(T, dims, ids, valsz))
            end
            addcase!(cases, "sparse", "mul!_matrix", () -> mul!(out, A, Z), guarded(check_mul);
                     T, shape, layout="rows$(mrows)_$zlabel", zero_alloc=true, work=mrows * k)
        end
    end
    a = get!(() -> randn(scenario_rng(rep, "sparse/a/$T/$(m)x$(ncol)"), T, m), acache, (T, dims, :a))::Vector{T}
    b = get!(() -> randn(scenario_rng(rep, "sparse/b/$T/$(m)x$(ncol)"), T, ncol), acache, (T, dims, :b))::Vector{T}
    for (zlabel, Z, valsz) in (("z0", X, vals), ("z50", X50, vals50))
        function check_dot()
            Dz = dense_array(T, dims, ids, valsz)
            close_scalar(dot(a, Z, b), dot(a, Dz * b), dot(abs.(a), abs.(Dz) * abs.(b)), T)
        end
        addcase!(cases, "sparse", "dot_aXb", () -> dot(a, Z, b), guarded(check_dot);
                 T, shape, layout=zlabel, zero_alloc=false, work=k)
    end
    nothing
end

# mul! with a Symmetric operand and a SparseIterate vector of the same length (s = round(d·p))
function sparse_symmetric_cases!(cases, rep, T, p, density, label, scache)
    s = round(Int, density * p)
    key = "sparse/symmetric/$T/$p/$label"
    shape = "p$(p)_s$(s)"
    S = get!(() -> Symmetric(randn(scenario_rng(rep, "sparse/S/$T/$p"), T, p, p)), scache, (T, p))::Symmetric{T,Matrix{T}}
    ids = randperm(scenario_rng(rep, key * "/ids"), p)[1:s]
    vals = fixture_values(scenario_rng(rep, key * "/values"), T, s)
    vals50 = with_zeros(vals, zero_slots(scenario_rng(rep, key * "/zeros"), s, 0.5))
    out = zeros(T, p)
    for (zlabel, valsz) in (("z0", vals), ("z50", vals50))
        x = build_iterate(T, (p,), ids, valsz)
        function check_mul()
            mul!(out, S, x)
            close_enough(out, S * dense_array(T, (p,), ids, valsz))
        end
        addcase!(cases, "sparse", "mul!", () -> mul!(out, S, x), check_mul;
                 T, shape, layout="symmetric_$zlabel", zero_alloc=true, work=p * s)
    end
    nothing
end

function sparse_cases!(cases, rep; quick=false)
    types = quick ? (Float64,) : (Float32, Float64)
    grid = quick ? ((10^4, 10^2), (10^6, 10^3)) : ((10^4, 10^2), (10^6, 10^2), (10^6, 10^3), (10^6, 10^4))
    rows = quick ? (256,) : (256, 1024)       # rows of the dense operands, and p of the Symmetric ones
    densities = quick ? SPARSE_DENSITIES[2:2] : SPARSE_DENSITIES
    vcache, acache, scache = Dict{Any,Any}(), Dict{Any,Any}(), Dict{Any,Any}()
    for T in types, n in unique(first.(grid))
        addcase!(cases, "sparse", "construct", () -> (SparseIterate(T, n); nothing),
                 function ()
                     x = SparseIterate(T, n)
                     # retained memory: one T array and two Int arrays of length n, plus headers
                     abs(Base.summarysize(x) - (sizeof(T) + 2 * sizeof(Int)) * n) <= 1024 || error("unexpected retained size")
                     nactive(x) == 0 || error("a new iterate has active coordinates")
                 end;
                 T, shape="n$n", layout="empty", phase="construct", evals=1, zero_alloc=false, work=n)
    end
    for T in types, (n, s) in grid
        sparse_vector_cases!(cases, rep, T, n, s, vcache; rows=(n, s) == (10^4, 10^2) ? rows : ())
    end
    for T in types, dims in ((1000, 64), (256, 1000)), (density, label) in densities
        sparse_matrix_cases!(cases, rep, T, dims, density, label, rows, acache)
    end
    sym_densities = quick ? SPARSE_DENSITIES[2:2] : SPARSE_DENSITIES[2:3]
    for T in types, p in rows, (density, label) in sym_densities
        sparse_symmetric_cases!(cases, rep, T, p, density, label, scache)
    end
    # FW-4 on the dense branch of the copy kernel: matrices at 20% (dedicated, Float64 sources)
    for dims in (quick ? ((1000, 64),) : ((1000, 64), (256, 1000)))
        L = prod(dims)
        key = "sparse/convert/$(dims[1])x$(dims[2])/20"
        ids = randperm(scenario_rng(rep, key * "/ids"), L)[1:round(Int, 0.2 * L)]
        sparse_convert_cases!(cases, rep, dims, ids, key, "$(dims[1])x$(dims[2])_d20")
    end
    nothing
end

const CASE_GROUPS = ("vectors", "smooth", "matrices", "constraints", "tensors", "products", "blocks", "owner", "sparse")

function build_cases(rep; suite="quick", groups=collect(CASE_GROUPS))
    suite in ("quick", "extended") || error("PB_SUITE must be quick or extended")
    isempty(groups) && error("Select at least one benchmark group")
    all(in(CASE_GROUPS), groups) || error("PB_GROUPS must select from $(join(CASE_GROUPS, ','))")
    length(unique(groups)) == length(groups) || error("Duplicate benchmark group")
    cases=PerfCase[]
    builders=(separable_cases!,smooth_cases!,matrix_cases!,constrained_cases!,tensor_cases!,product_cases!,block_cases!,owner_cases!,sparse_cases!)
    # Select groups and sizes before allocating data or constructing function objects.
    for (group,builder) in zip(CASE_GROUPS,builders)
        group in groups && builder(cases,rep;quick=suite=="quick")
    end
    suite == "extended" && "vectors" in groups && group_layout_cases!(cases, rep)
    ids=[c.meta.id for c in cases]
    length(unique(ids))==length(ids) || error("duplicate case IDs")
    cases
end
