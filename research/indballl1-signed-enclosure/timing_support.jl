# Each measured batch uses a concrete callable and a Float64 result sink.
function timed(run::F, n) where {F}
    before = time_ns()
    sink = run(n)
    elapsed = time_ns() - before
    @assert isfinite(sink)
    Float64(elapsed) / n
end

function timemin(run::F, n) where {F}
    best = Inf
    for _ in 1:3
        best = min(best, timed(run, n))
    end
    best
end

function allocation(run::F, n) where {F}
    run(n)
    bytes = @allocated run(n)
    allocations = @allocations run(n)
    bytes/n, allocations/n
end

function ns_estimate(run::F) where {F}
    run(8)
    timemin(run, 20)
end

function paired(io, meta, baseline::A, candidate::B, control::C, rounds; batchlimit=200000) where {A,B,C}
    estimate = max(ns_estimate(baseline), ns_estimate(candidate), ns_estimate(control), 1.0)
    batch = clamp(round(Int, batchlimit / estimate), 4, 100000)
    acount = min(128, batch)
    a_bytes, a_allocs = allocation(baseline, acount)
    b_bytes, b_allocs = allocation(candidate, acount)
    aa_bytes, aa_allocs = allocation(control, acount)
    for k in 1:rounds
        if isodd(k)
            ab_left, ab_right = timemin(baseline, batch), timemin(candidate, batch)
            aa_right, aa_left = timemin(control, batch), timemin(baseline, batch)
        else
            ab_right, ab_left = timemin(candidate, batch), timemin(baseline, batch)
            aa_left, aa_right = timemin(baseline, batch), timemin(control, batch)
        end
        row = (meta..., k, isodd(k) ? "AB" : "BA", isodd(k) ? "BA" : "AB", batch,
               ab_left, ab_right, aa_left, aa_right, a_bytes, b_bytes, aa_bytes,
               a_allocs, b_allocs, aa_allocs)
        println(io, join(row, '\t'))
    end
end

const TIMING_COLUMNS = "rep\torder\taa_order\tbatch\ta_ns\tb_ns\taa0_ns\taa1_ns\ta_bytes\tb_bytes\taa_bytes\ta_allocs\tb_allocs\taa_allocs"
