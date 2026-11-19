function score_cases(emit)
    for T in (Float32, Float64)
        run(name, path, v, x, level, factors=()) =
            emit(name, path, v, x, level, factors)
        u = eps(T)
        z = nextfloat(zero(T))
        for (name, v, x, c) in (
            ("empty", T[], T[], one(T)),
            ("zero_inside", T[0.125,-0.5], T[-0.0,0], one(T)),
            ("zero_boundary", T[-0.25,0.75], T[0,0], one(T)),
            ("zero_outside", T[-2,1,0.25], zeros(T,3), T(0.75)),
            ("zero_ulp_outside", T[-0.25,0.75], T[0,0], prevfloat(one(T))),
            ("exact_signed_face", T[-0.25,0.75,0], T[1,-1,0.5], one(T)),
            ("near_signed_face", T[-0.25,0.75,0], T[1,-prevfloat(one(T)),0.5], one(T)),
            ("many_leave", T[-8,1,0,-0.25,0.125,-0.75], T[1,-1,1,1,-1,1], T(0.75)),
            ("negative_level_on_face", T[1,3,-2,0.125], T[1,-1,1,0], T(20)),
            ("signed_zero_level", T[3,4], T[0,-1], T(-0.0)),
            ("infinite_level_zero", T[3,4], T[0,0], T(Inf)),
            ("infinite_level_face", T[3,4], T[1,0], T(Inf)),
            ("subnormal_zero", T[8z,-4z,z], zeros(T,3), T(3z)),
            ("subnormal_face", T[-8z,-4z,z], T[1,-1,0], T(3z)),
            ("aggregate_subnormal_100", zeros(T,4096), ones(T,4096), T(100z)),
            ("aggregate_subnormal_2047", zeros(T,4096), ones(T,4096), T(2047z)),
            ("large_uniform_face", fill(T(-0.1),100_001), ones(T,100_001), T(50)),
            ("huge_zero", fill(floatmax(T)/2, 4), zeros(T,4), floatmax(T)/2),
            ("huge_cancellation", T[-floatmax(T),4z], T[1,0], floatmax(T)),
            ("range_overflow", fill(floatmax(T), 3), zeros(T,3), zero(T)),
            ("count_rise_two_zero", T[3,1+3u], zeros(T,2), T(2)-2u),
            ("count_rise_two_face", -T[3,1+3u], ones(T,2), T(2)-2u),
        )
            run(name, all(iszero,x) ? "zero" : "face", v, x, c)
        end
        for width in (2, 8, 32), k in unique([1, min(3,width), width]), j in 1:4
            v = T[div(mod(83*i+127*j,97), 2)/16-1.5 for i in 1:width]
            x = T[i <= k ? (-1)^i : mod(i,4)/5 for i in 1:width]
            run("dyadic_$(width)_$(k)_$j", "face", v, x, T((2j-1)/8))
        end
        α = T(2)
        for x in (T[1,-1,0],zeros(T,3))
            run("scaled", all(iszero,x) ? "zero_scaled" : "face_scaled", T[-1,0.25,3], x, T(0.125), (α,))
        end
        run("nested_merged", "face_nested", T[-1,0.25,3], T[1,-1,0], T(0.125), (α,α))
        run("scaled_nonbinary", "face_scaled", T[-0.2,0.7,-3], T[1,-1,1], T(0.125), (T(0.7),))
        tiny = ldexp(one(T), T === Float32 ? -100 : -600)
        large = ldexp(one(T), T === Float32 ? 100 : 1000)
        folded = ldexp(one(T), T === Float32 ? -100 : -200)
        run("nested_nonmerged", "face_nested", T[-folded,folded,0], T[1,-1,0], large, (tiny,tiny))
        run("scaled_range", "scaled_documented_range", T[floatmax(T)/2, floatmax(T)/2], T[0,0], T(2), (T(0.125),))
    end
    emit("integer_face_rounds_to_tie", "face_working_type", Float32[-0.25,0.75],
         Int32[16_777_217,-16_777_216], 1f0, ())
    emit("scaled_empty_support", "face", fill(-0.1,3), ones(3), 2.0^-60, ())
    emit("rounded_zero_count", "zero", ones(2001), zeros(2001), 8e-14, ())
    # First/last and zero/full face. Power-of-two scaling retains the exact
    # effective radius. The million-tie rho is a documented reconstruction.
    families = (
        ("count_rise_02", 0.2f0, 0.1f0, 0.3f0, 100_000),
        ("count_rise_0102", 0.2f0, 0.1f0, 0.102f0, 10_000),
        ("count_rise_f64", 1.8, 0.9, 0.9000008, 1_000_000),
        ("round3", 1f0, 0.009486833f0, 0.1f0, 100_000),
        ("round3_reconstruction_rho006", 1f0, Float32(0.9/(1+0.06*1e12*2.0^-24)), 0.1f0, 1_000_000),
        ("issue1003_a", 1f0, 0.17933106f0, 0.5f0, 100_000),
        ("issue1003_c", 1., 0.8999997002398832, 0.1, 1_000_000),
        ("issue1003_d", 1f0, 0.0027806521f0, 0.5f0, 1_000_000),
    )
    for (name, head, tie, c, m) in families, firsthead in (true,false), face in (false,true)
        T = typeof(head)
        v = fill(face ? -tie : tie, m+1)
        v[firsthead ? 1 : end] = face ? -head : head
        x = fill(face ? one(T) : zero(T), m+1)
        path = face ? "face" : "zero"
        id = name * (firsthead ? "_first" : "_last")
        emit(id, path, v, x, c, ())
        if name == "round3" || name == "round3_reconstruction_rho006"
            emit(id, path*"_scaled", v, x, c/2, (T(2),))
        end
    end
    # Issue 1003's second input has three magnitude groups.
    v = [fill(0.3f0,100_000);fill(1.2f0,10);zeros(Float32,800_096)]
    emit("issue1003_b", "zero", v, zero(v), 2.7f0, ())
    emit("issue1003_b", "face", -v, ones(Float32,length(v)), 2.7f0, ())
end
