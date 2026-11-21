using Random

const DIMENSIONS = (2, 3, 4, 8, 16, 64, 256, 2048)

function cases(emit)
    rng = Xoshiro(9031119)
    for T in (Float32, Float64), n in DIMENSIONS, j in 1:16
        magnitudes = if j % 4 == 0
            ones(T, n)
        elseif j % 4 == 1
            rand(rng, T, n) .+ T(0.5)
        elseif j % 4 == 2
            one(T) .+ eps(T) .* T.(rand(rng, 0:31, n))
        else
            rand(rng, T[0, 0.25, 0.5, 1], n)
        end
        x = magnitudes .* rand(rng, T[-1, 1], n)
        total = sum(magnitudes)
        iszero(total) && continue
        for ratio in T[0.001, 0.01, 0.1, 0.4, 0.9, 0.999]
            emit("historical", "j$(j)-$(ratio)", x, ratio * total)
        end
    end

    rng = Xoshiro(827501)
    for T in (Float32, Float64)
        emit("fixed", "reported", T[1, 1], T(0.1))
        emit("fixed", "reported-matrix", reshape(T[1, 1], 1, 2), T(0.1))
        emit("fixed", "large-tie", ones(T, 2048), T(2.048))
        emit("contract", "rounded-copy", T[1, eps(T) / 4, -0.0], one(T))
        emit("contract", "strict-copy", T[1, -0.5, -0.0], T(1.5))
        emit("contract", "singleton", T[-2], T(0.35))
        emit("contract", "zeros", T[-0.0, +0.0], one(T))
        for n in DIMENSIONS, j in 1:6
            mags = if j == 1
                ones(T, n)
            elseif j == 2
                [isodd(i) ? prevfloat(one(T)) : nextfloat(one(T)) for i in 1:n]
            elseif j == 3
                rand(rng, T, n) .+ T(0.5)
            elseif j == 4
                [i in 1:2 ? T(4-i) : isodd(i) ? T(0.125) : zero(T) for i in 1:n]
            elseif j == 5
                rand(rng, T[0.125, 0.25, 0.5, 1], n)
            else
                rand(rng, T, n) .* T.(exp2.(rand(rng, -6:6, n)))
            end
            signs = j <= 2 ? fill(j == 1 ? one(T) : -one(T), n) : rand(rng, T[-1, 1], n)
            x = mags .* signs
            total = sum(abs, x)
            breakpoint = maximum(mags) - minimum(mags)
            radii = (T(0.0001) * total, T(0.02) * total, T(0.45) * total, T(0.99) * total,
                     prevfloat(total), total, nextfloat(total), T(1.25) * total,
                     prevfloat(breakpoint), breakpoint, nextfloat(breakpoint))
            for (k, radius) in pairs(radii)
                radius >= T(0.0001) * total || continue
                emit("adversarial", "j$(j)-r$(k)", x, radius)
                permuted = x[randperm(rng, n)]
                rows = iseven(n) ? 2 : n
                emit("adversarial", "j$(j)-r$(k)-permuted-matrix",
                     reshape(permuted, rows, :), radius)
            end
        end
    end
end
