# For these gradients each indicated residual is known analytically.
function norm_case(::Type{T}, n, path; exponent = 0) where {T}
    tail = ldexp(sqrt(T(0.69) * (eps(T) / 2)), exponent)
    head = ldexp(one(T), exponent)
    v = fill(tail, n)
    x = zeros(T, n)
    v[1] = head
    heads, tails = 1, n - 1
    c = zero(T)
    if path == "zero_outside"
        v[1] = T(1.25) * head
        c = head / 4
    elseif path == "singleton_face"
        v[end], x[end] = head, -one(T)
        c = head
        tails -= 1
    elseif path == "signed_face"
        v[1], v[end] = T(-1.25) * head, T(1.25) * head
        x[1], x[end] = one(T), -one(T)
        c = head / 2
        heads, tails = 2, n - 2
    else
        @assert path == "zero_penalty"
    end
    expected = setprecision(BigFloat, 192) do
        sqrt(heads * BigFloat(head)^2 + tails * BigFloat(tail)^2)
    end
    v, x, c, expected
end
