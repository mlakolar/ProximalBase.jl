using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
  @compile_workload begin
    for T in (Float32, Float64)
      x = T[1, -2, 0, 3]
      y = similar(x)
      X = T[1 0 2 0; 0 1 0 2; 1 1 0 0; 0 0 1 1; 2 0 0 1]
      b = T[1, 2, -1, 0, 3]
      for f in (ZeroFunction(), LinearFunction(copy(x)), SqrNormL2(T(0.2)),
                NormL1(T(0.1)), Scaled(LeastSquaresLoss(b, X), T(0.2)),
                QuadraticFunction(Matrix{T}(I, 4, 4)))
        f(x)
        prox!(y, f, x, one(T))
        value_and_prox!(y, f, x, one(T))
        if issmooth(f)
          gradient!(y, f, x)
          value_and_gradient!(y, f, x)
        end
      end
      prox_block(NormL1(T(0.1)), x[1], 1, one(T))
      s = SparseIterate(x)
      a = BlockIterate(x, [Int[3, 1], Int[4, 2]])
      S = SymmetricSparseIterate(T[1 2; 2 0])
      copy(s); copy(a); copy(S)
      matrixview(S); blockview(a, 1)
      X * s; X * a
    end
  end
end
