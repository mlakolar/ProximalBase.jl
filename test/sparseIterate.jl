facts("sparseIterate") do

  x = SparseIterate(3)
  x[3] = 1.
  x[1] = 1.
  @fact nnz(x) --> 2
  x[1] = 0.
  @fact nnz(x) --> 2
  dropzeros!(x)
  @fact nnz(x) --> 1

  x = SparseIterate(10)
  xv = randn(5)
  for i=1:5
    x[i] = xv[i]
  end
  y = randn(10)
  @fact dot(xv, y[1:5]) --> dot(x, y)
end
