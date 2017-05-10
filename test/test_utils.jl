facts("utils") do

  @fact shrink(3.0, 1.0) --> 2.0
  @fact shrink(-2.0, 1.0) --> -1.0
  @fact shrink(0.5, 1.0) --> 0.
  
end
