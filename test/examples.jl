module ExamplesTest

using Test

# Every script in examples/ runs to its end, which includes its own checks of the
# result, each in a fresh module and with its printed output discarded.
const DIR = joinpath(dirname(@__DIR__), "examples")

@testset "examples" begin
  for file in sort(filter(endswith(".jl"), readdir(DIR)))
    @testset "$file" begin
      m = Module(Symbol(splitext(file)[1]))
      ok = redirect_stdout(devnull) do
        Base.include(m, joinpath(DIR, file))
        true
      end
      @test ok
    end
  end
end

end
