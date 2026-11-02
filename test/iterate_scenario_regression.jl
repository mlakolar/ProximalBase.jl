module IterateScenarioRegressionTest
# Scenario I6 (PR #430): a new regression test file.
using Test
using ProximalBase
@testset "scenario regression" begin
  x = AtomIterate(Float64, 4, 2)
  x[3] = 1
  @test nactive(x) == 1 && isactive(x, 2)
end
end
