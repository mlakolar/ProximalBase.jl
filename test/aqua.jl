module AquaTest

using Test
using ProximalBase
using Aqua

@testset "Aqua" begin
  Aqua.test_all(ProximalBase)
end

end
