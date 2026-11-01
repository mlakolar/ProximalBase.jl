module ConformanceSpecialValuesTest

using Test
using ProximalBase
using ..Conformance

# E2-E6/O2: NaN scope, domain limits, and disabled terms.

@row begin
  id = "S1.value"; case = "S1"; family = :special_values; oracle = :value
  types = "Float64"; backend = :cpu; rules = "E6 O2"
  problem = "Outside box has infinite value; projection is feasible."
  expected = Inf
  observed = IndBox(0.0, 1.0)([2.0])
end

@row begin
  id = "S1.prox"; case = "S1"; family = :special_values; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "E6 O2"
  problem = "Outside box has infinite value; projection is feasible."
  expected = [1.0]
  observed = prox(IndBox(0.0, 1.0), [2.0], 1.0)
end

@row begin
  id = "S1.joint-value"; case = "S1"; family = :special_values; oracle = :value
  types = "Float64"; backend = :cpu; rules = "E6 O2"
  problem = "Outside box has infinite value; projection is feasible."
  expected = 0.0
  observed = value_and_prox!(zeros(1), IndBox(0.0, 1.0), [2.0], 1.0)
end

@row begin
  id = "S2.value"; case = "S2"; family = :special_values; oracle = :value
  types = "Float64"; backend = :cpu; rules = "E4"
  problem = "Zero ridge is the zero function before NaN handling."
  expected = 0.0
  observed = SqrNormL2(0.0)([NaN])
end

@row begin
  id = "S2.gradient"; case = "S2"; family = :special_values; oracle = :gradient
  types = "Float64"; backend = :cpu; rules = "E4"
  problem = "Zero ridge is the zero function before NaN handling."
  expected = [0.0]
  observed = gradient(SqrNormL2(0.0), [NaN])
  broken = (group = :G2, baseline = Returned(v -> same(v, [NaN])))
end

@row begin
  id = "S2.prox"; case = "S2"; family = :special_values; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "E4"
  problem = "Zero ridge is the zero function before NaN handling."
  expected = [NaN]
  observed = prox(SqrNormL2(0.0), [NaN], 1.0)
end

@row begin
  id = "S3.NormL1"; case = "S3"; family = :special_values; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "E2"
  problem = "NaN affects its coordinate or the whole coupled block."
  expected = [NaN, 3.0]
  observed = prox(NormL1(1.0), [NaN, 4.0], 1.0)
end

@row begin
  id = "S3.NormL2"; case = "S3"; family = :special_values; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "E2"
  problem = "NaN affects its coordinate or the whole coupled block."
  expected = [NaN, NaN]
  observed = prox(NormL2(1.0), [NaN, 4.0], 1.0)
  broken = (group = :G2, baseline = Returned(v -> same(v, [0.0, 0.0])))
end

@row begin
  id = "S3.GroupNormL2"; case = "S3"; family = :special_values; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "E2"
  problem = "NaN affects its coordinate or the whole coupled block."
  expected = [NaN, 3.0]
  observed = prox(GroupNormL2(1.0, [1:1, 2:2], [1.0, 1.0]), [NaN, 4.0], 1.0)
  broken = (group = :G2, baseline = Returned(v -> same(v, [0.0, 3.0])))
end

@row begin
  id = "S4.value"; case = "S4"; family = :special_values; oracle = :value
  types = "Float64"; backend = :cpu; rules = "E4 O2"
  problem = "Zero weight is free; infinite weight fixes its entry at zero."
  expected = Inf
  observed = NormL1(1.0, [0.0, Inf])([3.0, 4.0])
end

@row begin
  id = "S4.prox"; case = "S4"; family = :special_values; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "E4 O2"
  problem = "Zero weight is free; infinite weight fixes its entry at zero."
  expected = [3.0, 0.0]
  observed = prox(NormL1(1.0, [0.0, Inf]), [3.0, 4.0], 1.0)
end

@row begin
  id = "S4.joint-value"; case = "S4"; family = :special_values; oracle = :value
  types = "Float64"; backend = :cpu; rules = "E4 O2"
  problem = "Zero weight is free; infinite weight fixes its entry at zero."
  expected = 0.0
  observed = value_and_prox!(zeros(2), NormL1(1.0, [0.0, Inf]), [3.0, 4.0], 1.0)
end

@row begin
  id = "S5.boundary"; case = "S5"; family = :special_values; oracle = :value
  types = "Float64"; backend = :cpu; rules = "E2 E6"
  problem = "Entropy boundary limit, domain and NaN value."
  expected = 0.0
  observed = NegEntropy(1.0)([0.0])
end

@row begin
  id = "S5.outside"; case = "S5"; family = :special_values; oracle = :value
  types = "Float64"; backend = :cpu; rules = "E2 E6"
  problem = "Entropy boundary limit, domain and NaN value."
  expected = Inf
  observed = NegEntropy(1.0)([-1.0])
end

@row begin
  id = "S5.nan"; case = "S5"; family = :special_values; oracle = :value
  types = "Float64"; backend = :cpu; rules = "E2 E6"
  problem = "Entropy boundary limit, domain and NaN value."
  expected = NaN
  observed = NegEntropy(1.0)([NaN])
  broken = (group = :G2, baseline = Returned(v -> same(v, Inf)))
end

@row begin
  id = "S6.value"; case = "S6"; family = :special_values; oracle = :value
  types = "Float64"; backend = :cpu; rules = "E4"
  problem = "Zero nuclear level has no domain constraint and copies NaN input."
  expected = 0.0
  observed = NuclearNorm(0.0)([NaN 0.0; 0.0 1.0])
  broken = (group = :G2, baseline = Returned(v -> same(v, Inf)))
end

@row begin
  id = "S6.prox"; case = "S6"; family = :special_values; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "E4"
  problem = "Zero nuclear level has no domain constraint and copies NaN input."
  expected = [NaN 0.0; 0.0 1.0]
  observed = prox(NuclearNorm(0.0), [NaN 0.0; 0.0 1.0], 1.0)
end

@row begin
  id = "S9.box.prox"; case = "S9"; family = :special_values; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "E2 O2"
  problem = "NaN projection cannot use indicator zero-value shortcut."
  expected = [NaN, 0.0]
  observed = let y = zeros(2); value_and_prox!(y, IndBox(-1.0, 1.0), [NaN, 0.0], 1.0); y end
end

@row begin
  id = "S9.box.value"; case = "S9"; family = :special_values; oracle = :value
  types = "Float64"; backend = :cpu; rules = "E2 O2"
  problem = "NaN projection cannot use indicator zero-value shortcut."
  expected = NaN
  observed = value_and_prox!(zeros(2), IndBox(-1.0, 1.0), [NaN, 0.0], 1.0)
  broken = (group = :G2, baseline = Returned(v -> same(v, 0.0)))
end

@row begin
  id = "S9.ball.prox"; case = "S9"; family = :special_values; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "E2 O2"
  problem = "NaN projection cannot use indicator zero-value shortcut."
  expected = [NaN, NaN]
  observed = let y = zeros(2); value_and_prox!(y, IndBallL2(1.0), [NaN, 0.0], 1.0); y end
end

@row begin
  id = "S9.ball.value"; case = "S9"; family = :special_values; oracle = :value
  types = "Float64"; backend = :cpu; rules = "E2 O2"
  problem = "NaN projection cannot use indicator zero-value shortcut."
  expected = NaN
  observed = value_and_prox!(zeros(2), IndBallL2(1.0), [NaN, 0.0], 1.0)
  broken = (group = :G2, baseline = Returned(v -> same(v, 0.0)))
end

@row begin
  id = "L1.value"; case = "L1"; family = :special_values; oracle = :value
  types = "Float64"; backend = :cpu; rules = "E6 R1 R2"
  problem = "Logistic positive-label tail; E6 smooth budget 4 ULP per scalar."
  expected = bigref(() -> log1p(exp(-BigFloat(40))))
  observed = LogisticLoss([1.0], ones(1, 1))([40.0])
  compare = (a, b) -> within_ulps(a, b, 4)
end

@row begin
  id = "L1.gradient"; case = "L1"; family = :special_values; oracle = :gradient
  types = "Float64"; backend = :cpu; rules = "E6 R1 R2"
  problem = "Logistic positive-label tail; E6 smooth budget 4 ULP per scalar."
  expected = [bigref(() -> -inv(1 + exp(BigFloat(40))))]
  observed = gradient(LogisticLoss([1.0], ones(1, 1)), [40.0])
  compare = (a, b) -> length(a) == 1 && within_ulps(a[1], b[1], 4)
end

@row begin
  id = "special_values.NormL1.Float32.nan.prox"; case = ""; family = :special_values; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "E2"
  problem = "NaN propagates only to dependent coordinates; coupled blocks are entirely NaN."
  expected = Float32[NaN, 3]
  observed = prox(NormL1(one(Float32)), Float32[NaN, 4], one(Float32))
end

@row begin
  id = "special_values.NormL1.Float32.nan.value"; case = ""; family = :special_values; oracle = :value
  types = "Float32"; backend = :cpu; rules = "E2"
  problem = "NaN propagates only to dependent coordinates; coupled blocks are entirely NaN."
  expected = Float32(NaN)
  observed = NormL1(one(Float32))(Float32[NaN, 4])
end

@row begin
  id = "special_values.NormL1.Float32.nan-beats-inf"; case = ""; family = :special_values; oracle = :value
  types = "Float32"; backend = :cpu; rules = "E2"
  problem = "NaN beats infinity in a sum or membership decision."
  expected = Float32(NaN)
  observed = NormL1(one(Float32))(Float32[Inf, NaN])
end

@row begin
  id = "special_values.NormL1.Float32.infinite-value"; case = ""; family = :special_values; oracle = :value
  types = "Float32"; backend = :cpu; rules = "E3 E5"
  problem = "Norm or indicator value is positive infinity at an infinite entry."
  expected = Float32(Inf)
  observed = NormL1(one(Float32))(Float32[-Inf, 4])
end

@row begin
  id = "special_values.NormL2.Float32.nan.prox"; case = ""; family = :special_values; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "E2"
  problem = "NaN propagates only to dependent coordinates; coupled blocks are entirely NaN."
  expected = Float32[NaN, NaN]
  observed = prox(NormL2(one(Float32)), Float32[NaN, 4], one(Float32))
  broken = (group = :G2, baseline = Returned(v -> same(v, Float32[0.0, 0.0])))
end

@row begin
  id = "special_values.NormL2.Float32.nan.value"; case = ""; family = :special_values; oracle = :value
  types = "Float32"; backend = :cpu; rules = "E2"
  problem = "NaN propagates only to dependent coordinates; coupled blocks are entirely NaN."
  expected = Float32(NaN)
  observed = NormL2(one(Float32))(Float32[NaN, 4])
end

@row begin
  id = "special_values.NormL2.Float32.nan-beats-inf"; case = ""; family = :special_values; oracle = :value
  types = "Float32"; backend = :cpu; rules = "E2"
  problem = "NaN beats infinity in a sum or membership decision."
  expected = Float32(NaN)
  observed = NormL2(one(Float32))(Float32[Inf, NaN])
end

@row begin
  id = "special_values.NormL2.Float32.infinite-value"; case = ""; family = :special_values; oracle = :value
  types = "Float32"; backend = :cpu; rules = "E3 E5"
  problem = "Norm or indicator value is positive infinity at an infinite entry."
  expected = Float32(Inf)
  observed = NormL2(one(Float32))(Float32[-Inf, 4])
end

@row begin
  id = "special_values.GroupNormL2.Float32.nan.prox"; case = ""; family = :special_values; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "E2"
  problem = "NaN propagates only to dependent coordinates; coupled blocks are entirely NaN."
  expected = Float32[NaN, NaN]
  observed = prox(GroupNormL2(one(Float32), [1:2], Float32[1]), Float32[NaN, 4], one(Float32))
  broken = (group = :G2, baseline = Returned(v -> same(v, Float32[0.0, 0.0])))
end

@row begin
  id = "special_values.GroupNormL2.Float32.nan.value"; case = ""; family = :special_values; oracle = :value
  types = "Float32"; backend = :cpu; rules = "E2"
  problem = "NaN propagates only to dependent coordinates; coupled blocks are entirely NaN."
  expected = Float32(NaN)
  observed = GroupNormL2(one(Float32), [1:2], Float32[1])(Float32[NaN, 4])
end

@row begin
  id = "special_values.GroupNormL2.Float32.nan-beats-inf"; case = ""; family = :special_values; oracle = :value
  types = "Float32"; backend = :cpu; rules = "E2"
  problem = "NaN beats infinity in a sum or membership decision."
  expected = Float32(NaN)
  observed = GroupNormL2(one(Float32), [1:2], Float32[1])(Float32[Inf, NaN])
end

@row begin
  id = "special_values.GroupNormL2.Float32.infinite-value"; case = ""; family = :special_values; oracle = :value
  types = "Float32"; backend = :cpu; rules = "E3 E5"
  problem = "Norm or indicator value is positive infinity at an infinite entry."
  expected = Float32(Inf)
  observed = GroupNormL2(one(Float32), [1:2], Float32[1])(Float32[-Inf, 4])
end

@row begin
  id = "special_values.NormLinf.Float32.nan.prox"; case = ""; family = :special_values; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "E2"
  problem = "NaN propagates only to dependent coordinates; coupled blocks are entirely NaN."
  expected = Float32[NaN, NaN]
  observed = prox(NormLinf(one(Float32)), Float32[NaN, 4], one(Float32))
end

@row begin
  id = "special_values.NormLinf.Float32.nan.value"; case = ""; family = :special_values; oracle = :value
  types = "Float32"; backend = :cpu; rules = "E2"
  problem = "NaN propagates only to dependent coordinates; coupled blocks are entirely NaN."
  expected = Float32(NaN)
  observed = NormLinf(one(Float32))(Float32[NaN, 4])
end

@row begin
  id = "special_values.NormLinf.Float32.nan-beats-inf"; case = ""; family = :special_values; oracle = :value
  types = "Float32"; backend = :cpu; rules = "E2"
  problem = "NaN beats infinity in a sum or membership decision."
  expected = Float32(NaN)
  observed = NormLinf(one(Float32))(Float32[Inf, NaN])
end

@row begin
  id = "special_values.NormLinf.Float32.infinite-value"; case = ""; family = :special_values; oracle = :value
  types = "Float32"; backend = :cpu; rules = "E3 E5"
  problem = "Norm or indicator value is positive infinity at an infinite entry."
  expected = Float32(Inf)
  observed = NormLinf(one(Float32))(Float32[-Inf, 4])
end

@row begin
  id = "special_values.IndBox.Float32.nan.prox"; case = ""; family = :special_values; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "E2"
  problem = "NaN propagates only to dependent coordinates; coupled blocks are entirely NaN."
  expected = Float32[NaN, 1]
  observed = prox(IndBox(-one(Float32), one(Float32)), Float32[NaN, 4], one(Float32))
end

@row begin
  id = "special_values.IndBox.Float32.nan.value"; case = ""; family = :special_values; oracle = :value
  types = "Float32"; backend = :cpu; rules = "E2"
  problem = "NaN propagates only to dependent coordinates; coupled blocks are entirely NaN."
  expected = Float32(NaN)
  observed = IndBox(-one(Float32), one(Float32))(Float32[NaN, 4])
  broken = (group = :G2, baseline = Returned(v -> same(v, Inf32)))
end

@row begin
  id = "special_values.IndBox.Float32.nan-beats-inf"; case = ""; family = :special_values; oracle = :value
  types = "Float32"; backend = :cpu; rules = "E2"
  problem = "NaN beats infinity in a sum or membership decision."
  expected = Float32(NaN)
  observed = IndBox(-one(Float32), one(Float32))(Float32[Inf, NaN])
  broken = (group = :G2, baseline = Returned(v -> same(v, Inf32)))
end

@row begin
  id = "special_values.IndBox.Float32.infinite-value"; case = ""; family = :special_values; oracle = :value
  types = "Float32"; backend = :cpu; rules = "E3 E5"
  problem = "Norm or indicator value is positive infinity at an infinite entry."
  expected = Float32(Inf)
  observed = IndBox(-one(Float32), one(Float32))(Float32[-Inf, 4])
end

@row begin
  id = "special_values.IndBallL2.Float32.nan.prox"; case = ""; family = :special_values; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "E2"
  problem = "NaN propagates only to dependent coordinates; coupled blocks are entirely NaN."
  expected = Float32[NaN, NaN]
  observed = prox(IndBallL2(one(Float32)), Float32[NaN, 4], one(Float32))
end

@row begin
  id = "special_values.IndBallL2.Float32.nan.value"; case = ""; family = :special_values; oracle = :value
  types = "Float32"; backend = :cpu; rules = "E2"
  problem = "NaN propagates only to dependent coordinates; coupled blocks are entirely NaN."
  expected = Float32(NaN)
  observed = IndBallL2(one(Float32))(Float32[NaN, 4])
  broken = (group = :G2, baseline = Returned(v -> same(v, Inf32)))
end

@row begin
  id = "special_values.IndBallL2.Float32.nan-beats-inf"; case = ""; family = :special_values; oracle = :value
  types = "Float32"; backend = :cpu; rules = "E2"
  problem = "NaN beats infinity in a sum or membership decision."
  expected = Float32(NaN)
  observed = IndBallL2(one(Float32))(Float32[Inf, NaN])
  broken = (group = :G2, baseline = Returned(v -> same(v, Inf32)))
end

@row begin
  id = "special_values.IndBallL2.Float32.infinite-value"; case = ""; family = :special_values; oracle = :value
  types = "Float32"; backend = :cpu; rules = "E3 E5"
  problem = "Norm or indicator value is positive infinity at an infinite entry."
  expected = Float32(Inf)
  observed = IndBallL2(one(Float32))(Float32[-Inf, 4])
end

@row begin
  id = "special_values.IndSimplex.Float32.nan.prox"; case = ""; family = :special_values; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "E2"
  problem = "NaN propagates only to dependent coordinates; coupled blocks are entirely NaN."
  expected = Float32[NaN, NaN]
  observed = prox(IndSimplex(one(Float32)), Float32[NaN, 4], one(Float32))
end

@row begin
  id = "special_values.IndSimplex.Float32.nan.value"; case = ""; family = :special_values; oracle = :value
  types = "Float32"; backend = :cpu; rules = "E2"
  problem = "NaN propagates only to dependent coordinates; coupled blocks are entirely NaN."
  expected = Float32(NaN)
  observed = IndSimplex(one(Float32))(Float32[NaN, 4])
  broken = (group = :G2, baseline = Returned(v -> same(v, Inf32)))
end

@row begin
  id = "special_values.IndSimplex.Float32.nan-beats-inf"; case = ""; family = :special_values; oracle = :value
  types = "Float32"; backend = :cpu; rules = "E2"
  problem = "NaN beats infinity in a sum or membership decision."
  expected = Float32(NaN)
  observed = IndSimplex(one(Float32))(Float32[Inf, NaN])
  broken = (group = :G2, baseline = Returned(v -> same(v, Inf32)))
end

@row begin
  id = "special_values.IndSimplex.Float32.infinite-value"; case = ""; family = :special_values; oracle = :value
  types = "Float32"; backend = :cpu; rules = "E3 E5"
  problem = "Norm or indicator value is positive infinity at an infinite entry."
  expected = Float32(Inf)
  observed = IndSimplex(one(Float32))(Float32[-Inf, 4])
end

@row begin
  id = "special_values.SqrNormL2.Float32.nan.prox"; case = ""; family = :special_values; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "E2"
  problem = "NaN propagates only to dependent coordinates; coupled blocks are entirely NaN."
  expected = Float32[NaN, 2]
  observed = prox(SqrNormL2(one(Float32)), Float32[NaN, 4], one(Float32))
end

@row begin
  id = "special_values.SqrNormL2.Float32.nan.value"; case = ""; family = :special_values; oracle = :value
  types = "Float32"; backend = :cpu; rules = "E2"
  problem = "NaN propagates only to dependent coordinates; coupled blocks are entirely NaN."
  expected = Float32(NaN)
  observed = SqrNormL2(one(Float32))(Float32[NaN, 4])
end

@row begin
  id = "special_values.SqrNormL2.Float32.nan-beats-inf"; case = ""; family = :special_values; oracle = :value
  types = "Float32"; backend = :cpu; rules = "E2"
  problem = "NaN beats infinity in a sum or membership decision."
  expected = Float32(NaN)
  observed = SqrNormL2(one(Float32))(Float32[Inf, NaN])
end

@row begin
  id = "special_values.SqrNormL2.Float32.infinite-value"; case = ""; family = :special_values; oracle = :value
  types = "Float32"; backend = :cpu; rules = "E3 E5"
  problem = "Norm or indicator value is positive infinity at an infinite entry."
  expected = Float32(Inf)
  observed = SqrNormL2(one(Float32))(Float32[-Inf, 4])
end

@row begin
  id = "special_values.ridge.Float32.negative-infinity.gradient"; case = ""; family = :special_values; oracle = :gradient
  types = "Float32"; backend = :cpu; rules = "E3"
  problem = "Uncentered ridge has the negative-infinite gradient and proximal limit."
  expected = Float32[-Inf]
  observed = gradient(SqrNormL2(one(Float32)), Float32[-Inf])
end

@row begin
  id = "special_values.ridge.Float32.negative-infinity.prox"; case = ""; family = :special_values; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "E3"
  problem = "Uncentered ridge has the negative-infinite gradient and proximal limit."
  expected = Float32[-Inf]
  observed = prox(SqrNormL2(one(Float32)), Float32[-Inf], one(Float32))
end

@row begin
  id = "special_values.BernoulliLoss.Float32.mask.value"; case = ""; family = :special_values; oracle = :value
  types = "Float32"; backend = :cpu; rules = "E2 E6"
  problem = "NaN probability precedes a zero observation weight."
  expected = Float32(NaN)
  observed = BernoulliLoss(Float32[1], zero(Float32))(Float32[NaN])
end

@row begin
  id = "special_values.BernoulliLoss.Float32.mask.gradient"; case = ""; family = :special_values; oracle = :gradient
  types = "Float32"; backend = :cpu; rules = "E2 E6"
  problem = "NaN probability precedes a zero observation weight."
  expected = Float32[NaN]
  observed = gradient(BernoulliLoss(Float32[1], zero(Float32)), Float32[NaN])
end

@row begin
  id = "special_values.BernoulliLoss.Float32.outside.value"; case = ""; family = :special_values; oracle = :value
  types = "Float32"; backend = :cpu; rules = "E6"
  problem = "Finite probability outside domain gives positive infinity."
  expected = Float32(Inf)
  observed = BernoulliLoss(Float32[0.5])(Float32[-1])
end

@row begin
  id = "special_values.BernoulliLoss.Float32.outside.gradient"; case = ""; family = :special_values; oracle = :gradient
  types = "Float32"; backend = :cpu; rules = "E6"
  problem = "Supported derivative outside domain is NaN (decision 44)."
  expected = Float32[NaN]
  observed = gradient(BernoulliLoss(Float32[0.5]), Float32[-1])
end

@row begin
  id = "special_values.NuclearNorm.Float32.nan.value"; case = ""; family = :special_values; oracle = :value
  types = "Float32"; backend = :cpu; rules = "E2"
  problem = "Spectral block is entirely undefined at a NaN entry, before LAPACK."
  expected = Float32(NaN)
  observed = NuclearNorm(one(Float32))(Float32[NaN 0; 0 1])
  broken = (group = :G6, baseline = Returned(v -> same(v, Inf32)))
end

@row begin
  id = "special_values.NuclearNorm.Float32.nan.prox"; case = ""; family = :special_values; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "E2"
  problem = "Spectral block is entirely undefined at a NaN entry, before LAPACK."
  expected = fill(Float32(NaN), 2, 2)
  observed = prox(NuclearNorm(one(Float32)), Float32[NaN 0; 0 1], one(Float32))
end

@row begin
  id = "special_values.NormL1.Float64.nan.prox"; case = ""; family = :special_values; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "E2"
  problem = "NaN propagates only to dependent coordinates; coupled blocks are entirely NaN."
  expected = Float64[NaN, 3]
  observed = prox(NormL1(one(Float64)), Float64[NaN, 4], one(Float64))
end

@row begin
  id = "special_values.NormL1.Float64.nan.value"; case = ""; family = :special_values; oracle = :value
  types = "Float64"; backend = :cpu; rules = "E2"
  problem = "NaN propagates only to dependent coordinates; coupled blocks are entirely NaN."
  expected = Float64(NaN)
  observed = NormL1(one(Float64))(Float64[NaN, 4])
end

@row begin
  id = "special_values.NormL1.Float64.nan-beats-inf"; case = ""; family = :special_values; oracle = :value
  types = "Float64"; backend = :cpu; rules = "E2"
  problem = "NaN beats infinity in a sum or membership decision."
  expected = Float64(NaN)
  observed = NormL1(one(Float64))(Float64[Inf, NaN])
end

@row begin
  id = "special_values.NormL1.Float64.infinite-value"; case = ""; family = :special_values; oracle = :value
  types = "Float64"; backend = :cpu; rules = "E3 E5"
  problem = "Norm or indicator value is positive infinity at an infinite entry."
  expected = Float64(Inf)
  observed = NormL1(one(Float64))(Float64[-Inf, 4])
end

@row begin
  id = "special_values.NormL2.Float64.nan.prox"; case = ""; family = :special_values; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "E2"
  problem = "NaN propagates only to dependent coordinates; coupled blocks are entirely NaN."
  expected = Float64[NaN, NaN]
  observed = prox(NormL2(one(Float64)), Float64[NaN, 4], one(Float64))
  broken = (group = :G2, baseline = Returned(v -> same(v, [0.0, 0.0])))
end

@row begin
  id = "special_values.NormL2.Float64.nan.value"; case = ""; family = :special_values; oracle = :value
  types = "Float64"; backend = :cpu; rules = "E2"
  problem = "NaN propagates only to dependent coordinates; coupled blocks are entirely NaN."
  expected = Float64(NaN)
  observed = NormL2(one(Float64))(Float64[NaN, 4])
end

@row begin
  id = "special_values.NormL2.Float64.nan-beats-inf"; case = ""; family = :special_values; oracle = :value
  types = "Float64"; backend = :cpu; rules = "E2"
  problem = "NaN beats infinity in a sum or membership decision."
  expected = Float64(NaN)
  observed = NormL2(one(Float64))(Float64[Inf, NaN])
end

@row begin
  id = "special_values.NormL2.Float64.infinite-value"; case = ""; family = :special_values; oracle = :value
  types = "Float64"; backend = :cpu; rules = "E3 E5"
  problem = "Norm or indicator value is positive infinity at an infinite entry."
  expected = Float64(Inf)
  observed = NormL2(one(Float64))(Float64[-Inf, 4])
end

@row begin
  id = "special_values.GroupNormL2.Float64.nan.prox"; case = ""; family = :special_values; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "E2"
  problem = "NaN propagates only to dependent coordinates; coupled blocks are entirely NaN."
  expected = Float64[NaN, NaN]
  observed = prox(GroupNormL2(one(Float64), [1:2], Float64[1]), Float64[NaN, 4], one(Float64))
  broken = (group = :G2, baseline = Returned(v -> same(v, [0.0, 0.0])))
end

@row begin
  id = "special_values.GroupNormL2.Float64.nan.value"; case = ""; family = :special_values; oracle = :value
  types = "Float64"; backend = :cpu; rules = "E2"
  problem = "NaN propagates only to dependent coordinates; coupled blocks are entirely NaN."
  expected = Float64(NaN)
  observed = GroupNormL2(one(Float64), [1:2], Float64[1])(Float64[NaN, 4])
end

@row begin
  id = "special_values.GroupNormL2.Float64.nan-beats-inf"; case = ""; family = :special_values; oracle = :value
  types = "Float64"; backend = :cpu; rules = "E2"
  problem = "NaN beats infinity in a sum or membership decision."
  expected = Float64(NaN)
  observed = GroupNormL2(one(Float64), [1:2], Float64[1])(Float64[Inf, NaN])
end

@row begin
  id = "special_values.GroupNormL2.Float64.infinite-value"; case = ""; family = :special_values; oracle = :value
  types = "Float64"; backend = :cpu; rules = "E3 E5"
  problem = "Norm or indicator value is positive infinity at an infinite entry."
  expected = Float64(Inf)
  observed = GroupNormL2(one(Float64), [1:2], Float64[1])(Float64[-Inf, 4])
end

@row begin
  id = "special_values.NormLinf.Float64.nan.prox"; case = ""; family = :special_values; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "E2"
  problem = "NaN propagates only to dependent coordinates; coupled blocks are entirely NaN."
  expected = Float64[NaN, NaN]
  observed = prox(NormLinf(one(Float64)), Float64[NaN, 4], one(Float64))
end

@row begin
  id = "special_values.NormLinf.Float64.nan.value"; case = ""; family = :special_values; oracle = :value
  types = "Float64"; backend = :cpu; rules = "E2"
  problem = "NaN propagates only to dependent coordinates; coupled blocks are entirely NaN."
  expected = Float64(NaN)
  observed = NormLinf(one(Float64))(Float64[NaN, 4])
end

@row begin
  id = "special_values.NormLinf.Float64.nan-beats-inf"; case = ""; family = :special_values; oracle = :value
  types = "Float64"; backend = :cpu; rules = "E2"
  problem = "NaN beats infinity in a sum or membership decision."
  expected = Float64(NaN)
  observed = NormLinf(one(Float64))(Float64[Inf, NaN])
end

@row begin
  id = "special_values.NormLinf.Float64.infinite-value"; case = ""; family = :special_values; oracle = :value
  types = "Float64"; backend = :cpu; rules = "E3 E5"
  problem = "Norm or indicator value is positive infinity at an infinite entry."
  expected = Float64(Inf)
  observed = NormLinf(one(Float64))(Float64[-Inf, 4])
end

@row begin
  id = "special_values.IndBox.Float64.nan.prox"; case = ""; family = :special_values; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "E2"
  problem = "NaN propagates only to dependent coordinates; coupled blocks are entirely NaN."
  expected = Float64[NaN, 1]
  observed = prox(IndBox(-one(Float64), one(Float64)), Float64[NaN, 4], one(Float64))
end

@row begin
  id = "special_values.IndBox.Float64.nan.value"; case = ""; family = :special_values; oracle = :value
  types = "Float64"; backend = :cpu; rules = "E2"
  problem = "NaN propagates only to dependent coordinates; coupled blocks are entirely NaN."
  expected = Float64(NaN)
  observed = IndBox(-one(Float64), one(Float64))(Float64[NaN, 4])
  broken = (group = :G2, baseline = Returned(v -> same(v, Inf)))
end

@row begin
  id = "special_values.IndBox.Float64.nan-beats-inf"; case = ""; family = :special_values; oracle = :value
  types = "Float64"; backend = :cpu; rules = "E2"
  problem = "NaN beats infinity in a sum or membership decision."
  expected = Float64(NaN)
  observed = IndBox(-one(Float64), one(Float64))(Float64[Inf, NaN])
  broken = (group = :G2, baseline = Returned(v -> same(v, Inf)))
end

@row begin
  id = "special_values.IndBox.Float64.infinite-value"; case = ""; family = :special_values; oracle = :value
  types = "Float64"; backend = :cpu; rules = "E3 E5"
  problem = "Norm or indicator value is positive infinity at an infinite entry."
  expected = Float64(Inf)
  observed = IndBox(-one(Float64), one(Float64))(Float64[-Inf, 4])
end

@row begin
  id = "special_values.IndBallL2.Float64.nan.prox"; case = ""; family = :special_values; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "E2"
  problem = "NaN propagates only to dependent coordinates; coupled blocks are entirely NaN."
  expected = Float64[NaN, NaN]
  observed = prox(IndBallL2(one(Float64)), Float64[NaN, 4], one(Float64))
end

@row begin
  id = "special_values.IndBallL2.Float64.nan.value"; case = ""; family = :special_values; oracle = :value
  types = "Float64"; backend = :cpu; rules = "E2"
  problem = "NaN propagates only to dependent coordinates; coupled blocks are entirely NaN."
  expected = Float64(NaN)
  observed = IndBallL2(one(Float64))(Float64[NaN, 4])
  broken = (group = :G2, baseline = Returned(v -> same(v, Inf)))
end

@row begin
  id = "special_values.IndBallL2.Float64.nan-beats-inf"; case = ""; family = :special_values; oracle = :value
  types = "Float64"; backend = :cpu; rules = "E2"
  problem = "NaN beats infinity in a sum or membership decision."
  expected = Float64(NaN)
  observed = IndBallL2(one(Float64))(Float64[Inf, NaN])
  broken = (group = :G2, baseline = Returned(v -> same(v, Inf)))
end

@row begin
  id = "special_values.IndBallL2.Float64.infinite-value"; case = ""; family = :special_values; oracle = :value
  types = "Float64"; backend = :cpu; rules = "E3 E5"
  problem = "Norm or indicator value is positive infinity at an infinite entry."
  expected = Float64(Inf)
  observed = IndBallL2(one(Float64))(Float64[-Inf, 4])
end

@row begin
  id = "special_values.IndSimplex.Float64.nan.prox"; case = ""; family = :special_values; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "E2"
  problem = "NaN propagates only to dependent coordinates; coupled blocks are entirely NaN."
  expected = Float64[NaN, NaN]
  observed = prox(IndSimplex(one(Float64)), Float64[NaN, 4], one(Float64))
end

@row begin
  id = "special_values.IndSimplex.Float64.nan.value"; case = ""; family = :special_values; oracle = :value
  types = "Float64"; backend = :cpu; rules = "E2"
  problem = "NaN propagates only to dependent coordinates; coupled blocks are entirely NaN."
  expected = Float64(NaN)
  observed = IndSimplex(one(Float64))(Float64[NaN, 4])
  broken = (group = :G2, baseline = Returned(v -> same(v, Inf)))
end

@row begin
  id = "special_values.IndSimplex.Float64.nan-beats-inf"; case = ""; family = :special_values; oracle = :value
  types = "Float64"; backend = :cpu; rules = "E2"
  problem = "NaN beats infinity in a sum or membership decision."
  expected = Float64(NaN)
  observed = IndSimplex(one(Float64))(Float64[Inf, NaN])
  broken = (group = :G2, baseline = Returned(v -> same(v, Inf)))
end

@row begin
  id = "special_values.IndSimplex.Float64.infinite-value"; case = ""; family = :special_values; oracle = :value
  types = "Float64"; backend = :cpu; rules = "E3 E5"
  problem = "Norm or indicator value is positive infinity at an infinite entry."
  expected = Float64(Inf)
  observed = IndSimplex(one(Float64))(Float64[-Inf, 4])
end

@row begin
  id = "special_values.SqrNormL2.Float64.nan.prox"; case = ""; family = :special_values; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "E2"
  problem = "NaN propagates only to dependent coordinates; coupled blocks are entirely NaN."
  expected = Float64[NaN, 2]
  observed = prox(SqrNormL2(one(Float64)), Float64[NaN, 4], one(Float64))
end

@row begin
  id = "special_values.SqrNormL2.Float64.nan.value"; case = ""; family = :special_values; oracle = :value
  types = "Float64"; backend = :cpu; rules = "E2"
  problem = "NaN propagates only to dependent coordinates; coupled blocks are entirely NaN."
  expected = Float64(NaN)
  observed = SqrNormL2(one(Float64))(Float64[NaN, 4])
end

@row begin
  id = "special_values.SqrNormL2.Float64.nan-beats-inf"; case = ""; family = :special_values; oracle = :value
  types = "Float64"; backend = :cpu; rules = "E2"
  problem = "NaN beats infinity in a sum or membership decision."
  expected = Float64(NaN)
  observed = SqrNormL2(one(Float64))(Float64[Inf, NaN])
end

@row begin
  id = "special_values.SqrNormL2.Float64.infinite-value"; case = ""; family = :special_values; oracle = :value
  types = "Float64"; backend = :cpu; rules = "E3 E5"
  problem = "Norm or indicator value is positive infinity at an infinite entry."
  expected = Float64(Inf)
  observed = SqrNormL2(one(Float64))(Float64[-Inf, 4])
end

@row begin
  id = "special_values.ridge.Float64.negative-infinity.gradient"; case = ""; family = :special_values; oracle = :gradient
  types = "Float64"; backend = :cpu; rules = "E3"
  problem = "Uncentered ridge has the negative-infinite gradient and proximal limit."
  expected = Float64[-Inf]
  observed = gradient(SqrNormL2(one(Float64)), Float64[-Inf])
end

@row begin
  id = "special_values.ridge.Float64.negative-infinity.prox"; case = ""; family = :special_values; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "E3"
  problem = "Uncentered ridge has the negative-infinite gradient and proximal limit."
  expected = Float64[-Inf]
  observed = prox(SqrNormL2(one(Float64)), Float64[-Inf], one(Float64))
end

@row begin
  id = "special_values.BernoulliLoss.Float64.mask.value"; case = ""; family = :special_values; oracle = :value
  types = "Float64"; backend = :cpu; rules = "E2 E6"
  problem = "NaN probability precedes a zero observation weight."
  expected = Float64(NaN)
  observed = BernoulliLoss(Float64[1], zero(Float64))(Float64[NaN])
end

@row begin
  id = "special_values.BernoulliLoss.Float64.mask.gradient"; case = ""; family = :special_values; oracle = :gradient
  types = "Float64"; backend = :cpu; rules = "E2 E6"
  problem = "NaN probability precedes a zero observation weight."
  expected = Float64[NaN]
  observed = gradient(BernoulliLoss(Float64[1], zero(Float64)), Float64[NaN])
end

@row begin
  id = "special_values.BernoulliLoss.Float64.outside.value"; case = ""; family = :special_values; oracle = :value
  types = "Float64"; backend = :cpu; rules = "E6"
  problem = "Finite probability outside domain gives positive infinity."
  expected = Float64(Inf)
  observed = BernoulliLoss(Float64[0.5])(Float64[-1])
end

@row begin
  id = "special_values.BernoulliLoss.Float64.outside.gradient"; case = ""; family = :special_values; oracle = :gradient
  types = "Float64"; backend = :cpu; rules = "E6"
  problem = "Supported derivative outside domain is NaN (decision 44)."
  expected = Float64[NaN]
  observed = gradient(BernoulliLoss(Float64[0.5]), Float64[-1])
end

@row begin
  id = "special_values.NuclearNorm.Float64.nan.value"; case = ""; family = :special_values; oracle = :value
  types = "Float64"; backend = :cpu; rules = "E2"
  problem = "Spectral block is entirely undefined at a NaN entry, before LAPACK."
  expected = Float64(NaN)
  observed = NuclearNorm(one(Float64))(Float64[NaN 0; 0 1])
  broken = (group = :G6, baseline = Returned(v -> same(v, Inf)))
end

@row begin
  id = "special_values.NuclearNorm.Float64.nan.prox"; case = ""; family = :special_values; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "E2"
  problem = "Spectral block is entirely undefined at a NaN entry, before LAPACK."
  expected = fill(Float64(NaN), 2, 2)
  observed = prox(NuclearNorm(one(Float64)), Float64[NaN 0; 0 1], one(Float64))
end

@row begin
  id = "special_values.zero-weighted.value"; case = ""; family = :special_values; oracle = :value
  types = "Float64"; backend = :cpu; rules = "E4"
  problem = "Zero level retains infinite-weight domain constraint."
  expected = Inf
  observed = NormL1(0.0, [0.0, Inf])([3.0, 4.0])
end

@row begin
  id = "special_values.zero-weighted.prox"; case = ""; family = :special_values; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "E4"
  problem = "Zero level retains infinite-weight domain constraint."
  expected = [3.0, 0.0]
  observed = prox(NormL1(0.0, [0.0, Inf]), [3.0, 4.0], 1.0)
end

@row begin
  id = "special_values.zero-box.value"; case = ""; family = :special_values; oracle = :value
  types = "Float64"; backend = :cpu; rules = "E4"
  problem = "Zero scaling retains the box domain."
  expected = Inf
  observed = (0.0 * IndBox(0.0, 1.0))([2.0])
end

@row begin
  id = "special_values.zero-box.prox"; case = ""; family = :special_values; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "E4"
  problem = "Zero scaling retains the box domain."
  expected = [1.0]
  observed = prox(0.0 * IndBox(0.0, 1.0), [2.0], 1.0)
end

@row begin
  id = "special_values.zero-NegLog"; case = ""; family = :special_values; oracle = :constructor
  types = "Float64"; backend = :cpu; rules = "E4"
  problem = "Open half-line has no domain-preserving projection."
  expected = Throws(:ArgumentError)
  observed = Scaled(NegLog(), 0.0)
end

@row begin
  id = "special_values.LinearFunction.inf"; case = ""; family = :special_values; oracle = :constructor
  types = "Float64"; backend = :cpu; rules = "E4"
  problem = "Decision 47 rejects nonfinite linear coefficients in their own type."
  expected = Throws(:ArgumentError)
  observed = let v = Inf; LinearFunction([v]); true end
  broken = (group = :G2, baseline = Returned(v -> same(v, true)))
end

@row begin
  id = "special_values.LinearFunction.negative-inf"; case = ""; family = :special_values; oracle = :constructor
  types = "Float64"; backend = :cpu; rules = "E4"
  problem = "Decision 47 rejects nonfinite linear coefficients in their own type."
  expected = Throws(:ArgumentError)
  observed = let v = -Inf; LinearFunction([v]); true end
  broken = (group = :G2, baseline = Returned(v -> same(v, true)))
end

@row begin
  id = "special_values.LinearFunction.nan"; case = ""; family = :special_values; oracle = :constructor
  types = "Float64"; backend = :cpu; rules = "E4"
  problem = "Decision 47 rejects nonfinite linear coefficients in their own type."
  expected = Throws(:ArgumentError)
  observed = let v = NaN; LinearFunction([v]); true end
  broken = (group = :G2, baseline = Returned(v -> same(v, true)))
end

@row begin
  id = "special_values.Tilt.inf"; case = ""; family = :special_values; oracle = :constructor
  types = "Float64"; backend = :cpu; rules = "E4"
  problem = "Decision 47 rejects nonfinite linear coefficients in their own type."
  expected = Throws(:ArgumentError)
  observed = let v = Inf; Tilt(NormL1(), [v]); true end
  broken = (group = :G2, baseline = Returned(v -> same(v, true)))
end

@row begin
  id = "special_values.Tilt.negative-inf"; case = ""; family = :special_values; oracle = :constructor
  types = "Float64"; backend = :cpu; rules = "E4"
  problem = "Decision 47 rejects nonfinite linear coefficients in their own type."
  expected = Throws(:ArgumentError)
  observed = let v = -Inf; Tilt(NormL1(), [v]); true end
  broken = (group = :G2, baseline = Returned(v -> same(v, true)))
end

@row begin
  id = "special_values.Tilt.nan"; case = ""; family = :special_values; oracle = :constructor
  types = "Float64"; backend = :cpu; rules = "E4"
  problem = "Decision 47 rejects nonfinite linear coefficients in their own type."
  expected = Throws(:ArgumentError)
  observed = let v = NaN; Tilt(NormL1(), [v]); true end
  broken = (group = :G2, baseline = Returned(v -> same(v, true)))
end

@row begin
  id = "special_values.ridge.Float32.nan.gradient"; case = ""; family = :special_values; oracle = :gradient
  types = "Float32"; backend = :cpu; rules = "E2"
  problem = "Separable NaN gradient leaves the other coordinate defined."
  expected = Float32[NaN, 4]
  observed = gradient(SqrNormL2(one(Float32)), Float32[NaN, 4])
end

@row begin
  id = "special_values.NegEntropy.Float32.nan"; case = ""; family = :special_values; oracle = :value
  types = "Float32"; backend = :cpu; rules = "E2"
  problem = "NaN entropy argument is undefined, not outside-domain infinity."
  expected = Float32(NaN)
  observed = NegEntropy(one(Float32))(Float32[NaN])
  broken = (group = :G2, baseline = Returned(v -> same(v, Inf32)))
end

@row begin
  id = "special_values.NegEntropy.Float32.nan-beats-inf"; case = ""; family = :special_values; oracle = :value
  types = "Float32"; backend = :cpu; rules = "E2"
  problem = "NaN beats infinity in the entropy reduction."
  expected = Float32(NaN)
  observed = NegEntropy(one(Float32))(Float32[Inf, NaN])
  broken = (group = :G2, baseline = Returned(v -> same(v, Inf32)))
end

@row begin
  id = "special_values.NegEntropy.Float32.nan.prox"; case = ""; family = :special_values; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "E2 R2"
  problem = "Scalar entropy stationarity y+log(y)+1=2 gives y=1; R2 smooth budget 4 ULP."
  expected = Float32[NaN, 1]
  observed = prox(NegEntropy(one(Float32)), Float32[NaN, 2], one(Float32))
  compare = (a, b) -> length(a) == 2 && same(a[1], b[1]) && within_ulps(a[2], b[2], 4)
end

@row begin
  id = "special_values.IndNonnegative.Float32.nan.value"; case = ""; family = :special_values; oracle = :value
  types = "Float32"; backend = :cpu; rules = "E2"
  problem = "Separable versus coupled indicator NaN scope."
  expected = Float32(NaN)
  observed = IndNonnegative()(Float32[NaN, 4])
  broken = (group = :G2, baseline = Returned(v -> same(v, Inf32)))
end

@row begin
  id = "special_values.IndNonnegative.Float32.nan.prox"; case = ""; family = :special_values; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "E2"
  problem = "Separable versus coupled indicator NaN scope."
  expected = Float32[NaN, 4]
  observed = prox(IndNonnegative(), Float32[NaN, 4], one(Float32))
end

@row begin
  id = "special_values.IndBallL1.Float32.nan.value"; case = ""; family = :special_values; oracle = :value
  types = "Float32"; backend = :cpu; rules = "E2"
  problem = "Separable versus coupled indicator NaN scope."
  expected = Float32(NaN)
  observed = IndBallL1(one(Float32))(Float32[NaN, 4])
  broken = (group = :G2, baseline = Returned(v -> same(v, Inf32)))
end

@row begin
  id = "special_values.IndBallL1.Float32.nan.prox"; case = ""; family = :special_values; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "E2"
  problem = "Separable versus coupled indicator NaN scope."
  expected = Float32[NaN, NaN]
  observed = prox(IndBallL1(one(Float32)), Float32[NaN, 4], one(Float32))
end

@row begin
  id = "special_values.ridge.Float64.nan.gradient"; case = ""; family = :special_values; oracle = :gradient
  types = "Float64"; backend = :cpu; rules = "E2"
  problem = "Separable NaN gradient leaves the other coordinate defined."
  expected = Float64[NaN, 4]
  observed = gradient(SqrNormL2(one(Float64)), Float64[NaN, 4])
end

@row begin
  id = "special_values.NegEntropy.Float64.nan"; case = ""; family = :special_values; oracle = :value
  types = "Float64"; backend = :cpu; rules = "E2"
  problem = "NaN entropy argument is undefined, not outside-domain infinity."
  expected = Float64(NaN)
  observed = NegEntropy(one(Float64))(Float64[NaN])
  broken = (group = :G2, baseline = Returned(v -> same(v, Inf)))
end

@row begin
  id = "special_values.NegEntropy.Float64.nan-beats-inf"; case = ""; family = :special_values; oracle = :value
  types = "Float64"; backend = :cpu; rules = "E2"
  problem = "NaN beats infinity in the entropy reduction."
  expected = Float64(NaN)
  observed = NegEntropy(one(Float64))(Float64[Inf, NaN])
  broken = (group = :G2, baseline = Returned(v -> same(v, Inf)))
end

@row begin
  id = "special_values.NegEntropy.Float64.nan.prox"; case = ""; family = :special_values; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "E2 R2"
  problem = "Scalar entropy stationarity y+log(y)+1=2 gives y=1; R2 smooth budget 4 ULP."
  expected = Float64[NaN, 1]
  observed = prox(NegEntropy(one(Float64)), Float64[NaN, 2], one(Float64))
  compare = (a, b) -> length(a) == 2 && same(a[1], b[1]) && within_ulps(a[2], b[2], 4)
end

@row begin
  id = "special_values.IndNonnegative.Float64.nan.value"; case = ""; family = :special_values; oracle = :value
  types = "Float64"; backend = :cpu; rules = "E2"
  problem = "Separable versus coupled indicator NaN scope."
  expected = Float64(NaN)
  observed = IndNonnegative()(Float64[NaN, 4])
  broken = (group = :G2, baseline = Returned(v -> same(v, Inf)))
end

@row begin
  id = "special_values.IndNonnegative.Float64.nan.prox"; case = ""; family = :special_values; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "E2"
  problem = "Separable versus coupled indicator NaN scope."
  expected = Float64[NaN, 4]
  observed = prox(IndNonnegative(), Float64[NaN, 4], one(Float64))
end

@row begin
  id = "special_values.IndBallL1.Float64.nan.value"; case = ""; family = :special_values; oracle = :value
  types = "Float64"; backend = :cpu; rules = "E2"
  problem = "Separable versus coupled indicator NaN scope."
  expected = Float64(NaN)
  observed = IndBallL1(one(Float64))(Float64[NaN, 4])
  broken = (group = :G2, baseline = Returned(v -> same(v, Inf)))
end

@row begin
  id = "special_values.IndBallL1.Float64.nan.prox"; case = ""; family = :special_values; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "E2"
  problem = "Separable versus coupled indicator NaN scope."
  expected = Float64[NaN, NaN]
  observed = prox(IndBallL1(one(Float64)), Float64[NaN, 4], one(Float64))
end

end
