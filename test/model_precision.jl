module ModelPrecisionTest

using Test
using ProximalBase

const PB = ProximalBase

@testset "Model precision" begin
  for (T, want) in ((Float32, Float32), (Float64, Float64), (BigFloat, BigFloat),
                    (Int, Float64), (BigInt, BigFloat), (Bool, Float64))
    @test PB._model_precision(T) === want
    @test one(PB._model_precision(T)) isa want
    @test PB._model_precision(Float32, T) === want
  end
  @test PB._model_precision(Float32, Float64, Int, Bool) === Float64
  @test PB._model_precision(Float64, BigInt, Float32) === BigFloat
  @test PB._model_precision(BigFloat, Float64) === BigFloat
  for types in ((ComplexF32,), (Float32, ComplexF64), (Complex{Int}, Float64))
    @test_throws ArgumentError PB._model_precision(types...)
    @test occursin("requires real entries", try
      PB._model_precision(types...)
    catch e
      sprint(showerror, e)
    end)
  end
end

@testset "Float16 roles" begin
  @test PB._check_float16_roles() === nothing
  @test PB._check_float16_roles(argument = Float32, destination = Float64,
                               direction = BigFloat, data = Int) === nothing
  for role in (:argument, :destination, :direction, :data)
    e = try
      PB._check_float16_roles(; role => Float16)
    catch e
      e
    end
    @test e isa ArgumentError
    @test e.msg == "Float16 is not supported as $role in ProximalBase 0.6; use Float32 or Float64"
  end
  @test_throws r"as argument" PB._check_float16_roles(argument = Float16, data = Float16)
  @test_throws r"as data" PB._check_float16_roles(data = Matrix{Float16})
  # Parameter and step types are intentionally absent from the role check.
  parameter, step = Float16(1), Float16(0.5)
  @test PB._check_float16_roles(argument = Float32, destination = Float32) === nothing
  @test PB._model_precision(Float32, typeof(parameter), typeof(step)) === Float32
end

@testset "PrecisionError" begin
  e = PrecisionError("x")
  @test e.msg == "x"
  @test e isa Exception
  @test !(e isa ArgumentError)
  @test :PrecisionError in names(PB)
  @test sprint(showerror, e) == "PrecisionError: x"
  io = IOBuffer()
  showerror(io, e)
  @test String(take!(io)) == "PrecisionError: x"
end

end
