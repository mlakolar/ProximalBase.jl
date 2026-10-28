module CompatTest

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays

const PB = ProximalBase

@testset "stdlib compatibility boundary" begin
  @testset "aliases" begin
    @test PB._BlasInt === LinearAlgebra.BlasInt
    @test PB._libblastrampoline === LinearAlgebra.libblastrampoline
    @test PB._potrf! === LAPACK.potrf!
    @test PB._potri! === LAPACK.potri!
    for name in (:AbstractSparseMatrixCSC, :SparseMatrixCSCOrView, :AdjOrTransSparseVectorOrView)
      alias = Symbol(:_, name)
      @test isdefined(PB, alias) == isdefined(SparseArrays, name)
      if isdefined(SparseArrays, name)
        @test getfield(PB, alias) === getfield(SparseArrays, name)
      end
    end
    for (name, alias) in ((:Dense, :_CHOLMODDense), (:VTypes, :_CHOLMODVTypes))
      available = isdefined(SparseArrays, :CHOLMOD) && isdefined(SparseArrays.CHOLMOD, name)
      @test isdefined(PB, alias) == available
      if available
        @test getfield(PB, alias) === getfield(SparseArrays.CHOLMOD, name)
      end
    end
  end

  @testset "BLAS symbols" begin
    library = Base.Libc.Libdl.dlopen(PB._libblastrampoline)
    for name in (:dsyevd_, :ssyevd_, :dsyevr_, :ssyevr_, :dgesdd_, :sgesdd_,
                 :dgeqrf_, :sgeqrf_, :dorgqr_, :sorgqr_)
      @eval @testset $(string(name)) begin
        @test PB._blasfunc($(QuoteNode(name))) === BLAS.@blasfunc($name)
        address = Base.Libc.Libdl.dlsym($library, PB._blasfunc($(QuoteNode(name))); throw_error = false)
        @test address !== nothing
      end
    end
  end

  @testset "source boundary" begin
    source = dirname(@__DIR__)
    private_names = r"(?<!\w)(?:BlasInt|libblastrampoline|@blasfunc|potrf!|potri!|nonzeroinds|AbstractSparseMatrixCSC|SparseMatrixCSCOrView|AdjOrTransSparseVectorOrView|CHOLMOD|VTypes)(?![\w!])"
    modules = Dict("LinearAlgebra" => LinearAlgebra, "SparseArrays" => SparseArrays,
                   "BLAS" => BLAS, "LAPACK" => LAPACK)
    qualified = r"(?=\b(LinearAlgebra|SparseArrays|BLAS|LAPACK)\.(@?[A-Za-z_]\w*!?))"
    imports = r"(?m)^\s*(?:using|import)\s+(LinearAlgebra|SparseArrays)(?:\.(BLAS|LAPACK))?\s*:\s*([^\n]+)"
    for directory in ("src", "ext"), (root, _, files) in walkdir(joinpath(source, directory)), file in files
      endswith(file, ".jl") || continue
      path = joinpath(root, file)
      path == joinpath(source, "src", "compat.jl") && continue
      @testset "$(relpath(path, source))" begin
        text = read(path, String)
        @test isempty(collect(eachmatch(private_names, text)))
        if VERSION >= v"1.11"
          for m in eachmatch(qualified, text)
            @test Base.ispublic(modules[m.captures[1]], Symbol(m.captures[2]))
          end
          for m in eachmatch(imports, text)
            M = modules[something(m.captures[2], m.captures[1])]
            for name in split(first(split(m.captures[3], '#')), ',')
              name = first(split(strip(name)))
              @test Base.ispublic(M, Symbol(name))
            end
          end
        end
      end
    end
  end
end

end
