using Documenter, Literate, ProximalBase, LinearAlgebra
BLAS.set_num_threads(1)
include("check_api.jl")
if get(ENV, "PB_DOCS_SKIP_ISOLATED", "false") != "true" || get(ENV, "CI", "false") == "true"
    include("run_examples.jl")
end

# Preserve the historical Literate pipeline: examples are authored once and each
# generated page is executed in its own Documenter example sandbox.
const EXAMPLES = joinpath(@__DIR__, "..", "examples")
const GENERATED = joinpath(@__DIR__, "src", "examples")
const TITLES = Dict(
    "lasso" => "Least-squares and lasso primitives",
    "proximal_gradient" => "Proximal-gradient iteration",
    "coordinate_operations" => "Coordinate-level operations",
    "sparse_working_sets" => "Sparse working sets",
    "block_storage" => "Ordered grouped storage",
    "symmetric_storage" => "Independent symmetric coordinates",
    "custom_function" => "A custom separable function",
    "block_prox_extension" => "A compact-vector block extension",
)
const EXAMPLE_ORDER = ["lasso", "proximal_gradient", "coordinate_operations", "sparse_working_sets",
                       "block_storage", "symmetric_storage", "custom_function", "block_prox_extension"]
Set(filter(endswith(".jl"), readdir(EXAMPLES))) == Set(name * ".jl" for name in EXAMPLE_ORDER) ||
    error("Every example must have a title and navigation entry")
const SECTION = r"^# -+ (.+?) -*$"m
headings(s) = replace(s, SECTION => m -> "# ## " * match(SECTION, m)[1] * "\n#")
rm(GENERATED; force=true, recursive=true)
mkpath(GENERATED)
example_pages = Pair{String,String}[]
for name in EXAMPLE_ORDER
    # The first authored comment is already the page title.
    Literate.markdown(joinpath(EXAMPLES,name * ".jl"), GENERATED;
        documenter=true, credit=false, preprocess=s -> headings(replace(s, r"^# " => "# # "; count=1)))
    push!(example_pages, TITLES[name] => "examples/$name.md")
end

# Render the reconciled contracts once; obsolete historical acceptance documents
# remain linked at their immutable snapshot instead of becoming live requirements.
const CONTRACT_PAGES = joinpath(@__DIR__, "src", "developer", "contracts")
rm(CONTRACT_PAGES; force=true, recursive=true)
mkpath(CONTRACT_PAGES)
for file in ("block-prox-0.6.md", "iterate-family-0.6.md", "symmetric-sparse-iterate-0.6.md")
    source = read(joinpath(@__DIR__, "design", file), String)
    metadata = "```@meta\nEditURL = \"../../../design/$file\"\n```\n\n"
    write(joinpath(CONTRACT_PAGES, file), metadata * source)
end

DocMeta.setdocmeta!(ProximalBase, :DocTestSetup, :(using ProximalBase, LinearAlgebra); recursive=true)
makedocs(;
    modules=[ProximalBase],
    sitename="ProximalBase.jl",
    authors="Mladen Kolar and contributors",
    checkdocs=:exports,
    doctest=true,
    warnonly=false,
    linkcheck=get(ENV, "DOCUMENTER_LINKCHECK", "false") == "true",
    format=Documenter.HTML(
        prettyurls=get(ENV, "CI", "false") == "true",
        collapselevel=1,
        canonical="https://mlakolar.github.io/ProximalBase.jl/stable/",
        mathengine=Documenter.KaTeX(),
        size_threshold=400_000,
        edit_link="master",
    ),
    pages=[
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "Working with functions" => [
            "Function objects and capabilities" => "function_objects.md",
            "Operations on functions" => "operations.md",
            "Proximal operators and gradients" => "interface.md",
            "Writing your own function" => "extending.md",
        ],
        "Function library" => [
            "Browse available functions" => "function_library.md",
            "Smooth functions and losses" => "smooth.md",
            "Penalties" => "penalties.md",
            "Functions built from functions" => "calculus.md",
        ],
        "Building optimization algorithms" => [
            "Overview" => "optimization_algorithms.md",
            "Proximal gradient methods" => "proximal_gradient.md",
            "Coordinate descent and active sets" => "coordinate_descent.md",
        ],
        "Iterates" => [
            "Overview and working sets" => "iterates.md",
            "SparseIterate" => "sparse_iterate.md",
            "SymmetricSparseIterate" => "symmetric_sparse_iterate.md",
            "BlockIterate" => "block_iterate.md",
            "Coordinate oracles" => "iterate_oracles.md",
            "Symmetric parameters" => "symmetric_parameters.md",
        ],
        "Compatibility and migration" => "migration.md",
        "Examples" => example_pages,
        "Developer Documentation" => [
            "Overview and contract map" => "developer/index.md",
            "System architecture" => "architecture.md",
            "SparseIterate internals" => "developer/sparse_storage.md",
            "Symmetric owner internals" => "developer/symmetric_storage.md",
            "BlockIterate internals" => "developer/block_storage.md",
            "Coordinate views and aliasing" => "developer/views_aliasing.md",
            "Function and oracle dispatch" => "developer/oracle_dispatch.md",
            "Numerical usage: arrays, precision and ownership" => "numerical_usage.md",
            "Numerical contract" => "numerics.md",
            "Pure block proximal contract" => "developer/contracts/block-prox-0.6.md",
            "Iterate-family contract" => "developer/contracts/iterate-family-0.6.md",
            "Symmetric-owner contract" => "developer/contracts/symmetric-sparse-iterate-0.6.md",
            "Conformance and change process" => "developer/change_process.md",
        ],
        "API reference" => [
            "Overview" => "api.md",
            "The interface" => "api/interface.md",
            "Functions" => "api/functions.md",
            "Functions built from functions" => "api/calculus.md",
            "Iterates and utilities" => "api/iterates.md",
            "Coverage matrix" => "api/coverage.md",
        ],
        hide("storage.md"),
    ],
)
