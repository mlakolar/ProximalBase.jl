# Build a throwaway preview of the proposed navigation with the prototypes in place.
# From the repository root:
#   julia --project=docs docs/design/docs-redesign-0.7/preview/build_preview.jl [OUTDIR]
# OUTDIR must not exist or must be empty; without it, a new temporary directory is used.
# Nothing outside OUTDIR is written or deleted; docs/src is only read and copied to OUTDIR/src.
# This is a preview, not the strict build: docstring coverage, doctests and the link
# check are off. Every @example block runs and every cross-reference must resolve.
using Documenter, Literate, ProximalBase, LinearAlgebra

const ROOT = normpath(joinpath(@__DIR__, "..", "..", "..", ".."))
const DESIGN = normpath(joinpath(@__DIR__, ".."))
const OUT = length(ARGS) >= 1 ? abspath(ARGS[1]) : mktempdir()
isdir(OUT) && !isempty(readdir(OUT)) && error("$OUT is not empty; give a new or empty directory")
mkpath(OUT)
const SRC = joinpath(OUT, "src")
cp(joinpath(ROOT, "docs", "src"), SRC)
rm(joinpath(SRC, "examples"); force=true, recursive=true)
rm(joinpath(SRC, "developer", "contracts"); force=true, recursive=true)

# Examples and design contracts from the same inputs as docs/make.jl, with preview
# titles and order for the examples and without the EditURL metadata of the contracts.
const EXAMPLES = ["lasso" => "Evaluate a lasso objective and its proximal step",
    "proximal_gradient" => "Write a proximal-gradient loop",
    "coordinate_operations" => "Update one coordinate at a time",
    "sparse_working_sets" => "Keep a working set with SparseIterate",
    "block_storage" => "Store groups of coefficients with BlockIterate",
    "symmetric_storage" => "Work with a symmetric matrix variable",
    "group_lasso_prox" => "Take group-lasso proximal steps",
    "custom_function" => "Write a separable function",
    "block_prox_extension" => "Add a block proximal operator to your function"]
const SECTION = r"^# -+ (.+?) -*$"m
headings(s) = replace(s, SECTION => m -> "# ## " * match(SECTION, m)[1] * "\n#")
mkpath(joinpath(SRC, "examples"))
for (name, _) in EXAMPLES
    Literate.markdown(joinpath(ROOT, "examples", name * ".jl"), joinpath(SRC, "examples");
        documenter=true, credit=false, preprocess=s -> headings(replace(s, r"^# " => "# # "; count=1)))
end
mkpath(joinpath(SRC, "developer", "contracts"))
for file in ("block-prox-0.6.md", "iterate-family-0.6.md", "symmetric-sparse-iterate-0.6.md")
    cp(joinpath(ROOT, "docs", "design", file), joinpath(SRC, "developer", "contracts", file))
end

# Prototype figures, with their relative paths rewritten for the new location.
mkpath(joinpath(SRC, "assets", "redesign"))
for f in filter(endswith(".svg"), readdir(joinpath(DESIGN, "figures")))
    cp(joinpath(DESIGN, "figures", f), joinpath(SRC, "assets", "redesign", f))
end
function place(prototype, destination)
    text = read(joinpath(DESIGN, "prototypes", prototype), String)
    depth = count(==('/'), destination)
    prefix = depth == 0 ? "assets/redesign/" : repeat("../", depth) * "assets/redesign/"
    text = replace(text, r"\((?:\.\./)+figures/([a-z-]+\.svg)\)" => SubstitutionString("($(prefix)\\1)"))
    # Links to the figure scripts and the verification record exist only in the design
    # folder (the capability prototype's note is not for publication): keep the text only.
    text = replace(text, r"\[(`[^`]+`)\]\((?:\.\./)+figures/[a-z_]+\.(?:py|jl)\)" => s"\1")
    text = replace(text, r"\[([^\]]+)\]\((?:\.\./)+verification/[^)]*\)" => s"\1")
    write(joinpath(SRC, destination), text)
end
place("index.md", "index.md")
place("quickstart.md", "quickstart.md")
place("catalog.md", "function_library.md")
place("capabilities.md", "capabilities.md")
place("guarantees.md", "spec/index.md")
place("visualizations.md", "visualizations.md")
place("functions/leastsquaresloss.md", "losses.md")
place("functions/norml1.md", "proto_norml1.md")
place("functions/groupnorml2.md", "proto_groupnorml2.md")
place("functions/indballl1.md", "constraints.md")

# The old selection guide is consolidated into the catalog. Its page and its explicit
# anchor stay, as a hidden forwarding page.
write(joinpath(SRC, "choosing.md"), "# [Choose a function or operation](@id choosing)\n\nThis page moved to the [function catalog](@ref function-catalog) and the [operations guide](@ref operation-guide).\n")
# The authoring half of the old contract index becomes a developer page with a new
# anchor; the user overview keeps the old page, its anchor and a forwarding heading.
let old = read(joinpath(ROOT, "docs", "src", "spec", "index.md"), String)
    body = replace(old, r"^# \[How to read the numerical contracts\]\(@id numerical-contracts-guide\)"m =>
        "# [How the guarantee pages are written](@id contract-pages)")
    body = replace(body, r"^## Family-page schema$"m => "## [Family-page schema](@id contract-pages-schema)")
    # The family map is replaced by the user index; the rest is authoring guidance.
    body = replace(body, r"^## Family map.*?(?=^## Status vocabulary)"ms => "")
    write(joinpath(SRC, "developer", "contract_pages.md"), body)
end

spec(name, file) = name => "spec/$file.md"
DocMeta.setdocmeta!(ProximalBase, :DocTestSetup, :(using ProximalBase, LinearAlgebra); recursive=true)
makedocs(;
    root=OUT, source="src", build="build", remotes=nothing,
    modules=[ProximalBase], sitename="ProximalBase.jl", checkdocs=:none, doctest=false,
    warnonly=[:missing_docs], linkcheck=false,
    format=Documenter.HTML(prettyurls=false, collapselevel=1, inventory_version="0.7.0-DEV", mathengine=Documenter.KaTeX(),
        size_threshold=400_000, edit_link=nothing),
    pages=[
        "Home" => "index.md",
        "Getting started" => [
            "Five-minute quick start" => "quickstart.md",
            "Tutorial: a lasso problem step by step" => "getting_started.md",
            "Tutorial: proximal gradient methods" => "proximal_gradient.md",
            "Tutorial: coordinate descent" => "coordinate_descent.md",
            "From operations to an algorithm" => "optimization_algorithms.md",
            "Exercises: proximal operations by hand" => "learn/proximal_exercises.md",
        ],
        "Functions" => [
            "Function catalog" => "function_library.md",
            "Capability tables" => "capabilities.md",
            "Losses (prototype entry, standalone)" => "losses.md",
            "Sparsity penalties (prototype entry, standalone)" => "proto_norml1.md",
            "Sparsity penalties (current page)" => "penalties.md",
            "Norm and group penalties (prototype entry, standalone)" => "proto_groupnorml2.md",
            "Norm and group penalties (current page)" => "group_penalty.md",
            "Constraints (prototype entry)" => "constraints.md",
            "smooth.md (current page, unmodified)" => "smooth.md",
            "Scaling a function" => "calculus.md",
            "Figures (prototype)" => "visualizations.md",
        ],
        "User guide" => [
            "Operations on functions" => "operations.md",
            "Function properties" => "function_objects.md",
            "Restricting a function" => "restrict.md",
            "Iterates and working sets" => "iterates.md",
            "SparseIterate" => "sparse_iterate.md",
            "BlockIterate" => "block_iterate.md",
            "SymmetricSparseIterate" => "symmetric_sparse_iterate.md",
            "Coordinate and block operations" => "iterate_oracles.md",
            "Symmetric parameters" => "symmetric_parameters.md",
            "Writing your own function" => "extending.md",
        ],
        "Examples" => [title => "examples/$name.md" for (name, title) in EXAMPLES],
        "Reference" => [
            "API reference" => ["Overview" => "api.md", "Interface" => "api/interface.md",
                "Functions" => "api/functions.md", "Scaling" => "api/calculus.md",
                "Iterates and utilities" => "api/iterates.md"],
            "Numerical guarantees" => [
                "Overview" => "spec/index.md",
                "Arrays, precision and aliasing" => "numerical_usage.md",
                "Range, non-finite values, sparse products" => "numerics.md",
                "Why floating-point range matters" => "learn/floating_point.md",
                spec("LeastSquaresLoss", "leastsquares"), spec("LogisticLoss", "logisticloss"),
                spec("PoissonLoss", "poissonloss"), spec("SqrtLeastSquaresLoss", "sqrt_leastsquares"),
                spec("LogDetLoss", "logdet"), spec("NormL1", "norml1"), spec("ElasticNet", "elasticnet"),
                spec("NormL1Nonneg", "norml1nonneg"), spec("MCP", "mcp"), spec("SCAD", "scad"),
                spec("NormL0", "norml0"), spec("NormL0, diagonal = false", "norml0offdiag"),
                spec("NormL2", "norml2"), spec("NormLinf", "normlinf"), spec("GroupNormL2", "groupnorml2"),
                spec("SparseGroupNormL2", "sparsegroupnorml2"), spec("GroupNormLinf", "groupnormlinf"),
                spec("GroupMCP", "groupmcp"), spec("GroupSCAD", "groupscad"),
                spec("IndBox and IndBallLinf", "indbox"), spec("IndNonnegative", "indnonnegative"),
                spec("IndBallL1", "indballl1"), spec("IndBallL2", "indballl2"), spec("IndBallL0", "indballl0"),
                spec("SqrNormL2", "sqrnorml2"), spec("QuadraticFunction", "quadratic"),
                spec("KronQuadratic", "kronquadratic"), spec("LinearFunction and ZeroFunction", "linear"),
                spec("Scaled", "scaled"), spec("Optimality scores", "kkt_violation"),
            ],
            "Glossary" => "glossary.md",
            "Changes from earlier versions" => "migration.md",
        ],
        "Developer guide" => [
            "Overview and package structure" => "developer/index.md",
            "System architecture" => "architecture.md",
            "SparseIterate internals" => "developer/sparse_storage.md",
            "SymmetricSparseIterate internals" => "developer/symmetric_storage.md",
            "BlockIterate internals" => "developer/block_storage.md",
            "Coordinate views and aliasing" => "developer/views_aliasing.md",
            "Function and oracle dispatch" => "developer/oracle_dispatch.md",
            "Numerical routines" => ["Overview" => "developer/routines/index.md",
                [replace(f, ".md" => "") => "developer/routines/$f" for f in sort(readdir(joinpath(SRC, "developer", "routines"))) if f != "index.md"]...],
            "How the guarantee pages are written" => "developer/contract_pages.md",
            "Design contracts" => ["Pure block proximal contract" => "developer/contracts/block-prox-0.6.md",
                "Iterate-family contract" => "developer/contracts/iterate-family-0.6.md",
                "SymmetricSparseIterate contract" => "developer/contracts/symmetric-sparse-iterate-0.6.md"],
            "API coverage matrix" => "api/coverage.md",
            "Change process" => "developer/change_process.md",
            "Writing documentation" => "developer/writing_documentation.md",
        ],
        Documenter.hide("choosing.md"), Documenter.hide("interface.md"), Documenter.hide("storage.md"),
    ],
)
println("Preview built in ", joinpath(OUT, "build"))
