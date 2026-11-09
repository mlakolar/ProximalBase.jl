using Documenter, Literate, ProximalBase

# The examples become pages, run when the documentation is built. Each gets a
# title, and a section marker "# --- title ---" becomes a heading.
const EXAMPLES = joinpath(@__DIR__, "..", "examples")
const GENERATED = joinpath(@__DIR__, "src", "examples")
const TITLES = Dict(
    "01" => "The lasso three ways", "02" => "A lasso path by coordinate descent",
    "03" => "Nonconvex penalties", "04" => "The group lasso", "05" => "Sparse logistic regression",
    "06" => "Robust and count regression", "07" => "Constrained least squares",
    "08" => "The graphical lasso", "09" => "Covariance and correlation matrices",
    "10" => "Matrix completion", "11" => "Sparse plus low rank", "12" => "Multitask regression",
    "13" => "A differential network", "14" => "A working-set solver",
    "15" => "Building functions, and your own", "16" => "Constraints on separable penalties",
    "17" => "Sparse principal components", "18" => "Total variation, isotonic regression and SLOPE",
    "19" => "Weighted isotonic regression")
const SECTION = r"^# -+ (.+?) -*$"m
headings(s) = replace(s, SECTION => m -> "# ### " * uppercasefirst(match(SECTION, m)[1]) * "\n#")
rm(GENERATED; force = true, recursive = true)
mkpath(GENERATED)
example_pages = Pair{String, String}[]
for file in sort(filter(endswith(".jl"), readdir(EXAMPLES)))
    title = TITLES[file[1:2]]
    Literate.markdown(joinpath(EXAMPLES, file), GENERATED; documenter = true, credit = false,
                      preprocess = s -> "# # $title\n#\n" * headings(s))
    push!(example_pages, title => joinpath("examples", replace(file, r"\.jl$" => ".md")))
end

# Publish the authoritative design contracts without duplicating their source.
# The originals remain under docs/design/; these staged Markdown files are
# generated for Documenter's navigation, just as the example pages above are.
const CONTRACT_SOURCE = joinpath(@__DIR__, "design")
const CONTRACT_TARGET = joinpath(@__DIR__, "src", "developer", "contracts")
const ITERATE_CONTRACTS = (
    "iterate-family-0.6.md",
    "iterate-family-0.6-acceptance.md",
    "symmetric-sparse-iterate-0.6.md",
    "symmetric-sparse-iterate-0.6-acceptance.md",
)
rm(CONTRACT_TARGET; force = true, recursive = true)
mkpath(CONTRACT_TARGET)
for file in ITERATE_CONTRACTS
    source = read(joinpath(CONTRACT_SOURCE, file), String)
    # One owner-contract link was written relative to docs/design/. Rebase it
    # to the generated docs/src/developer/contracts/ page location.
    source = replace(source, "../src/numerics.md" => "../../numerics.md")
    write(joinpath(CONTRACT_TARGET, file), source)
end

# Publish the source-level inventory beside the contracts. It is an audit,
# not a normative contract, and keeps its authoritative source in docs/audit/.
const AUDIT_SOURCE = joinpath(@__DIR__, "audit", "sparse-iterate-06-method-inventory.md")
const AUDIT_TARGET = joinpath(@__DIR__, "src", "developer", "sparse-iterate-method-inventory.md")
write(AUDIT_TARGET, read(AUDIT_SOURCE, String))

makedocs(
    sitename = "ProximalBase.jl",
    authors = "Mladen Kolar",
    modules = [ProximalBase],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://mlakolar.github.io/ProximalBase.jl/stable/",
        mathengine = Documenter.KaTeX(),
        size_threshold = 400_000,
    ),
    pages = [
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "Package structure" => "architecture.md",
        "Algorithms and interface" => [
            "Coordinate descent" => "coordinate_descent.md",
            "SparseIterate: storage and working sets" => "sparse_iterate.md",
            "SymmetricSparseIterate: coordinate owner" => "symmetric_sparse_iterate.md",
            "Sparse iterates and coordinate oracles" => "iterates.md",
            "Proximal operators and interface" => "interface.md",
            "Numerical contract" => "numerics.md",
        ],
        "Functions" => [
            "Smooth functions" => "smooth.md",
            "Penalties" => "penalties.md",
            "Sets" => "sets.md",
            "Functions built from functions" => "calculus.md",
        ],
        "Developer documentation" => [
            "Overview and contract index" => "developer/index.md",
            "Coordinate views and aliasing" => "developer/coordinate_views.md",
            "SparseIterate method inventory" => "developer/sparse-iterate-method-inventory.md",
            "Iterate-family 0.6 contract" => "developer/contracts/iterate-family-0.6.md",
            "Iterate-family acceptance" => "developer/contracts/iterate-family-0.6-acceptance.md",
            "Symmetric owner 0.6 contract" => "developer/contracts/symmetric-sparse-iterate-0.6.md",
            "Symmetric owner acceptance" => "developer/contracts/symmetric-sparse-iterate-0.6-acceptance.md",
        ],
        "Writing your own function" => "extending.md",
        "Compatibility and migration" => "migration.md",
        "Examples" => example_pages,
        "API reference" => [
            "The interface" => "api/interface.md",
            "Functions" => "api/functions.md",
            "Functions built from functions" => "api/calculus.md",
            "Iterates and utilities" => "api/iterates.md",
        ],
    ],
    checkdocs = :exports,
    linkcheck = get(ENV, "DOCUMENTER_LINKCHECK", "false") == "true",
)

deploydocs(repo = "github.com/mlakolar/ProximalBase.jl.git", devbranch = "master", push_preview = false)
