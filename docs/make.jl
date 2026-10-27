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
    "17" => "Sparse principal components", "18" => "Total variation, isotonic regression and SLOPE")
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
            "Sparse iterates" => "iterates.md",
            "Proximal operators and interface" => "interface.md",
        ],
        "Functions" => [
            "Smooth functions" => "smooth.md",
            "Penalties" => "penalties.md",
            "Sets" => "sets.md",
            "Functions built from functions" => "calculus.md",
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
