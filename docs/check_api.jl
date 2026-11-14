# Live coverage check; keep the reference explicit rather than relying on autodocs.
using ProximalBase
function check_api_coverage()
    reference = join(read.(filter(endswith(".md"), readdir(joinpath(@__DIR__, "src", "api"); join=true)), String), "\n")
    docs_names = Set{String}()
    for block in eachmatch(r"```@docs\n(.*?)```"s, reference), line in split(block.captures[1], '\n')
        name = first(split(strip(line), '('))
        isempty(name) || push!(docs_names, replace(name, "ProximalBase." => ""))
    end
    coverage = read(joinpath(@__DIR__, "src", "api", "coverage.md"), String)
    # names includes qualified `public` bindings on Julia 1.11+ as well as exports.
    required = union(Set(string.(names(ProximalBase))), Set([
        "ScalarLoss", "Squared", "deriv", "deriv2", "curvature", "value_and_deriv",
        "check_parameters", "entry_parameters", "check_symmetric_parameters", "samegroups"]))
    for name in sort!(collect(required))
        name in docs_names || error("Missing explicit grouped @docs entry: $name")
        name == "ProximalBase" || occursin(Regex("`(?:ProximalBase\\.)?" * name * "`"), coverage) ||
            error("Missing API matrix row: $name")
    end
    println("API coverage: ", length(required), " module/export/public bindings have explicit grouped references")
end
check_api_coverage()
