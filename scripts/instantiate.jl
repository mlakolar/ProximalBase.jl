# Set up this package or an environment that uses the local ProximalBase.
using Pkg, TOML

package_root = realpath(dirname(@__DIR__))
length(ARGS) <= 1 || error("usage: julia scripts/instantiate.jl [ENVIRONMENT_DIRECTORY]")
environment = isempty(ARGS) ? package_root : realpath(ARGS[1])
project_path = joinpath(environment, "Project.toml")
original_project = read(project_path, String)
project = TOML.parse(original_project)

Pkg.activate(environment)
try
    if environment != package_root && haskey(get(project, "deps", Dict()), "ProximalBase")
        # Explicitly use this checkout, including on Julia 1.10, which ignores [sources].
        Pkg.develop(Pkg.PackageSpec(path=package_root))
    end
    Pkg.instantiate()
finally
    # Pkg.develop may rewrite the project file; retain the checked-in contents.
    write(project_path, original_project)
end
