# Prepare a fresh environment for a package revision using this benchmark project.
using Pkg, TOML
length(ARGS) == 2 || error("usage: julia benchmark/setup.jl PACKAGE_CHECKOUT NEW_ENVIRONMENT")
source = realpath(ARGS[1])
environment = abspath(ARGS[2])
project = TOML.parsefile(joinpath(@__DIR__, "Project.toml"))
package = TOML.parsefile(joinpath(source, "Project.toml"))
package["name"] == "ProximalBase" && package["uuid"] == project["deps"]["ProximalBase"] ||
    error("Expected a ProximalBase checkout")
isfile(joinpath(environment, "Project.toml")) && error("Use a fresh environment directory")
mkpath(environment)
project["sources"]["ProximalBase"]["path"] = source
open(joinpath(environment, "Project.toml"), "w") do io
    TOML.print(io, project)
end
Pkg.activate(environment)
Pkg.develop(path=source)
Pkg.instantiate()
