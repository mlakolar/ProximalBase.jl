using ProximalBase, Documenter
include(joinpath(@__DIR__,"..","examples","lasso.jl"))
include(joinpath(@__DIR__,"..","examples","custom_function.jl"))
makedocs(;
    modules=[ProximalBase],
    sitename="ProximalBase.jl",
    authors="Mladen Kolar and contributors",
    checkdocs=:exports,
    doctest=true,
    format=Documenter.HTML(prettyurls=get(ENV,"CI","false")=="true", edit_link="master"),
    pages=["Overview"=>"index.md","Storage"=>"storage.md","Mathematical interface"=>"interface.md",
           "Numerical contract"=>"numerics.md","Extending"=>"extending.md","API"=>"api.md"],
)
