module TestTiers
using TOML
const TIERS = ("quick", "standard", "thorough", "full")
function inventory(root=@__DIR__)
    manifest = TOML.parsefile(joinpath(root,"tiers.toml"))
    tiers = manifest["tiers"]
    Set(keys(tiers)) == Set(TIERS) || error("test manifest must declare all four tiers")
    names = reduce(vcat, (String.(tiers[t]) for t in TIERS))
    length(names) == length(unique(names)) || error("duplicate test file in manifest")
    helpers = String.(manifest["not_tests"]["files"])
    declared = Set(vcat(names,helpers))
    found = Set(relpath(joinpath(dir,f),root)[1:end-3] for (dir,_,files) in walkdir(root) for f in files if endswith(f,".jl"))
    declared == found || error("test inventory mismatch: undeclared=$(setdiff(found,declared)), missing=$(setdiff(declared,found))")
    tiers
end
function select(tiers,tier)
    stop = findfirst(==(tier),TIERS)
    stop === nothing && error("unknown test tier: $tier")
    sort!(reduce(vcat,(String.(tiers[t]) for t in TIERS[1:stop])))
end
end
