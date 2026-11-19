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
    (; tiers, weights=validated_weights(manifest, names))
end

function validated_weights(manifest, names)
    weights = get(manifest, "weights", nothing)
    weights isa AbstractDict || error("test manifest must declare a [weights] table")
    missing_names = setdiff(names, keys(weights))
    extra_names = setdiff(collect(keys(weights)), names)
    isempty(missing_names) && isempty(extra_names) ||
        error("test weight mismatch: missing=$(sort(missing_names)), extra=$(sort(extra_names))")
    for (name, w) in weights
        w isa Real && !(w isa Bool) && isfinite(w) && w > 0 ||
            error("test weight for $name must be a finite positive number")
    end
    weights
end

function select(manifest,tier)
    stop = findfirst(==(tier),TIERS)
    stop === nothing && error("unknown test tier: $tier")
    sort!(reduce(vcat,(String.(manifest.tiers[t]) for t in TIERS[1:stop])))
end

# Partition the selected files deterministically and without duplication.
# Files are placed heaviest first (ties by name) onto the shard with the smallest
# cumulative weight (ties to the lowest index); loads are exact rationals so that
# neither overflow nor rounding can alter a placement. The requested shard keeps
# the original order of `names`. Each shard still validates the complete manifest.
function shard_files(names::Vector{String}, index::Int, count::Int; weights)
    count >= 1 || error("shard count must be positive")
    1 <= index <= count || error("shard index must lie in 1:$count")
    length(names) >= count || error("more shards than selected test files")
    selected_weights = [Rational{BigInt}(weights[name]) for name in names]
    loads = zeros(Rational{BigInt}, count)
    assignment = zeros(Int, length(names))
    for i in sortperm(eachindex(names); by=i -> (-selected_weights[i], names[i]))
        shard = argmin(loads)
        assignment[i] = shard
        loads[shard] += selected_weights[i]
    end
    names[assignment .== index]
end

end
