function load_arm(name, source)
    arm = Module(name)
    Base.include(arm, joinpath(abspath(source), "ProximalBase.jl"))
    Base.include(arm, joinpath(@__DIR__, "enclosures.jl"))
    arm
end
