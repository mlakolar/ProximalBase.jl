# Each checkout retains the original module name in its own namespace. This
# also preserves qualified internal names and docstrings during inclusion.
function load_checkout(checkout, namespace)
    container = Module(namespace)
    Base.include(container, abspath(checkout, "src", "ProximalBase.jl"))
    return container.ProximalBase
end

function penalty(pb, level, factors)
    f = pb.NormLinf(level)
    for factor in factors
        f = pb.Scaled(f, factor)
    end
    return f
end

tsv(io, fields...) = println(io, join(fields, '\t'))
