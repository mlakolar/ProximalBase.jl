# Publishing is separate from validation: CI calls this only on master pushes
# after all test/documentation gates pass, using the inspected HTML artifact.
using Documenter
deploydocs(;
    root=@__DIR__,
    target="build",
    repo="github.com/mlakolar/ProximalBase.jl.git",
    devbranch="master",
    push_preview=false,
)
