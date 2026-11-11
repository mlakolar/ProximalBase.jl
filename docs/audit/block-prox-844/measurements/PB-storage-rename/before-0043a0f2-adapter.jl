# Comparison-only Main bindings for immutable PB 0043a0f2; never package aliases.
using ProximalBase
const BlockIterate = ProximalBase.AtomIterate
const blockview = ProximalBase.atom
const blockindices = ProximalBase.atomindices
include(ARGS[1])
