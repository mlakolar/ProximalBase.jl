# Negative control (never a candidate): unconditional dense value copy with bulk metadata.
function _copy_active!(dest::BlockIterate{T}, src::BlockIterate) where {T}
  copyto!(dest.storage, src.storage)
  copyto!(dest.active, src.active)
  copyto!(dest.position, src.position)
  dest.nactive = src.nactive
  dest
end
