# Candidate B: dense values + sparse metadata, eligible only when every group of src is active.
function _copy_active!(dest::BlockIterate{T}, src::BlockIterate) where {T}
  n = src.nactive
  if n == length(src.groups)
    # every group of src is active: every entry of dest is overwritten from src
    copyto!(dest.storage, src.storage)
    @inbounds for a in 1:n
      k = src.active[a]
      dest.active[a] = k
      dest.position[k] = a
    end
    dest.nactive = n
    return dest
  end
  @inbounds for a in 1:dest.nactive
    k = dest.active[a]
    if src.position[k] == 0
      z = zero(T)
      for i in dest.groups[k]
        dest.storage[i] = z
      end
    end
    dest.position[k] = 0
  end
  @inbounds for a in 1:n
    k = src.active[a]
    for i in src.groups[k]
      dest.storage[i] = src.storage[i]
    end
    dest.active[a] = k
    dest.position[k] = a
  end
  dest.nactive = n
  dest
end
