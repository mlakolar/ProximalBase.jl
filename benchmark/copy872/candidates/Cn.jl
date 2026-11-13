# Candidate Cn: C with the all-active bulk copies in a separate @noinline function; the
# loops for partly active sources are master's, unchanged.
function _copy_active!(dest::BlockIterate{T}, src::BlockIterate) where {T}
  src.nactive == length(src.groups) && return _copy_all_active!(dest, src)
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
  @inbounds for a in 1:src.nactive
    k = src.active[a]
    for i in src.groups[k]
      dest.storage[i] = src.storage[i]
    end
    dest.active[a] = k
    dest.position[k] = a
  end
  dest.nactive = src.nactive
  dest
end

@noinline function _copy_all_active!(dest::BlockIterate, src::BlockIterate)
  copyto!(dest.storage, src.storage)
  copyto!(dest.active, src.active)
  copyto!(dest.position, src.position)
  dest.nactive = src.nactive
  dest
end
