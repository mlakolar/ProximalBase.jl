# Candidate CD: C's all-active dense path, then D's singleton path, then master's loops.
function _copy_active!(dest::BlockIterate{T}, src::BlockIterate) where {T}
  n = src.nactive
  if n == length(src.groups)
    copyto!(dest.storage, src.storage)
    copyto!(dest.active, src.active)
    copyto!(dest.position, src.position)
    dest.nactive = n
    return dest
  end
  length(src.groups) == length(src.storage) && return _copy_singletons!(dest, src)
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

# A partition of 1:p into p groups: each group holds exactly one index.
function _copy_singletons!(dest::BlockIterate{T}, src::BlockIterate) where {T}
  @inbounds for a in 1:dest.nactive
    k = dest.active[a]
    src.position[k] == 0 && (dest.storage[first(dest.groups[k])] = zero(T))
    dest.position[k] = 0
  end
  @inbounds for a in 1:src.nactive
    k = src.active[a]
    i = first(src.groups[k])
    dest.storage[i] = src.storage[i]
    dest.active[a] = k
    dest.position[k] = a
  end
  dest.nactive = src.nactive
  dest
end

