# Candidate C2: bulk metadata at every activity level (active prefix + whole position map);
# dense values only when every group of src is active.
function _copy_active!(dest::BlockIterate{T}, src::BlockIterate) where {T}
  n = src.nactive
  if n == length(src.groups)
    copyto!(dest.storage, src.storage)
  else
    @inbounds for a in 1:dest.nactive
      k = dest.active[a]
      if src.position[k] == 0
        z = zero(T)
        for i in dest.groups[k]
          dest.storage[i] = z
        end
      end
    end
    @inbounds for a in 1:n
      for i in src.groups[src.active[a]]
        dest.storage[i] = src.storage[i]
      end
    end
  end
  copyto!(dest.active, 1, src.active, 1, n)
  copyto!(dest.position, src.position)
  dest.nactive = n
  dest
end
