abstract type AtomIterator end

struct OrderedIterator <: AtomIterator
  length::Int64
end

Base.start(x::OrderedIterator) = 1
Base.next(x::OrderedIterator, i) = (i, i + 1)
Base.done(x::OrderedIterator, i) = i > x.length

function reset!(x::OrderedIterator, newLength)
  x = OrderedIterator(newLength)
end


mutable struct RandomIterator <: AtomIterator
  order::Vector{Int64}
  length::Int64
end

function reset!(x::RandomIterator, newLength)
  1 <= newLength <= length(x.order) || throw(ArgumentError("1 <= newLength <= $(length(x.order)) not satisfied"))
  x.length = newLength
  @inbounds for i=1:newLength
    x.order[i] = i
  end
  @inbounds for i=1:newLength-1
    j = rand(i:newLength)
    x.order[i], x.order[j] = x.order[j], x.order[i]
  end
  x
end

Base.start(x::RandomIterator) = 1
Base.next(x::RandomIterator, i) = (x.order[i], i + 1)
Base.done(x::RandomIterator, i) = i > x.length
