function pushbounds!(b, k, v::LowerUpperBounds)
    lb, ub = get_bounds(v)

    if lb > ub
        throw(ArgumentError("The given lowerbound ($lb) for $k is greater than the upperbound ($ub)."))
    end
    push!(b, (lb = lb, ub = ub))
end

function pushbounds!(b, k, v::Priors)
    if v isa Distributions.Sampleable
        d = v
    elseif last(v) isa Symbol
        if length(v) == 2
            d = v[1]
        else
            d = v[2]
        end
    else
        d = v[2]
    end

    push!(b, d)
end

boundstype(::InternalStorage{B}) where {B} = B
boundstype(a::AbstractDesignAnalysis) = boundstype(get_internal_storage(a))

get_bounds(ist::InternalStorage) = ist.bounds
get_bounds(a::AbstractDesignAnalysis) = get_bounds(get_internal_storage(a))
get_bounds(v::Tuple{<:VectorOrScalar, <:VectorOrScalar}) = promote(v[1], v[2])
get_bounds(v::Tuple{<:VectorOrScalar, <:VectorOrScalar, <:Symbol}) = promote(v[1], v[2])
function get_bounds(v::Tuple{<:VectorOrScalar, <:VectorOrScalar, <:VectorOrScalar})
    promote(v[2], v[3])
end
function get_bounds(v::Tuple{
        <:VectorOrScalar, <:VectorOrScalar, <:VectorOrScalar, <:Symbol})
    promote(v[2], v[3])
end
get_bounds(v::Distributions.Sampleable) = v
get_bounds(v::Tuple{<:Distributions.Sampleable, <:Symbol}) = v[1]

function get_bounds_eltype(args...)
    es = map(x -> eltype(eltype(getbounds(x))), args)
    promote_type(es...)
end

function add_bounds!(bounds, syms)
    for s in syms
        istunable(s, false) && push!(bounds, s => getbounds(s))
    end
end

function get_bounds(model::AbstractTimeDependentSystem)
    sts = unknowns(model)
    params = parameters(model)
    elt = get_bounds_eltype(sts, params)

    bounds = Vector{Pair{Num, Tuple{elt, elt}}}()
    add_bounds!(bounds, sts)
    add_bounds!(bounds, params)

    return bounds
end

function prepare_bounds(x0, alg, prob)
    opt = get_optimizer(alg)
    if allowsbounds(opt) && finite_bounds(prob)
        lb = lowerbound(prob)
        ub = upperbound(prob)

        extend_bounds!(lb, ub, x0)
    else
        lb, ub = nothing, nothing
    end
    lb, ub
end

function similar_bound(prob_b, ::Type{T}) where {T}
    B = mapreduce(typeof, promote_type, prob_b)
    if !isconcretetype(B)
        @debug "Failed to obtain concrete type for bound. Got $T."
    end
    lb = similar(prob_b, promote_type(B, T))
    copyto!(lb, prob_b)
end

function lowerbound(ist::InternalStorage)
    # copy in case this will be extended
    lowerbound(copy(ist.bounds))
    # @debug "lb type: $T"
    # lb = similar_bound(prob_lb, T)
end

function upperbound(ist::InternalStorage)
    upperbound(copy(ist.bounds))
    # ub = similar_bound(prob_ub, T)
end

"""
    lowerbound(prob)

Get the lower bound of the search space defined in the inverse problem.
"""
lowerbound(prob, args...) = lowerbound(get_internal_storage(prob), args...)

"""
    upperbound(prob)

Get the upper bound of the search space defined in the inverse problem.
"""
upperbound(prob, args...) = upperbound(get_internal_storage(prob), args...)

function lowerbound(b::Vector)
    r = [first(bi) for bi in b]
    isempty(r) ? [-Inf] : r
end

function upperbound(b::Vector)
    r = [last(bi[2]) for bi in b]
    isempty(r) ? [Inf] : r
end
