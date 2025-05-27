const VectorOrScalar = Union{Number, AbstractVector}

const LowerUpperBounds = Union{
    Tuple{T1, T2},              # (lb, ub)
    Tuple{T1, T2, T3},          # (initial, lb, ub)
    Tuple{T1, T2, Symbol},      # (lb, ub, transform)
    Tuple{T1, T2, T3, Symbol}  # (initial, lb, ub, transform)
} where {T1 <: VectorOrScalar, T2 <: VectorOrScalar, T3 <: VectorOrScalar}

const Priors = Union{
    D,
    Tuple{D, Symbol}
} where {D <: Distributions.Sampleable}

include("internal_storage.jl")
# include("petab_override.jl")
include("bounds.jl")
