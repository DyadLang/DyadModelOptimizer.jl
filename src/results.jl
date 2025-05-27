"""
    CalibrationResult

This structure contains the results obtained from calling [`calibrate`](@ref).
The struct behaves like an `AbstractVector`, but it also provides a [`Tables.jl` interface](https://tables.juliadata.org/stable/#Using-the-Interface-(i.e.-consuming-Tables.jl-compatible-sources)).
The result is considered a table with a single row, the column names being the names in the search space.
For example if the `search_space` contains something like `[a => (0, 1.), (b => (0, 2.))]`, you can get the tuned value for `a`
with `result[:a]`, where `result` is the corresponding `CalibrationResult`.

One consequence for the Tables.jl interface is that one can easily create a `DataFrame` out of the result with just
`DataFrame(results)`, or can write it to as `.csv` file with `CSV.write(io, result)`.
To create a vector of pairs, `Tables.columnnames(results) .=> results` can be used.

A convergence plot showing how the value of the loss function evolved during calibration can be obtained using `convergenceplot(result)`.

Note that the order of the columns in the result is not necessarily the order that was provided in the `search_space` argument of
the inverse problem depending on internal optimizations. The recommended way of obtaining the order of the columns is via the Tables.jl
interface, `Tables.columnnames(results)`.
"""
struct CalibrationResult{U, C, R, O, A, L, E} <: AbstractVector{eltype(U)}
    u::U
    prob::C
    retcode::R
    original::O
    alg::A
    loss_history::L
    elapsed::E
end

Base.length(c::CalibrationResult) = length(c.u)
Base.eltype(::CalibrationResult{U}) where {U} = eltype(U)

Base.firstindex(res::CalibrationResult) = firstindex(res.u)
Base.lastindex(res::CalibrationResult) = lastindex(res.u)

Base.size(res::CalibrationResult, dim...) = size(res.u, dim...)

Base.LinearIndices(res::CalibrationResult) = LinearIndices(res.u)
Base.IndexStyle(::Type{<:CalibrationResult}) = Base.IndexLinear()

Base.@propagate_inbounds Base.getindex(res::CalibrationResult, i::Int) = getindex(res.u, i)
Base.getindex(res::CalibrationResult, i::Symbol) = Tables.getcolumn(res, i)

Tables.istable(::Type{<:CalibrationResult}) = true
Tables.rowaccess(::Type{<:CalibrationResult}) = true
Tables.rows(res::CalibrationResult) = [Tables.Row(res)]

Tables.columnnames(row::CalibrationResult) = collect(Symbol.(search_space_names(row.prob)))
Tables.getcolumn(row::CalibrationResult, i::Int) = row[i]

function Tables.getcolumn(row::CalibrationResult, nm::Symbol)
    i = indexof(nm, Tables.columnnames(row))
    row[i]
end

abstract type AbstractParametricUncertaintyEnsemble{T} <: AbstractVector{eltype(T)} end

Base.eltype(::AbstractParametricUncertaintyEnsemble{T}) where {T} = eltype(T)
Base.length(res::AbstractParametricUncertaintyEnsemble) = length(res.u)

Base.firstindex(res::AbstractParametricUncertaintyEnsemble) = firstindex(res.u)
Base.lastindex(res::AbstractParametricUncertaintyEnsemble) = lastindex(res.u)

Base.size(res::AbstractParametricUncertaintyEnsemble, dim...) = size(res.u, dim...)

Base.LinearIndices(res::AbstractParametricUncertaintyEnsemble) = LinearIndices(res.u)
Base.IndexStyle(::Type{<:AbstractParametricUncertaintyEnsemble}) = Base.IndexLinear()

Base.@propagate_inbounds function Base.getindex(res::AbstractParametricUncertaintyEnsemble,
        i::Int)
    getindex(res.u, i)
end

Base.@propagate_inbounds function Base.getindex(res::AbstractParametricUncertaintyEnsemble,
        i::AbstractVector{<:Int})
    remake(res, u = getindex(res.u, i))
end

Tables.isrowtable(::Type{<:AbstractParametricUncertaintyEnsemble}) = true

function Tables.columnnames(res::AbstractParametricUncertaintyEnsemble)
    collect(Symbol.(search_space_names(res.prob)))
end

"""
    ParameterEnsemble

The structure contains the result of a parameter ensemble when a [`StochGlobalOpt`](@ref) method
was used to generate the population. To export results to a `DataFrame` use `DataFrame(ps)`
and to plot them use `plot(ps, experiment)`, where `ps` is the `ParameterEnsemble` and `experiment` is
the `AbstractExperiment`, whose default parameter (or initial condition) values will be used.
"""
Base.@kwdef struct ParameterEnsemble{U, P, A, E} <: AbstractParametricUncertaintyEnsemble{U}
    u::U
    prob::P
    alg::A
    elapsed::E
end

"""
    MCMCResult

The structure contains the result of a parameter ensemble when an [`MCMCOpt`](@ref) method
was used to generate the population. To export results to a `DataFrame` use `DataFrame(ps)`
and to plot them use `plot(ps, experiment)`, where `ps` is the `VpopResult` and `experiment` is
the `AbstractExperiment`, whose default parameter (or initial condition) values will be used.

The original Markov chain object that the MCMC method generated can be accessed using `ps.original`.
This object contains summary statistics for each parameter and diagnostics of the MCMC method
(e.g. Rhat and Effective Sample Size). See [the MCMCChains.jl documentation](https://turinglang.org/MCMCChains.jl/stable/diagnostics/)
for more information about diagnostic checks that can be run on `ps.original`.
"""
Base.@kwdef struct MCMCResult{U, P, O, A, E} <: AbstractParametricUncertaintyEnsemble{U}
    u::U
    prob::P
    original::O
    alg::A
    elapsed::E
end
