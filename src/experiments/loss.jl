############################ squared l2 loss ############################

"""
    squaredl2loss(tuned_vals, sol, data)

Sum of squared error loss:

``\\sum_{i=1}^{M} \\sum_{j=1}^{N} \\left( \\text{sol}_{i,j} - \\text{data}_{i,j} \\right)^2``

where N is the number of saved timepoints and M the number of measured states in the solution.
"""
squaredl2loss(x, sol, data::T; kw...) where {T} = squaredl2loss(data_shape(T), sol, data)

function squaredl2loss(::VectorLike{A},
        sol,
        data) where {A <: AbstractExperimentData{T}} where {T}
    sum(squaredl2loss.(((),), (sol,), data))
end

function squaredl2loss(::MismatchedMatrixLike{T}, sol, data) where {T}
    idxs = map(x -> in(x, data.time), sol.t)
    sol_matched = @view sol[:, idxs]
    squaredl2loss(MatrixLike{T}(), sol_matched, data)
end

function squaredl2loss(::MatrixLike{T}, sol::AbstractDiffEqArray, data) where {T}
    𝟘 = zero(promote_type(eltype(sol), T))
    err = 𝟘
    @assert size(sol) == size(data)
    @inbounds for (s, d) in zip(sol.u, eachcol(data))
        for i in eachindex(s, d)
            if !ismissing(d[i])
                err += (s[i] - d[i])^2
            else
                err += 𝟘
            end
        end
    end
    return err
end

# subarrays from `MismatchedMatrixLike`
function squaredl2loss(::MatrixLike{T}, sol, data) where {T}
    𝟘 = zero(promote_type(eltype(sol), T))
    err = 𝟘
    @assert size(sol) == size(data)
    @inbounds for (s, d) in zip(eachcol(sol), eachcol(data))
        for i in eachindex(s, d)
            if !ismissing(d[i])
                err += (s[i] - d[i])^2
            else
                err += 𝟘
            end
        end
    end
    return err
end

function squaredl2loss(::VectorLike{T}, sol, data) where {T <: Number}
    idx = eachindex(skipmissing(data))
    err = zero(promote_type(eltype(sol), nonmissingtype(T)))
    @inbounds for i in idx
        # scalar solution, we only have one state
        err += (sol[i] - data[i])^2
    end
    return err
end

function squaredl2loss(::VectorLike{T},
        sol::AbstractDiffEqArray,
        data) where {T <: Number}
    idx = eachindex(skipmissing(data))
    err = zero(promote_type(eltype(sol), nonmissingtype(T)))
    @inbounds for i in idx
        err += (sol.u[i] - data[i])^2
    end
    return err
end

_2arg_squaredl2loss(sol, data) = sum((sol .- data) .^ 2)

############################ l2 loss ############################

"""
    l2loss(tuned_vals, sol, data)

Sum of l2 loss:

``\\sqrt{\\sum_{i=1}^{M} \\sum_{j=1}^{N} \\left( \\text{sol}_{i,j} - \\text{data}_{i,j} \\right)^2}``

where N is the number of saved timepoints and M the number of measured states in the solution.
"""
l2loss(x, sol, data) = sqrt(squaredl2loss(x, sol, data))

############################ mean squared l2 loss ############################

"""
    meansquaredl2loss(tuned_vals, sol, data)

Mean of squared l2 loss:

``\\frac{(\\sum_{i=1}^{M} \\sum_{j=1}^{N} \\left( \\text{sol}_{i,j} - \\text{data}_{i,j} \\right)^2)}{N}``

where N is the number of saved timepoints and M the number of measured states in the solution.
"""
meansquaredl2loss(x, sol, data::T; kw...) where {T} = meansquaredl2loss(
    data_shape(T), sol, data)

function meansquaredl2loss(::VectorLike{A},
        sol,
        data) where {A <: AbstractExperimentData{T}} where {T}
    sum(meansquaredl2loss.(((),), (sol,), data))
end

function meansquaredl2loss(::VectorLike{T}, sol, data) where {T <: Number}
    squaredl2loss((), sol, data) / size(sol, 1)
end

function meansquaredl2loss(::MatrixLike{T}, sol, data) where {T}
    𝟘 = zero(promote_type(eltype(sol), nonmissingtype(T)))
    err = 𝟘
    # Here it's important to count the number of non missing
    # in each column of the data or each of the timeseries (for each state)
    # Since we transposed TimeSeriesData, this means that we need to
    # look at the rows
    @inbounds for i in eachindex(axes(sol, 1), axes(data, 1))
        curr_err = 𝟘
        s = @view sol[i, :]
        d = data[i, :]
        non_missing = count(.!ismissing.(d))
        f = 1 / non_missing
        for j in eachindex(s, d)
            if !ismissing(d[j])
                curr_err += (s[j] - d[j])^2
            else
                curr_err += 𝟘
            end
        end
        curr_err *= f
        err += curr_err
    end
    return err
end

function meansquaredl2loss(::MismatchedMatrixLike{T}, sol, data) where {T}
    idxs = map(x -> in(x, data.time), sol.t)
    # @debug "Indices of the solution that have data: $idxs"
    sol_matched = @view sol[:, idxs]
    meansquaredl2loss(MatrixLike{T}(), sol_matched, data)
end

############################ root mean squared l2 loss ############################

"""
    root_meansquaredl2loss(sol, data)

Root of mean squared l2 loss:

``\\sqrt{\\frac{(\\sum_{i=1}^{M} \\sum_{j=1}^{N} \\left( \\text{sol}_{i,j} - \\text{data}_{i,j} \\right)^2)}{N}}``

where N is the number of saved timepoints and M the number of measured states in the solution.
"""
root_meansquaredl2loss(x, sol, data; kw...) = sqrt(meansquaredl2loss(x, sol, data))

############################ normalized mean squared l2 loss ############################

"""
    norm_meansquaredl2loss(tuned_vals, sol, data)

Normalized mean squared l2 loss:

``\\frac{(\\sum_{i=1}^{M} \\sum_{j=1}^{N} \\left( \\text{sol}_{i,j} - \\text{data}_{i,j} \\right)^2)}{(\\sum_{i=1}^{M} \\sum_{j=1}^{N} \\left( \\text{sol}_{i,j} - mean\\_sol_{i} \\right)^2}``

where N is the number of saved timepoints and M the number of measured states in the solution.
"""
norm_meansquaredl2loss(x, sol, data::T; kw...) where {T} = norm_meansquaredl2loss(
    data_shape(T),
    sol,
    data)

function norm_meansquaredl2loss(::VectorLike{A},
        sol,
        data) where {A <: AbstractExperimentData{T}} where {T}
    sum(norm_meansquaredl2loss.(((),), (sol,), data))
end

function norm_meansquaredl2loss(::VectorLike{T}, sol, data) where {T <: Number}
    idx = eachindex(skipmissing(data))
    err = zero(promote_type(eltype(sol), nonmissingtype(T)))
    non_missing = count(.!ismissing.(data))
    f = 1 / non_missing
    m = mean(sol)
    @inbounds for i in idx
        err += ((sol[i] - data[i])^2 / (sol[i] - m)^2) * f
    end
    return err
end

function norm_meansquaredl2loss(
        ::VectorLike{T}, sol::AbstractDiffEqArray, data) where {T <: Number}
    idx = eachindex(skipmissing(data))
    err = zero(promote_type(eltype(sol), nonmissingtype(T)))
    non_missing = count(.!ismissing.(data))
    f = 1 / non_missing
    m = mean(sol)
    @inbounds for i in idx
        err += ((sol.u[i] - data[i])^2 / (sol.u[i] - m)^2) * f
    end
    return err
end

function norm_meansquaredl2loss(::MatrixLike{T}, sol, data) where {T}
    𝟘 = zero(promote_type(eltype(sol), nonmissingtype(T)))
    err = 𝟘
    # Here it's important to count the number of non missing
    # in each column of the data or each of the timeseries (for each state)
    # Since we transposed TimeSeriesData, this means that we need to
    # look at the rows
    @inbounds for i in eachindex(axes(sol, 1), axes(data, 1))
        curr_err = 𝟘
        s = @view sol[i, :]
        d = @view data[i, :]
        non_missing = count(.!ismissing.(d))
        f = 1 / non_missing
        m = mean(s)
        for i in eachindex(s, d)
            if !ismissing(d[i])
                curr_err += ((s[i] - d[i])^2 / (s[i] - m)^2)
            else
                curr_err += 𝟘
            end
        end
        curr_err *= f
        err += curr_err
    end
    return err
end

function norm_meansquaredl2loss(::MismatchedMatrixLike{T}, sol, data) where {T}
    idxs = map(x -> in(x, data.time), sol.t)
    # @debug "Indices of the solution that have data: $idxs"
    sol_matched = @view sol[:, idxs]
    norm_meansquaredl2loss(MatrixLike{T}(), sol_matched, data)
end

############################ zscore mean squared l2 loss ############################

zscore_func(x, mean, std) = @. (x - mean) / std

"""
    zscore_meansquaredl2loss(tuned_vals, sol, data)

Zscore mean squared l2 loss:

``\\frac{\\sum_{i=1}^{M} \\sum_{j=1}^{N} \\left( \\text{zscore(sol)}_{i,j} - \\text{zscore(data)}_{i,j} \\right)^2}{N}``

where N is the number of saved timepoints and M the number of measured states in the solution and `zscore` is the [standard score](https://en.wikipedia.org/wiki/Standard_score).
"""
zscore_meansquaredl2loss(x, sol, data::T; kw...) where {T} = zscore_meansquaredl2loss(
    data_shape(T),
    sol,
    data; kw...)

function zscore_meansquaredl2loss(::VectorLike{A},
        sol,
        data; kw...) where {A <: AbstractExperimentData{T}} where {T}
    sum(zscore_meansquaredl2loss.(((),), (sol,), data; kw...))
end

function zscore_meansquaredl2loss(::VectorLike{T}, sol, data; data_mean = mean(data),
        data_std = std(data)) where {T <: Number}
    idx = eachindex(skipmissing(data))
    err = zero(promote_type(eltype(sol), nonmissingtype(T)))
    non_missing = count(.!ismissing.(data))
    f = 1 / non_missing
    @inbounds for i in idx
        mn = isa(data_mean, AbstractArray) ? data_mean[i] : data_mean
        std_dev = isa(data_std, AbstractArray) ? data_std[i] : data_std
        norm_s = zscore_func(sol[i], mn, std_dev)
        norm_d = zscore_func(data[i], mn, std_dev)
        err += ((norm_s - norm_d)^2) * f
    end
    return err
end

function zscore_meansquaredl2loss(
        ::VectorLike{T}, sol::AbstractDiffEqArray, data; data_mean = mean(data),
        data_std = std(data)) where {T <: Number}
    idx = eachindex(skipmissing(data))
    err = zero(promote_type(eltype(sol), nonmissingtype(T)))
    non_missing = count(.!ismissing.(data))
    f = 1 / non_missing
    m = mean(sol)
    @inbounds for i in idx
        norm_s = zscore_func(sol[i], data_mean, data_std)
        norm_d = zscore_func(data[i], data_mean, data_std)
        err += ((norm_s - norm_d)^2) * f
    end
    return err
end

function zscore_meansquaredl2loss(
        ::MatrixLike{T}, sol, data; data_mean = mean(data, dims = 2),
        data_std = std(data, dims = 2)) where {T}
    𝟘 = zero(promote_type(eltype(sol), nonmissingtype(T)))
    err = 𝟘
    # Here it's important to count the number of non missing
    # in each column of the data or each of the timeseries (for each state)
    # Since we transposed TimeSeriesData, this means that we need to
    # look at the rows
    @inbounds for i in eachindex(axes(sol, 1), axes(data, 1))
        curr_err = 𝟘
        s = @view sol[i, :]
        d = @view data[i, :]
        non_missing = count(.!ismissing.(d))
        f = 1 / non_missing
        for j in eachindex(s, d)
            if !ismissing(d[i])
                norm_s = zscore_func(s[j], data_mean[i], data_std[i])
                norm_d = zscore_func(d[j], data_mean[i], data_std[i])
                curr_err += ((norm_s - norm_d)^2)
            else
                curr_err += 𝟘
            end
        end

        curr_err *= f
        err += curr_err
    end
    return err
end

function zscore_meansquaredl2loss(::MismatchedMatrixLike{T}, sol, data; kw...) where {T}
    idxs = map(x -> in(x, data.time), sol.t)
    # @debug "Indices of the solution that have data: $idxs"
    sol_matched = @view sol[:, idxs]
    zscore_meansquaredl2loss(MatrixLike{T}(), sol_matched, data; kw...)
end

"""
    zscore_meanabsl1loss(tuned_vals, sol, data)

Normalized mean absolute L1 Z-score.

``\\frac{\\sum_{i=1}^{M} \\sum_{j=1}^{N} abs\\left( \\text{zscore(sol)}_{i,j} - \\text{zscore(data)}_{i,j} \\right)}{N}``

where N is the number of saved timepoints and M the number of measured states in the solution.
"""
zscore_meanabsl1loss(x, sol, data::T; kw...) where {T} = zscore_meanabsl1loss(
    data_shape(T),
    sol,
    data; kw...)

function zscore_meanabsl1loss(::VectorLike{A},
        sol,
        data; kw...) where {A <: AbstractExperimentData{T}} where {T}
    sum(zscore_meanabsl1loss.(((),), (sol,), data; kw...))
end

function zscore_meanabsl1loss(::VectorLike{T}, sol, data; data_mean = mean(data),
        data_std = std(data)) where {T <: Number}
    err = zero(promote_type(eltype(sol), nonmissingtype(T)))

    score_sol = zscore_func(sol, data_mean, data_std)
    score_data = zscore_func(data, data_mean, data_std)
    err += mean(skipmissing(abs.(score_sol - score_data)))
    return err
end

function zscore_meanabsl1loss(
        ::VectorLike{T}, sol::AbstractDiffEqArray, data; data_mean = mean(data, dims = 1),
        data_std = std(data, dims = 1)) where {T <: Number}
    err = zero(promote_type(eltype(sol), nonmissingtype(T)))

    score_sol = zscore_func(Array(sol), data_mean, data_std)
    score_data = zscore_func(data, data_mean, data_std)
    pred = Array(sol)
    score_sol = @. ((pred - data_mean) / data_std)
    score_data = @. ((data - data_mean) / data_std)
    err += mean(skipmissing(abs.(score_sol - score_data)))

    return err
end

function zscore_meanabsl1loss(::MatrixLike{T}, sol, data; data_mean = mean(data, dims = 2),
        data_std = std(data, dims = 2)) where {T}
    𝟘 = zero(promote_type(eltype(sol), nonmissingtype(T)))
    err = 𝟘
    # Here it's important to count the number of non missing
    # in each column of the data or each of the timeseries (for each state)
    # Since we transposed TimeSeriesData, this means that we need to
    # look at the rows
    @inbounds for i in eachindex(axes(sol, 1), axes(data, 1))
        curr_err = 𝟘
        s = @view sol[i, :]
        d = @view data[i, :]
        non_missing = count(.!ismissing.(d))
        f = 1 / non_missing
        for j in eachindex(s, d)
            if !ismissing(d[i])
                norm_s = zscore_func(s[j], data_mean[i], data_std[i])
                norm_d = zscore_func(d[j], data_mean[i], data_std[i])
                curr_err += abs(norm_s - norm_d)
            else
                curr_err += 𝟘
            end
        end

        curr_err *= f
        err += curr_err
    end
    return err
end

function zscore_meanabsl1loss(::MismatchedMatrixLike{T}, sol, data; kw...) where {T}
    idxs = map(x -> in(x, data.time), sol.t)
    # @debug "Indices of the solution that have data: $idxs"
    sol_matched = @view sol[:, idxs]
    zscore_meanabsl1loss(MatrixLike{T}(), sol_matched, data; kw...)
end

"""
    ARMLoss(sol, bounds)

Allen-Rieger-Musante (ARM) loss :

``\\sum_{i=1}^{M} \\sum_{j=1}^{N} \\text{max} \\left[ \\left( \\text{sol}_{i,j} - \\frac{\\text{u}_{i,j} + \\text{l}_{i,j}}{2} \\right)^2 - \\left( \\frac{\\text{u}_{i,j} - \\text{l}_{i,j}}{2} \\right)^2, 0 \\right]``

where N is the number of saved timepoints, M the number of measured states in the solution
and `l, u` are the lower and upper bounds of each measured state respectively.

## Reference

Allen RJ, Rieger TR, Musante CJ. Efficient Generation and Selection of Virtual Populations
in Quantitative Systems Pharmacology Models. CPT Pharmacometrics Syst Pharmacol. 2016
Mar;5(3):140-6. doi: 10.1002/psp4.12063. Epub 2016 Mar 17. PMID: 27069777; PMCID: PMC4809626.
"""
function ARMLoss(x, sol, bounds::BoundsData{T, 2}) where {T}
    𝟘 = zero(promote_type(eltype(sol), T))
    err = 𝟘
    @inbounds for (s, b) in zip(sol.u, eachcol(bounds))
        for i in eachindex(s, b)
            err += max(
                (s[i] - (b[i][1] + b[i][2]) / 2.0)^2 - (b[i][2] / 2 - b[i][1] / 2)^2,
                𝟘)
        end
    end
    return err
end

function ARMLoss(x, sol, bounds::BoundsData{T, 1}) where {T}
    𝟘 = zero(promote_type(eltype(sol), T))
    err = 𝟘
    # need to use last(sol) instead of sol, otherwise there is index mismatch: sol_size=(N,1) and bounds_size=(N,)
    # if bounds is a Vector then sol will also be a Vector, so taking last is safe
    # ARMLoss was most likely used for the final timepoint of the trajectory in the paper where it was introduced
    #ls = last(sol)
    @inbounds for i in eachindex(Base.OneTo(length(sol)), bounds)
        err += max(
            (sol[i] - (bounds[i][1] + bounds[i][2]) / 2.0)^2 -
            (bounds[i][2] / 2 - bounds[i][1] / 2)^2,
            𝟘)
    end
    return err
end

struct LossContribution{T} <: AbstractVector{T}
    diff::Vector{T}
end

function LossContribution(data)
    T = eltype(eachcol(data)[2])
    LossContribution{T}(zeros(T, size(data, 2) - 1))
end

function (lc::LossContribution)(x, sol, data)
    𝟘 = zero(promote_type(eltype(sol), eltype(data)))
    diff = lc.diff
    diff .= 𝟘
    for (i, s, d) in zip(axes(sol, 1), sol.u, eachcol(data))
        Δ = diff[i]
        for j in eachindex(s, d)
            if !ismissing(d[j])
                Δ += s[j] - d[j]
            else
                Δ += 𝟘
            end
        end
        diff[i] = Δ
    end
    return sum(diff)
end

Base.eltype(::LossContribution{T}) where {T} = T

# Iteration
Base.length(lc::LossContribution) = length(lc.diff)
Base.iterate(lc::LossContribution, state...) = iterate(lc.diff, state...)

Base.firstindex(lc::LossContribution) = firstindex(lc.diff)
Base.lastindex(lc::LossContribution) = lastindex(lc.diff)

Base.size(lc::LossContribution, dim...) = size(lc.diff, dim...)

Base.LinearIndices(lc::LossContribution) = LinearIndices(lc.diff)
Base.IndexStyle(::Type{<:LossContribution}) = Base.IndexLinear()

Base.@propagate_inbounds Base.getindex(lc::LossContribution, i::Int) = getindex(lc.diff, i)

for func in [
    :squaredl2loss,
    :l2loss,
    :meansquaredl2loss,
    :ARMLoss,
    :LossContribution,
    :_2arg_squaredl2loss,
    :zscore_meanabsl1loss
]
    @eval begin
        has_fast_indexing(::typeof($func)) = Val(true)
    end
end
