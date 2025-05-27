abstract type AbstractExperimentData{T, N} <: AbstractArray{T, N} end

Base.size(a::AbstractExperimentData) = size(a.data)

const IndexUnion = Union{AbstractArray{Int}, AbstractArray{Bool}, Colon, AbstractRange}

Base.@propagate_inbounds function Base.getindex(a::T,
        i::IndexUnion) where {T <: AbstractExperimentData}
    nt = SciMLBase.struct_as_namedtuple(a)

    parameterless_type(T)(a.data[i], Base.tail(nt)...)
end

Base.@propagate_inbounds function Base.getindex(a::T,
        i::IndexUnion, j::IndexUnion) where {T <: AbstractExperimentData}
    nt = SciMLBase.struct_as_namedtuple(a)

    parameterless_type(T)(a.data[i, j], Base.tail(nt)...)
end

Base.@propagate_inbounds function Base.getindex(a::T,
        i::IndexUnion, j::Int) where {T <: AbstractExperimentData}
    nt = SciMLBase.struct_as_namedtuple(a)

    parameterless_type(T)(a.data[i, j], Base.tail(nt)...)
end

Base.@propagate_inbounds function Base.getindex(a::T,
        i::Int, j::IndexUnion) where {T <: AbstractExperimentData}
    nt = SciMLBase.struct_as_namedtuple(a)

    parameterless_type(T)(a.data[i, j], Base.tail(nt)...)
end

Base.@propagate_inbounds function Base.getindex(a::AbstractExperimentData,
        i::CartesianIndex)
    a.data[i]
end

Base.@propagate_inbounds Base.getindex(a::AbstractExperimentData, i::Vararg{Int}) = a.data[i...]

Base.IndexStyle(::Type{<:AbstractExperimentData}) = IndexLinear()

struct ReplicateData{T, N, A <: AbstractExperimentData{T, N}} <:
       AbstractExperimentData{T, N}
    data::Vector{A}
    save_names::Vector{Symbol}
end

Base.size(a::ReplicateData) = size(a.data)

struct NoData <: AbstractExperimentData{Nothing, 0} end

Base.getproperty(::NoData, s::Symbol) = error("Can't access $s as there is no data!")

hasdata(::AbstractExperimentData) = true
hasdata(::NoData) = false

function hasdata(rd::ReplicateData)
    all(hasdata, rd.data) && return true
    all(!hasdata, rd.data) && return false
    error("Some replicates have data and some don't")
end

struct TimeSeriesData{T, N, A <: AbstractArray{T, N}, TI, O, M} <:
       AbstractExperimentData{T, N}
    data::A
    time::TI
    save_names::Vector{Symbol}
    original::O

    function TimeSeriesData{M}(data::A,
            time::TI,
            save_names::Vector{Symbol},
            original::O) where
            {T, N, A <: AbstractArray{T, N}, TI <: AbstractVector, O, M}
        new{T, N, A, TI, O, M}(data, time, save_names, original)
    end

    function TimeSeriesData(data, save_names::Vector{Symbol}, column_names,
            indepvar, saveat, tspan, expect_match)
        t, matched = get_time(data, indepvar, saveat, tspan, expect_match)
        selected_data = matrix_from_table(data, column_names)
        TimeSeriesData{Val{matched}}(selected_data, t, save_names, data)
    end

    function TimeSeriesData(
            data, save_names::Symbol, column_names, indepvar, saveat, tspan, expect_match)
        t, matched = get_time(data, indepvar, saveat, tspan, expect_match)
        selected_data = matrix_from_table(data, column_names)
        TimeSeriesData{Val{matched}}(selected_data, t, [save_names], data)
    end
end

Base.@propagate_inbounds function Base.getindex(a::TD,
        i::IndexUnion) where
        {T, N, A, TI, O, M, TD <: TimeSeriesData{T, N, A, TI, O, M}}
    nt = SciMLBase.struct_as_namedtuple(a)
    new_data = a.data[i]
    new_time = a.time[i]

    TimeSeriesData{M}(new_data, new_time, Base.tail(Base.tail(nt))...)
end

Base.@propagate_inbounds function Base.getindex(a::TD,
        i::IndexUnion,
        j::IndexUnion) where
        {T, N, A, TI, O, M, TD <: TimeSeriesData{T, N, A, TI, O, M}}
    nt = SciMLBase.struct_as_namedtuple(a)
    new_data = a.data[i, j]
    new_time = a.time[j]

    TimeSeriesData{M}(new_data, new_time, Base.tail(Base.tail(nt))...)
end

Base.@propagate_inbounds function Base.getindex(a::TD,
        i::IndexUnion,
        j::Int) where
        {T, N, A, TI, O, M, TD <: TimeSeriesData{T, N, A, TI, O, M}}
    # j is the time index, so if it's Int, we don't have multiple times,
    # so we don't have a time series
    a.data[i, j]
end

Base.@propagate_inbounds function Base.getindex(a::TD,
        i::Int,
        j::IndexUnion) where
        {T, N, A, TI, O, M, TD <: TimeSeriesData{T, N, A, TI, O, M}}
    nt = SciMLBase.struct_as_namedtuple(a)
    new_data = a.data[i, j]
    new_time = a.time[j]

    TimeSeriesData{M}(new_data, new_time, Base.tail(Base.tail(nt))...)
end

Base.@propagate_inbounds function Base.getindex(a::TimeSeriesData,
        i::CartesianIndex)
    a.data[i]
end

function get_time(data, indepvar, saveat, tspan, expect_match = true)
    # Get by column
    t = Tables.getcolumn(data, indepvar)
    @debug "Experiment tspan: $tspan"
    # @debug t
    if issorted(t) || issorted(t, rev = true)
        matched = isequal(t, saveat)
        @debug "saveat matches time column: $matched"
        if expect_match && !matched && !isequal(last(tspan), t[end])
            @debug "tspan and saveat don't match the time in the data and we expected a match, " *
                   "assuming the time column is ignored."
            # assuming MatrixLike
            return (saveat, true)
        end

        (t, matched)
    else
        error("The first column in the data is not sorted. Are you sure this represents the time?")
    end
end

# function check_time(rd::ReplicateData)  # I don't think we'll ever need this.
#     map(d -> check_time(d.data), rd)
#     nothing
# end

function matrix_from_table(data, save_names)
    if length(save_names) == 1
        Tables.getcolumn(data, only(save_names))
    else
        # we store the data transposed
        sch = Tables.schema(data)
        if isnothing(sch)
            # TODO: handle not having the schema
            error("Input schema not supported yet.")
        end
        m = matrix_from_schema(sch, data, save_names)

        i = 1
        for name in save_names
            startswith(string(name), "__sigma_") && !in(name, sch.names) && continue
            m[i, :] .= Tables.getcolumn(data, name)
            i += 1
        end

        m
    end
end

function matrix_from_schema(schema, data, save_names)
    types = schema.types
    names = schema.names
    function filterfun(name)
        startswith(string(name), "__sigma_") && return false
        true
    end
    sn = filter(filterfun, save_names)

    if isempty(sn)
        error("Couldn't select any columns from the data!\nsave_names: $save_names\nschema names: $names")
    end

    idx = indexin(sn, collect(names))
    not_found = findall(isnothing, idx)

    if !isempty(not_found)
        if length(not_found) == length(save_names)
            error("None of $(join(sn[not_found], ", ")) have corresponding column names in the provided data.")
        else
            @warn "save_names $(join(sn[not_found], ", ")) have no corresponding column names in the provided data."
        end
    end

    T = reduce(promote_type, types[idx])
    first_col = Tables.getcolumn(data, Symbol(first(sn)))
    Matrix{T}(undef, length(sn), length(first_col))
end

function data_as_dict_warn(::Dict)
    @warn "Passing the data as a `Dict` might cause unexpected results" *
          "as the order of the keys is not well defined.\n" *
          "Use an ordered dict or a `NamedTuple` of `AbstractVector`s instead."
end

data_as_dict_warn(::Any) = nothing

struct SteadyStateData{T <: Number, A <: AbstractVector{T}, O} <:
       AbstractExperimentData{T, 1}
    data::A
    save_names::Vector{Symbol}
    original::O
end

function SteadyStateData(data, save_names::Symbol)
    selected_data = Tables.getcolumn(data, save_names)
    SteadyStateData(selected_data, [save_names], data)
end

function SteadyStateData(data, save_names::Union{Vector{Symbol}, Tuple{Vararg{Symbol}}})
    selected_data = Tables.getcolumn.((data,), save_names)
    SteadyStateData(reduce(vcat, selected_data), collect(save_names), data)
end

SteadyStateData(data, ::Nothing) = SteadyStateData(data, Symbol[], data)
SteadyStateData(data::AbstractVector, ::Nothing) = SteadyStateData(data, Symbol[], data)

struct BoundsData{T, N, A <: AbstractArray{NTuple{2, T}, N}, TI, O} <:
       AbstractExperimentData{T, N}
    data::A
    time::TI
    save_names::Vector{Symbol}
    original::O
end

# assuming symbolic like save_names vector
function BoundsData(
        data, save_names::Vector, column_names, indepvar, saveat, tspan, expect_match)
    BoundsData(
        data, Symbol.(save_names), column_names, indepvar, saveat, tspan, expect_match)
end

function BoundsData(data, save_names::Vector{Symbol}, column_names,
        indepvar, saveat, tspan, expect_match)
    t, matched = get_time(data, indepvar, saveat, tspan, expect_match)
    selected_data = matrix_from_table(data, column_names)
    BoundsData(selected_data, t, save_names, data)
end

struct MismatchedMatrixLike{T} end
struct MatrixLike{T} end
struct VectorLike{T} end

function data_shape(::Type{
        <:TimeSeriesData{T, N, A, O, TI, M},
}) where {T, N, A, O, TI,
        M <: Val{false}}
    MismatchedMatrixLike{T}()
end
function data_shape(::Type{
        <:TimeSeriesData{T, N, A, O, TI, M},
}) where {T, N, A, O, TI,
        M <: Val{true}}
    MatrixLike{T}()
end
data_shape(::Type{<:SubArray{T, 2, <:TimeSeriesData{T, 2}}}) where {T} = MatrixLike{T}()
data_shape(::Type{<:AbstractArray{T, 2}}) where {T} = MatrixLike{T}()
data_shape(::Type{<:DataFrame}) = MatrixLike{Any}()
data_shape(::Type{<:NamedTuple}) = MatrixLike{Any}()
function data_shape(::Type{
        <:NamedTuple{N, Tuple{C, Vararg{C}}},
}) where {N, T, C <: AbstractVector{T}}
    MatrixLike{T}()
end

data_shape(::Type{<:TimeSeriesData{T, 1}}) where {T} = VectorLike{T}()
data_shape(::Type{<:SubArray{T, 1, <:TimeSeriesData{T, 1}}}) where {T} = VectorLike{T}()
data_shape(::Type{<:SteadyStateData{T}}) where {T} = VectorLike{T}()
data_shape(::Type{<:AbstractArray{T, 1}}) where {T} = VectorLike{T}()
data_shape(::Type{<:DataFrameRow}) = VectorLike{Any}()

function data_shape(::Type{
        <:ReplicateData{T, N, A},
}) where
        {T, N, A <: AbstractExperimentData{T, N}}
    VectorLike{A}()
end

data_shape(::Type{<:Nothing}) = nothing
