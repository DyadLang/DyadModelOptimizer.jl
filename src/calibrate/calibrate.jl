"""
    calibrate(prob, alg; adtype = Optimization.AutoForwardDiff(), progress = true)

Find the best parameters to solve the inverse problem `prob` using the calibration algorithm given by `alg`.

## Arguments

  - `prob`: the [`InverseProblem`](@ref) to solve.
  - `alg`: the calibration algorithm used for building the loss function.
    This can be [`JuliaHubJob`](@ref) for launching batch jobs in JuliaHub. In this case, the actual `alg` is wrapped inside [`JuliaHubJob`](@ref) object.

## Keyword Arguments

  - `adtype`: Automatic differentiation choice, see the
    [Optimization.jl docs](https://docs.sciml.ai/Optimization/stable/API/optimization_function/#Automatic-Differentiation-Construction-Choice-Recommendations)
    for details. Defaults to `AutoForwardDiff()`.
  - `progress`: Show the progress of the optimization (current loss value) in a progress bar. Defaults to `true`.
"""
function calibrate(prob::AbstractInverseProblem, alg::AbstractCalibrationAlgorithm;
        adtype = Optimization.AutoForwardDiff(),
        x0 = initial_state(alg, prob),
        bounds = prepare_bounds(x0, alg, prob),
        progress = true,
        parentid = ProgressLogging.ROOTID,
        optimizer = get_optimizer(alg))
    op = OptimizationProblem(prob, alg, x0, adtype, bounds...)
    kwargs = get_kwargs(alg)
    callback, loss_history, id = loss_tracking_callback(progress,
        prob,
        alg,
        x0,
        parentid,
        op.f.f)

    local sol, elapsed_time
    try
        _kwargs = filter(k -> k[1] != :callback, pairs(kwargs))
        sol, elapsed_time = @timed solve(op, optimizer;
            maxiters = alg.optimizer isa MOI_OPTIMIZER_TYPE ? nothing :
                       get_maxiters(alg),
            maxtime = get_maxtime(alg),
            callback, _kwargs...)
    finally
        if progress isa Bool && progress
            # Ensure that the progress bar is finished
            Base.@logmsg ProgressLevel Progress(id, done = true)
        end
    end

    ist = get_internal_storage(prob)
    # @debug "sol before inverse transformation: $sol"
    u = apply_ss_transform(:inverse, visible_params_part(sol, ist), prob)

    CalibrationResult(u, prob, sol.retcode, sol, alg, loss_history, elapsed_time)
end

"""
    calibrate(res, alg; adtype = Optimization.AutoForwardDiff(), progress = true)

Continue calibration from `res` which is obtained from a previous calibration using the algorithm given by `alg`.

## Arguments

  - `res`: object of type [`CalibrationResult`](@ref) which is obtained from a previous calibration of an [`InverseProblem`](@ref).
  - `alg`: the calibration algorithm used for building the loss function.
    This can be [`JuliaHubJob`](@ref) for launching batch jobs in JuliaHub. In this case, the actual `alg` is wrapped inside [`JuliaHubJob`](@ref) object.

## Keyword Arguments

  - `adtype`: Automatic differentiation choice, see the
    [Optimization.jl docs](https://docs.sciml.ai/Optimization/stable/API/optimization_function/#Automatic-Differentiation-Construction-Choice-Recommendations)
    for details. Defaults to `AutoForwardDiff()`.
  - `progress`: Show the progress of the optimization (current loss value) in a progress bar. Defaults to `true`.
"""
function calibrate(res::CalibrationResult, alg::AbstractCalibrationAlgorithm = res.alg;
        progress = true,
        parentid = ProgressLogging.ROOTID,
        adtype = Optimization.AutoForwardDiff())
    prob = res.prob
    # if the result is imported we have a vector directly
    x0 = res.original isa SciMLBase.OptimizationSolution ? res.original.u : res.original
    # res.original has the parameters already transformed
    internal_alg_ps = internal_alg_params(alg, prob)
    if !isnothing(internal_alg_ps)
        ist = get_internal_storage(prob)
        # make sure that the internal alg params are present
        if length(internal_alg_params_part(x0, ist)) < length(internal_alg_ps)
            append!(x0, internal_alg_ps)
        end
    end
    lb, ub = prepare_bounds(x0, alg, prob)

    calibrate(prob, alg; adtype, x0, bounds = (lb, ub), progress, parentid)
end

# fallback for imported results
function calibrate(::CalibrationResult, ::Missing; kwargs...)
    error("Please specify the calibration algorithm.")
end

function calibrate(prob, alg::JuliaHubJob;
        adtype = Optimization.AutoForwardDiff(),
        progress = true,
        x0 = initial_state(alg.alg, prob),
        bounds = prepare_bounds(x0, alg.alg, prob),
        optimizer = get_optimizer(alg.alg),
        _module = Main)
    tempdir = mktempdir(; cleanup = true)
    serialize(joinpath(tempdir, "prob"), prob)
    serialize(joinpath(tempdir, "alg"), alg.alg)
    loaded_packages = get_loaded_packages(_module)
    import_statements = "using Distributed\n@everywhere using Serialization, JuliaHub, DyadModelOptimizer, Optimization, $(loaded_packages)\n"
    script = Base.remove_linenums!(quote
        auth = JuliaHub.authenticate()
        prob = deserialize(joinpath(@__DIR__, "appbundle", "prob"))
        alg = deserialize(joinpath(@__DIR__, "appbundle", "alg"))
        r = calibrate(prob, alg; adtype = $adtype)
        path = joinpath(@__DIR__, "appbundle", "r")
        serialize(path, r)
        dataset_name = $(alg.dataset_name)
        JuliaHub.upload_dataset(dataset_name, path; auth, update = true)
    end)
    script = join(string.(script.args), "\n")
    total_script = import_statements * script
    job = JuliaHub.submit_job(
        JuliaHub.BatchJob(JuliaHub.appbundle(tempdir,
                code = total_script),
            image = alg.batch_image);
        alg.auth,
        alg.node_specs...)
    return job
end

function calibrate(res::CalibrationResult, alg::JuliaHubJob;
        progress = true,
        adtype = Optimization.AutoForwardDiff(),
        _module = Main)
    tempdir = mktempdir(; cleanup = true)
    serialize(joinpath(tempdir, "res"), res)
    serialize(joinpath(tempdir, "alg"), alg.alg)
    loaded_packages = get_loaded_packages(_module)
    import_statements = "using Distributed\n@everywhere using Serialization, JuliaHub, DyadModelOptimizer, Optimization, $(loaded_packages)\n"
    script = Base.remove_linenums!(quote
        auth = JuliaHub.authenticate()
        res = deserialize(joinpath(@__DIR__, "appbundle", "res"))
        alg = deserialize(joinpath(@__DIR__, "appbundle", "alg"))
        r = calibrate(res, alg, adtype = $adtype)
        path = joinpath(@__DIR__, "appbundle", "r")
        serialize(path, r)
        dataset_name = $(alg.dataset_name)
        JuliaHub.upload_dataset(dataset_name, path; auth, update = true)
    end)
    script = join(string.(script.args), "\n")
    total_script = import_statements * script
    job = JuliaHub.submit_job(
        JuliaHub.BatchJob(JuliaHub.appbundle(tempdir,
                code = total_script),
            image = alg.batch_image);
        alg.auth,
        alg.node_specs...)
    return job
end

function SciMLBase.OptimizationProblem(prob::AbstractInverseProblem,
        alg::AbstractCalibrationAlgorithm,
        x0 = initial_state(alg, prob),
        adtype = Optimization.AutoForwardDiff(),
        lb = lowerbound(prob),
        ub = upperbound(prob))
    f = objective(prob, alg)
    lcons, ucons, cons_args = build_constraints(alg, prob)
    alg_cache = initialize_cache(alg, prob)
    if !isempty(cons_args)
        of = OptimizationFunction{true}(f, adtype; cons_args...)
        return OptimizationProblem(of, x0, (prob, alg, alg_cache); lb, ub, lcons, ucons)
    else
        of = OptimizationFunction{true}(f, adtype)
        return OptimizationProblem(of, x0, (prob, alg, alg_cache); lb, ub)
    end
end

function count_saved_states(ex, ist)
    length(get_saved_model_variables(ex))
end

include("multiple_shooting.jl")
include("collocation.jl")
include("constraints.jl")
include("kernels.jl")
include("loss_tracking.jl")
include("pem/prediction_error.jl")

function cost_contribution(alg::StochGlobalOpt,
        experiment::AbstractExperiment,
        x,
        prob;
        loss_func = get_loss_func(experiment))
    cost_contribution(alg.method, experiment, x, prob; loss_func)
end

function find_index(experiment, prob)
    experiments = get_experiments(prob)
    uuid = get_uuid(experiment)
    findfirst(t -> get_uuid(t) == uuid, experiments)
end

function internal_params_part(alg::MultipleShooting, experiment, x, prob)
    ist = get_internal_storage(prob)
    experiments = get_experiments(prob)
    extra_u0s = internal_params_part(x, ist)
    # @debug extra_u0s
    idx = find_index(experiment, prob)
    ist = get_internal_storage(prob)
    start_idx = 0
    end_idx = 0
    for (i, ex) in enumerate(Iterators.take(experiments, idx))
        n_diff_vars = length(get_diff_variables(get_model(ex)))
        tspan = timespan(experiment)#, x, prob) # FIXME
        saveat = get_saveat(experiment)
        n_intervals = length(split_timespan(alg, tspan, saveat))

        # n_starts = length(split_timespan(alg, timespan(ex, x, prob)))
        if i == idx
            start_idx = end_idx + 1
        end
        # we know the starting point of the first shooting segment
        # (the initial conditions for the experiment)
        end_idx += n_diff_vars * (n_intervals - 1)
    end
    n_diff_vars = length(get_diff_variables(get_model(experiment)))
    tspan = timespan(experiment)#, x, prob) # FIXME
    saveat = get_saveat(experiment)
    n_intervals = length(split_timespan(alg, tspan, saveat))
    # n_starts = length(split_timespan(alg, timespan(experiment, x, prob)))
    # n_u0 = n_states * n_starts
    # start_idx = end_idx - n_u0 + 1

    u0s = @views extra_u0s[start_idx:end_idx]
    reshape(u0s, n_diff_vars, (n_intervals - 1))
end

function initial_state(alg, prob::AbstractDesignAnalysis)
    problem_ps = copy(search_space_defaults(prob))

    internal_ps = internal_params(alg, prob)
    # @debug "internal_ps for alg: $internal_ps"

    if isempty(internal_ps)
        problem_ps
    elseif isempty(problem_ps)
        internal_ps
    else
        vcat(problem_ps, internal_ps)
    end
end

function initial_state(alg::StochGlobalOpt, invprob::AbstractInverseProblem, n = 1)
    # If we have bounds, the initial conditions are random in the bounds
    # otherwise random on (0,1)
    opt = get_optimizer(alg.method)
    if allowsbounds(opt) && finite_bounds(invprob)
        lb = map(identity, lowerbound(invprob))
        ub = map(identity, upperbound(invprob))
        # @debug "lb before transform: $lb"
        # lowerbound automatically applies the direct transform
        lb = apply_ss_transform(:inverse, lb, invprob)
        ub = apply_ss_transform(:inverse, ub, invprob)
    else
        n_ps = length(get_search_space(invprob))
        lb = zeros(n_ps)
        ub = ones(n_ps)
    end
    # @debug lb

    if n > 1
        # QuasiMonteCarlo.sample returns a matrix, but we need a vector of vectors
        problem_ps = [collect(c)
                      for c in eachcol(QuasiMonteCarlo.sample(n,
            # QuasiMonteCarlo does a conversion to float before sampling. We are doing it here for avoid JET errors.
            float.(lb),
            float.(ub),
            LatinHypercubeSample()))]
    elseif n == 1
        problem_ps = [rand() * (ub - lb) + lb]
    else
        throw(ArgumentError("sample_size has to be > 0, got $n"))
    end
    # @debug problem_ps
    internal_ps = internal_params(alg.method, invprob)

    transformed_ps = apply_ss_transform.(:direct, problem_ps, (invprob,))
    # @debug transformed_ps

    if isempty(internal_ps)
        transformed_ps
    else
        map(p -> vcat(p, internal_ps), transformed_ps)
    end
end

function apply_ss_transform(way::Symbol, ps, prob)
    new_ps = copy(ps)
    ss_transformations = get_ss_transformations(prob)
    ist = get_internal_storage(prob)

    transform_ps!(ss_transformations, way, new_ps, ist)
end

function transform_params!(way::Symbol, x, prob)
    ist = get_internal_storage(prob)
    ps = visible_params_part(x, ist)
    ss_transformations = get_ss_transformations(prob)

    transform_ps!(ss_transformations, way, ps, ist)
end

function transform_ps!(ss_transformations, way, ps, ist)
    (; transformed_tunables, transforms) = ss_transformations
    bi = ist.bi

    transformed_idxs = variable_index.((bi,), transformed_tunables)

    for (t, i) in zip(transforms, transformed_idxs)
        @views ps[i] = getproperty(t, way)(ps[i])
    end
    ps
end

transform_ps!(::Nothing, way, ps, prob) = ps

function internal_params(alg, prob)
    ps = eltype(search_space_defaults(prob))[]

    alg_specific_params = internal_alg_params(alg, prob)
    # @debug alg_specific_params
    if !isnothing(alg_specific_params)
        append!(ps, alg_specific_params)
    end
    # @debug "internal params: $ps"

    return map(identity, ps)
end

internal_params(::Any) = nothing
internal_alg_params(::Any, ::Any) = nothing

function get_model_transformations(prob::AbstractDesignAnalysis)
    ist = get_internal_storage(prob)
    ist.global_model_transforms
end

function append_params!(ps, model_transformations)
    for mt in model_transformations
        # @debug nameof(typeof(mt))
        ip = internal_params(mt)

        if !isnothing(ip)
            p = similar(ip)
            copyto!(p, ip)
            append!(ps, p)
        end
    end
end

function extend_bounds!(lb, ub, x0)
    @assert length(lb) == length(ub)
    Δl = length(x0) - length(lb)
    if Δl > 0
        append!(ub, fill(typemax(eltype(ub)), Δl))
        append!(lb, fill(typemin(eltype(lb)), Δl))
    end

    return lb, ub
end

function get_loaded_packages(_module)
    project_packages = keys(TOML.parsefile(Base.active_project())["deps"]) |> collect
    _loaded_packages = string.(filter(
        (x) -> typeof(getfield(_module, x)) <: Module &&
            x ∉ [:Main, :Base, :Core],
        names(_module, imported = true)))
    return join(
        _loaded_packages[findall(in(project_packages),
            _loaded_packages)],
        ", ")
end
