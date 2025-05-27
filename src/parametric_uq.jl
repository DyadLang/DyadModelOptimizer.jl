"""
    parametric_uq(prob, alg; sample_size, adtype = Optimization.AutoForwardDiff(), progress = true)

Create an ensemble of the given size of possible solutions for the inverse problem using the method specified by `alg`.
This can be seen as a representation of the parametric uncertainty characterizing the given inverse problem. This is also
called a multimodel in some domains.

## Arguments

  - `prob`: an [`InverseProblem`](@ref) object.
  - `alg`: a parametric uncertainty quantification algorithm, e.g. [`StochGlobalOpt`](@ref).
    This can be [`JuliaHubJob`](@ref) for launching batch jobs in JuliaHub. In this case, the actual `alg` is wrapped inside [`JuliaHubJob`](@ref) object.

## Keyword Arguments

  - `sample_size`: Required. Number of samples to generate.
  - `adtype`: Automatic differentiation choice, see the
    [Optimization.jl docs](https://docs.sciml.ai/Optimization/stable/API/optimization_function/#Automatic-Differentiation-Construction-Choice-Recommendations)
    for details. Defaults to `AutoForwardDiff()`.
  - `progress`: Show the progress of the optimization (current loss value) in a progress bar. Defaults to `true`.

## Stochastic Optimization

If `alg` is a [`StochGlobalOpt`](@ref) method, the optimization algorithm runs until convergence
or until `maxiters` has been reached. This process is repeated a number of times equal to `sample_size`
to create an ensemble of possible parametrizations.
"""
function parametric_uq(prob, alg::StochGlobalOpt{EnsembleSerial};
        sample_size,
        adtype = Optimization.AutoForwardDiff(),
        progress_name = "parametric_uq",
        progress = true,
        result_type = ParameterEnsemble,
        x0s = initial_state(alg, prob, sample_size))
    optimizer = get_optimizer(alg)
    lb, ub = prepare_bounds(first(x0s), alg, prob)
    id = uuid4()
    Base.@logmsg ProgressLevel Progress(id, name = progress_name)  # create a progress bar

    local res, elapsed_time
    try
        elapsed_time = @elapsed res = map(Base.OneTo(sample_size)) do i
            x0 = x0s[i]
            sol = calibrate(prob, alg.method; adtype, x0, bounds = (lb, ub), progress,
                parentid = id, optimizer)
            Base.@logmsg ProgressLevel Progress(id, i / sample_size, name = progress_name)

            sol
        end
    finally
        if progress isa Bool && progress
            # Ensure that the progress bar is finished
            Base.@logmsg ProgressLevel Progress(id, done = true)
        end
    end

    result_type(res, prob, alg, elapsed_time)
end

function parametric_uq(prob, alg::StochGlobalOpt{<:EnsembleDistributed};
        sample_size,
        batch_size = determine_batch_size(EnsembleDistributed(),
            sample_size),
        adtype = Optimization.AutoForwardDiff(),
        progress_name = "parametric_uq",
        progress = true,
        result_type = ParameterEnsemble,
        x0s = initial_state(alg, prob, sample_size))
    wp = CachingPool(workers())
    optimizer = get_optimizer(alg)
    lb, ub = prepare_bounds(first(x0s), alg, prob)
    id = uuid4()
    Base.@logmsg ProgressLevel Progress(id, name = progress_name)  # create a progress bar
    channel_bufflen = min(1000, sample_size)
    channel = RemoteChannel(() -> Channel{Bool}(channel_bufflen), 1)
    i = 0

    local res, elapsed_time
    try
        # The ideea of using two tasks to display the progress was inspired by ProgressMeter.jl
        # Derived from
        # https://github.com/timholy/ProgressMeter.jl/blob/ff488567b6f337c309d09a4dbd22018dca22fecd/src/ProgressMeter.jl#L1006-L1032
        # MIT licensed, see repository for details
        # Credit goes to the ProgressMeter.jl authors
        @sync begin
            # progress updating task
            @async while take!(channel)
                i += 1
                Base.@logmsg ProgressLevel Progress(id, i / sample_size,
                    name = progress_name)
            end

            # pmap task
            @sync begin
                elapsed_time = @elapsed res = pmap(wp,
                    Base.OneTo(sample_size);
                    batch_size) do i
                    x0 = x0s[i]
                    sol = calibrate(prob,
                        alg.method;
                        adtype,
                        x0,
                        bounds = (lb, ub),
                        optimizer)
                    put!(channel, true)
                    yield()

                    sol
                end
                put!(channel, false)
            end
        end
    finally
        if progress isa Bool && progress
            # Ensure that the progress bar is finished
            Base.@logmsg ProgressLevel Progress(id, done = true)
        end
    end

    result_type(res, prob, alg, elapsed_time)
end

function parametric_uq(prob, alg::StochGlobalOpt{<:EnsembleThreads};
        sample_size,
        adtype = Optimization.AutoForwardDiff(),
        progress_name = "parametric_uq",
        progress = true,
        result_type = ParameterEnsemble,
        x0s = initial_state(alg, prob, sample_size))
    optimizer = get_optimizer(alg)
    lb, ub = prepare_bounds(first(x0s), alg, prob)
    id = uuid4()
    Base.@logmsg ProgressLevel Progress(id, name = progress_name)  # create a progress bar
    i = Threads.Atomic{Int}(1)

    local res, elapsed_time
    try
        elapsed_time = @elapsed res = tmap(Base.OneTo(sample_size)) do j
            x0 = x0s[j]
            _prob = deepcopy(prob)
            sol = calibrate(_prob, alg.method; adtype, x0, bounds = (lb, ub), optimizer)
            Base.@logmsg ProgressLevel Progress(id, i[] / sample_size,
                name = progress_name)
            Threads.atomic_add!(i, 1)

            sol
        end
    finally
        if progress isa Bool && progress
            # Ensure that the progress bar is finished
            Base.@logmsg ProgressLevel Progress(id, done = true)
        end
    end

    result_type(res, prob, alg, elapsed_time)
end

function parametric_uq(prob, alg::MCMCOpt{<:EnsembleSerial}; sample_size,
        progress_name = "parametric_uq", result_type = MCMCResult)
    elapsed_time = @elapsed chain = turing_inference(prob, alg)
    res = chain_to_vp(alg, chain, prob, sample_size)

    MCMCResult(res, prob, chain, alg, elapsed_time)
end

function parametric_uq(prob, alg::MCMCOpt{<:EnsembleDistributed}; sample_size,
        progress_name = "parametric_uq",
        result_type = MCMCResult, N_chains = nworkers())
    wp = CachingPool(workers())

    elapsed_time = @elapsed chains = pmap(wp, Base.OneTo(N_chains)) do _
        turing_inference(prob, alg)
    end
    chains = reduce(Turing.chainscat, chains)
    res = chain_to_vp(alg, chains, prob, sample_size)

    MCMCResult(res, prob, chains, alg, elapsed_time)
end

function parametric_uq(prob, alg::JuliaHubJob; sample_size,
        adtype = Optimization.AutoForwardDiff(),
        result_type = ParameterEnsemble,
        x0s = initial_state(alg.alg, prob, sample_size),
        _module = Main)
    tempdir = mktempdir(; cleanup = true)
    serialize(joinpath(tempdir, "prob"), prob)
    serialize(joinpath(tempdir, "alg"), alg.alg)
    project_packages = keys(TOML.parsefile(Base.active_project())["deps"]) |> collect
    _loaded_packages = string.(filter(
        (x) -> typeof(getfield(_module, x)) <: Module &&
            x ∉ [:Main, :Base, :Core],
        names(_module, imported = true)))
    loaded_packages = join(
        _loaded_packages[findall(in(project_packages),
            _loaded_packages)],
        ", ")
    import_statements = "using Distributed\n@everywhere using Serialization, JuliaHub, DyadModelOptimizer, Optimization, $(loaded_packages)\n"
    script = Base.remove_linenums!(quote
        auth = JuliaHub.authenticate()
        prob = deserialize(joinpath(@__DIR__, "appbundle", "prob"))
        alg = deserialize(joinpath(@__DIR__, "appbundle", "alg"))
        ps = parametric_uq(prob,
            alg;
            sample_size = $sample_size,
            adtype = $adtype)
        path = joinpath(@__DIR__, "appbundle", "ps")
        serialize(path, ps)
        dataset_name = $(alg.dataset_name)
        JuliaHub.upload_dataset(dataset_name, path; auth, update = true)
    end)
    script = join(string.(script.args), "\n")
    total_script = import_statements * script
    job = JuliaHub.submit_job(
        JuliaHub.BatchJob(JuliaHub.appbundle(tempdir,
                code = total_script),
            image = alg.batch_image);
        alg.auth, alg.node_specs...)
    return job
end

function setup_ensemble(ps, experiment; safetycopy = true)
    prob = setup_problem(experiment, ps.prob, first(ps))
    # We need to use the prob argument of the prob_func as otherwise
    # solving the problem might not be thread safe
    prob_func(prob, i, repeat) = setup_problem(experiment, ps.prob, ps[i])
    EnsembleProblem(prob; prob_func, safetycopy)
end

"""
    solve_ensemble(ps, experiment)

Create and solve an [`EnsembleProblem`](https://docs.sciml.ai/DiffEqDocs/stable/features/ensemble/#Building-a-Problem) descrbing the parameters in the ensemble `ps` and using `experiment` for the parameter
and initial condition values.

## Positional Arguments

  - `ps`: Object obtained from doing [`parametric_uq`](@ref).
  - `experiment`: [`Experiment`](@ref) defined for the Inverse Problem.
"""
function solve_ensemble(ps, experiment; saveat_reduction = x -> only(unique(x)),
        ensemblealg = EnsembleThreads(),
        safetycopy = ensemblealg isa EnsembleSerial ? false : true,
        save_idxs = get_save_idxs(experiment))
    ensembleprob = setup_ensemble(ps, experiment; safetycopy)
    kwargs = get_kwargs(experiment)
    saveats = get_saveat.((experiment,), ps, (ps.prob,))
    saveat = saveat_reduction(saveats)
    alg = get_solve_alg(experiment)
    solve(ensembleprob,
        alg,
        ensemblealg;
        trajectories = length(ps),
        save_idxs,
        kwargs...,
        saveat)
end
