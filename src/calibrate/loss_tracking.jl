function loss_progress(opt_state, loss, alg, id, last_update, start_time, parentid)
    t = time_ns()
    Δt = t - last_update[]

    if Δt > 1e9
        msg = "loss: " * sprint(show, loss, context = :compact => true)
        iter = opt_state isa OptimizationState ? opt_state.iter : nothing

        maxiters = get_maxiters(alg)
        maxtime = get_maxtime(alg)

        if !isnothing(maxtime)
            elapsed = t - start_time
            # this might be a bit inaccurate due to compile time
            # on the first call to the OptimizationFunction
            p = (elapsed / 1e9) / maxtime
        elseif !isnothing(maxiters) && !isnothing(iter)
            p = iter / maxiters
            msg *= " | iteration $iter / $maxiters"
        else
            p = exp(-(norm(loss)^(1 / 8)))
        end

        Base.@logmsg ProgressLevel Progress(id,
            p,
            name = msg,
            parentid = parentid)
        last_update[] = time_ns()
    end
end

function check_initial_objective(cost, x0)
    f0 = try
        f0 = cost(x0)
        isinf(f0) && @warn "initial objective evaluation returned ∞"
        isnan(f0) && @warn "initial objective evaluation returned NaN"
        f0
    catch e
        @error "initial objective evaluation failed" exception=e
        rethrow()
    end

    return f0
end

function loss_tracking_callback(progress::Bool, prob, alg, x0,
        parentid = ProgressLogging.ROOTID,
        cost = objective(prob, alg),
        f0 = check_initial_objective(cost, x0))
    kwargs = get_kwargs(alg)
    loss_history = Vector{typeof(f0)}(undef, 0)
    id = uuid4()
    last_update = Ref(time_ns())
    start_time = time_ns()

    callback = if haskey(kwargs, :callback)
        if progress
            Base.@logmsg ProgressLevel Progress(id, name = "calibrating")  # create a progress bar
            function (opt_state, loss, args...)
                push!(loss_history, loss)
                loss_progress(opt_state, loss, alg, id, last_update, start_time, parentid)
                kwargs[:callback](opt_state, loss, args...)
            end
        else
            function (opt_state, loss, args...)
                push!(loss_history, loss)
                kwargs[:callback](opt_state, loss, args...)
            end
        end
    else
        if progress
            Base.@logmsg ProgressLevel Progress(id, name = "calibrating")  # create a progress bar
            function (opt_state, loss, args...)
                push!(loss_history, loss)
                loss_progress(opt_state, loss, alg, id, last_update, start_time, parentid)
                false
            end
        else
            function (opt_state, loss, args...)
                push!(loss_history, loss)
                false
            end
        end
    end

    return callback, loss_history, id
end

function loss_tracking_callback(::Nothing, prob, alg, args...)
    kwargs = get_kwargs(alg)
    callback = if haskey(kwargs, :callback)
        kwargs[:callback]
    else
        nothing
    end
    loss_history = nothing

    return callback, loss_history, nothing
end
