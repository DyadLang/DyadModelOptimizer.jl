using DyadModelOptimizer: get_saved_model_variables

function test_nowarn_plots(ps, prob, experiment)
    saved_vars = get_saved_model_variables(experiment)
    tf = [true, false]

    @testset "Plot recipes" begin
        @test_nowarn plot(experiment, prob)
        @testset "summary: $summary , show_data: $show_data " for summary in tf,
            show_data in tf

            @test_nowarn plot(ps, experiment; summary, show_data)
            st = rand(saved_vars)
            @debug "plotting $st"
            @test_nowarn plot(ps, experiment; states = st, summary, show_data)
            @test_nowarn plot(ps, experiment; states = [st], summary, show_data)
            if length(saved_vars) ≥ 2
                sts = collect(Iterators.reverse(collect(Iterators.take(saved_vars, 2))))
                @test_nowarn plot(ps, experiment; states = sts, summary, show_data)
            end
            @test_nowarn plot(ps, prob; summary, show_data)
        end
    end
end
