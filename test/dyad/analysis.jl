using DyadModelOptimizer
using DyadInterface
using ModelingToolkit
using ModelingToolkitStandardLibrary.Electrical
using ModelingToolkitStandardLibrary.Blocks: Sine
using ModelingToolkit: t_nounits as t
using Test
using JSON3
using Plots
using DataFrames
using DyadData

function create_model(; C₁ = 2.4, C₂ = 60.0, f = 1.0)
    # V = 10.0
    @named resistor1 = Resistor(R = 5.0)
    @named resistor2 = Resistor(R = 2.0)
    @named capacitor1 = Capacitor(C = C₁)
    @named capacitor2 = Capacitor(C = C₂)
    @named source = Voltage()
    @named input_signal = Sine(frequency = f)
    @named ground = Ground()
    @named ampermeter = CurrentSensor()

    eqs = [connect(input_signal.output, source.V)
           connect(source.p, capacitor1.n, capacitor2.n)
           connect(source.n, resistor1.p, resistor2.p, ground.g)
           connect(resistor1.n, capacitor1.p, ampermeter.n)
           connect(resistor2.n, capacitor2.p, ampermeter.p)]

    @named circuit_model = ODESystem(eqs, t,
        systems = [
            resistor1, resistor2, capacitor1, capacitor2,
            source, input_signal, ground, ampermeter
        ],
        # HACK: we can't specify overrides in the spec yet
        initialization_eqs = [capacitor2.v ~ 0.0])
end

@testset "CalibrationAnalysis" begin
    model = create_model(; C₁ = 3e-5, C₂ = 1e-6, f = 100.0)
    sys = structural_simplify(model)
    spec = CalibrationAnalysisSpec(;
        name = :test,
        model,
        abstol = 1e-6,
        reltol = 1e-6,
        data = DyadDataset(
            "juliasimtutorials", "circuit_data", independent_var = "timestamp",
            dependent_vars = ["ampermeter.i(t)"]),
        N_cols = 1,
        depvars_cols = ["ampermeter.i(t)"],
        depvars_names = ["ampermeter.i"],
        N_tunables = 1,
        search_space_names = ["capacitor2.C"],
        search_space_lb = [1e-7],
        search_space_ub = [1e-3],
        calibration_alg = "SingleShooting",
        optimizer_maxiters = 1000
    )

    r = run_analysis(spec)

    @test r.r.u[1]≈1e-5 rtol=1e-3

    @testset "Solution interface" begin
        m = AnalysisSolutionMetadata(r)

        @test length(m.artifacts) == 3

        @test_nowarn artifacts(r, m.artifacts[1].name)
        @test artifacts(r, m.artifacts[2].name) isa DataFrame

        vizdef = PlotlyVisualizationSpec(
            m.allowed_symbols[[2, 1]], (;), [Attribute("tstart", "start time", 0.0)])
        @test_nowarn customizable_visualization(r, vizdef)
    end
end
