using DyadModelOptimizer
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using ModelingToolkitStandardLibrary.Blocks
using OrdinaryDiffEqVerner
using Optimization
using DataFrames
using CSV
using Plots
using Test

using DataInterpolations
import ForwardDiff
using DyadData

function ballandbeamsystem(x0, ϕ_vec, t_vec)
    ϕ0 = ϕ_vec[1]
    @variables x(t)=x0 ϕ(t)=ϕ0 ball_position(t)
    @parameters g=9.81 I=0.2 Fv=1

    @named src = Interpolation(CubicSpline, ϕ_vec, t_vec)
    @named clk = ContinuousClock()
    @named Dϕ_input = RealInput()

    eqs = [
        connect(clk.output, src.input),
        connect(src.output, Dϕ_input),
        D(ϕ) ~ Dϕ_input.u,
        # I = J/(m*r^2)
        0 ~ (1 + I) * D(D(x)) - x * (D(ϕ))^2 - g * sin(ϕ) + Fv * D(x)
    ]
    @named ballandbeam = ODESystem(eqs, t; systems = [Dϕ_input, src, clk])
end

training_dataset = DyadDataset("juliasimtutorials", "ball_beam",
    independent_var = "timestamp", dependent_vars = ["ϕ", "x"])
df = build_dataframe(training_dataset)

t_vec = df[!, "timestamp"]
ϕ_vec = df[:, "ϕ"]

input_func = CubicSpline(ϕ_vec, t_vec)
dinput_vec = map(Base.Fix1(DataInterpolations.derivative, input_func), t_vec)

model = ballandbeamsystem(df.x[1], dinput_vec, t_vec)
sys = structural_simplify(model)

experiment = Experiment(
    df,
    sys,
    alg = Vern9(),
    overrides = [D(sys.x) => dinput_vec[1]],
    reltol = 1e-8, abstol = 1e-10)

prob = InverseProblem(experiment, [sys.I => (0, 2), sys.Fv => (0, 2)])

# ensure that the problem is setup correctly
sol0 = simulate(experiment, prob)
@test SciMLBase.successful_retcode(sol0)
@test sol0[sys.ϕ, end]≈ϕ_vec[end] rtol=0.3

alg = MultipleShooting(maxiters = 10^3, maxtime = 450, trajectories = 400,
    optimizer = IpoptOptimizer(),
    ensemblealg = EnsembleThreads(),
    continuitylossweight = 1e3)
# alg = SingleShooting(maxiters = 100, maxtime = 200)
r = calibrate(prob, alg)

# test we don't just hit the bound
@test r[:I] < 2
# The I for a perfectr sphere is 0.4, but we don't have that & the model is imprecise
@test r[:I]≈0.4 atol=0.1 broken=true
@test r.original.objective < 0.8

@test_nowarn plot_shooting_segments(experiment, r, ms = 0.8)
# alg = StochGlobalOpt(maxiters=5_000, method=MultipleShooting, trajectories=4)
# ps = parametric_uq(prob, alg, sample_size=10)
