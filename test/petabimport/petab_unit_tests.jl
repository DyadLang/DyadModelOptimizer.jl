using Test
using CSV
using LabelledArrays: LVector
using Logging: Error
using ModelingToolkit
using DataFrames
using OrdinaryDiffEq
using SBMLToolkit
using SBML
using DyadModelOptimizer
using OptimizationBBO
using DyadModelOptimizer: get_search_space, AbstractExperiment, PetabOverride,
                          PetabProblem,
                          PetabProblemMTK, get_petab_problem, get_experiments,
                          petab2mtk,
                          Parameter, get_measurements, get_parameters, get_conditions,
                          get_observations, get_neg_llh, str2sim, create_par,
                          get_noiseformulas,
                          get_petab_simulation_df, CalibrationResult,
                          get_experiment, get_defaults, get_neg_llh, _get_module,
                          sbml_promote_expand, get_initial_assignments,
                          get_var_and_assignment, get_neg_logprior, get_chi2_function,
                          problem_chi2, setup_problem, split_replicates, get_model,
                          collapse_timepoint_overrides!

## Setup test environment for unit tests
include("utils.jl")
const t = (@parameters t)[1]
@variables A(t) B(t) obs_a(t)=1.0 __sigma_obs_a(t)=1.0
@parameters a0 b0 k1 k2
D = Differential(t)

function isequal_except(x, y, exempt_fields::Vector{Symbol}; checktype = false)
    if checktype && !isequal(typeof(x), typeof(y))
        println("Types are different: $(typeof(x))\nand\n$(typeof(y))")
        return false
    end
    for field in fieldnames(typeof(x))
        if !(field in exempt_fields)
            if !isequal(getfield(x, field), getfield(y, field))
                println("Fields `$(field)` are different:\n$(getfield(x, field))\n\nand\n\n$(getfield(y, field))")
                return false
            end
        end
    end
    return true
end

# Download test case 0001
const case = "0001"
const hash = "363872fce883167fad9786f776cdc32ed4f3305314c3c12a1426630caf1377fb"
download_case(dir, case, hash)

const yaml = joinpath(dir, "_0001.yaml")
const sbmlmod = readSBML(joinpath(dir, "_model.xml"),
    doc -> begin
        set_level_and_version(3, 2)(doc)
        sbml_promote_expand(doc)
    end)
const observable_eqs = [obs_a ~ A, __sigma_obs_a ~ 1.0]
const odesys_nonsimplified = ODESystem(sbmlmod, observed = observable_eqs)
const odesys = structural_simplify(odesys_nonsimplified)
const data_true = DataFrame("timestamp" => [0.0, 10.0],
    "obs_a" => Union{Missing, Float64}[0.7, 0.1])

const parameter_df = CSV.read(joinpath(dir, "_parameters.tsv"), DataFrame)
const condition_df = CSV.read(joinpath(dir, "_conditions.tsv"), DataFrame)
const measurement_df = CSV.read(joinpath(dir, "_measurements.tsv"), DataFrame)
const observable_df = CSV.read(joinpath(dir, "_observables.tsv"), DataFrame)
const pp_true = PetabProblem(parameter_df, [condition_df], [measurement_df],
    [observable_df], [sbmlmod], DataFrame[], [DataFrame(:simulationConditionId => [])])
const parameter_df_prior = CSV.read(joinpath(@__DIR__, "_parameters_prior.tsv"), DataFrame)
const pp_true_prior = PetabProblem(parameter_df_prior, [condition_df], [measurement_df],
    [observable_df], [sbmlmod], DataFrame[], [DataFrame(:simulationConditionId => [])])
const parameter_df_defaults = CSV.read(joinpath(@__DIR__, "_parameters_defaults.tsv"),
    DataFrame)
const pp_true_defaults = PetabProblem(parameter_df_defaults, [condition_df],
    [measurement_df], [observable_df],
    [sbmlmod], DataFrame[], [DataFrame(:simulationConditionId => [])])
const conditions = get_conditions(condition_df, odesys)
const neg_llh = get_neg_llh(pp_true, [keys(conditions)...][1])
const trials_true = AbstractExperiment[Experiment(data_true,
    odesys,
    tspan = (0.0, 10.0),
    loss_func = neg_llh,
    saveat = [0.0, 10.0],
    save_names = [obs_a],
    name = "c0")]

const search_space_true = [
    a0 => (0.0, 10.0, :identity),
    b0 => (0.0, 10.0, :identity),
    k1 => (0.0, 10.0, :identity),
    k2 => (0.0, 10.0, :identity)
]

odesys_copy = deepcopy(odesys)
for (k, v) in Dict(B.val => 1.0, k2.val => 0.6, A.val => 1.0, k1.val => 0.8, b0.val => 0.0)
    odesys_copy.defaults[k] = v
end
const invprob_true = InverseProblem(IndependentExperiments(trials_true),
    search_space_true)
const opmod = Module()
Base.eval(opmod, :(using ModelingToolkit))
Base.eval(opmod, :(t = (@parameters t)[1]))
Base.eval(opmod, :(@parameters a0, k1, notinsbml))
Base.eval(opmod, :(@variables A(t) obs_a(t)))

const par = Parameter(parameterScale = "lin",
    lowerBound = 0.0,
    upperBound = 2.0,
    nominalValue = 1.0,
    estimate = true)

## Run unit tests
# Test import_petab
invprob = @test_nowarn import_petab(yaml)

mdl = get_petab_problem(yaml).models[1]
initial_assignments = get_initial_assignments(mdl)
sys = ODESystem(mdl, defaults = initial_assignments)
invprob2 = @test_logs min_level=Error import_petab(yaml, models = [sys])

# Test get_petab_problem
pp = get_petab_problem(yaml)
@test isequal_except(pp, pp_true, [:models])
@test_broken isequal(pp, pp_true)

# Test petab2mtk
pdf = DataFrame(parameterId = ["a0"],
    parameterScale = ["lin"],
    lowerBound = [0.0],
    upperBound = [2.0],
    nominalValue = [1.0],
    estimate = [true])
cdf = DataFrame(conditionId = ["c1"],
    a0 = [1.0])
mdf = DataFrame(observableId = ["obs_a"],
    measurement = [1.0],
    time = [0.0],
    simulationConditionId = ["c1"])
odf = DataFrame(observableId = ["obs_a"],
    observableFormula = ["A"])
pp = PetabProblem(pdf,
    [cdf],
    [mdf],
    [odf],
    [sbmlmod],
    DataFrame[],
    [DataFrame(:simulationConditionId => [])])
ppm = petab2mtk(pp)

ppm_true = PetabProblemMTK(odesys,
    [obs_a ~ A],
    Dict("c1" => Dict("u0" => Vector{Pair{Num, Any}}(),
        "params" => Pair{Num, Any}[a0 => 1.0])),
    neg_llh,
    (x) -> zero(eltype(x)),
    Dict("a0" => par),
    Dict("c1" => select(mdf, Not(:simulationConditionId))),
    Dict{String, DataFrame}())
@test isequal_except(ppm, ppm_true, [:sys, :neg_llh, :neg_logprior])

# Test get_initial_assignments
@parameters compartment
assignments = get_initial_assignments(sbmlmod)
assignments_true = Dict(A => a0 * compartment, B => b0 * compartment)
@test isequal(assignments, assignments_true)

# Test get_var_and_assignment
var, assignment = get_var_and_assignment(sbmlmod, "A", SBML.MathIdent("a0"))
@test isequal(var, A)
@test isequal(assignment, a0 * compartment)

# Test get_measurements
cdf = DataFrame(conditionId = ["c1", "c2"])
mdf = DataFrame(simulationConditionId = ["c1", "c2"],
    measurement = [1.0, 2.0],
    observableId = ["obs_a", "obs_a"],
    time = [0.0, 0.0])
mdf1 = DataFrame(measurement = [1.0],
    observableId = ["obs_a"],
    time = [0.0])
mdf2 = DataFrame(measurement = [2.0],
    observableId = ["obs_a"],
    time = [0.0])
measurements = get_measurements(cdf, mdf)

measurements_true = Dict("c1" => mdf1, "c2" => mdf2)

@test isequal(measurements, measurements_true)

# Test get_parameters
pdf = DataFrame(parameterId = ["a0"],
    parameterScale = ["lin"],
    lowerBound = [0.0],
    upperBound = [2.0],
    nominalValue = [1.0],
    estimate = 1)

params = get_parameters(pdf)

params_true = Dict("a0" => Parameter(parameterScale = "lin", lowerBound = 0.0,
    upperBound = 2.0, nominalValue = 1.0, estimate = true))

@test isequal(params, params_true)

# Test get_conditions
cdf = DataFrame("conditionId" => ["c1", "c2"],
    "a0" => [1.0, 2.0])
conds = get_conditions(cdf, odesys)

conds_true = Dict("c1" => Dict("u0" => [], "params" => [a0 => 1.0]),
    "c2" => Dict("u0" => [], "params" => [a0 => 2.0]))

@test isequal(conds, conds_true)

cdf = DataFrame("conditionId" => ["c1", "c2"])
conds = get_conditions(cdf, odesys)

conds_true = Dict("c1" => Dict("u0" => [], "params" => []),
    "c2" => Dict("u0" => [], "params" => []))
@test isequal(conds, conds_true)

# Test get_observations
obs = get_observations(pp_true, opmod)
obs_true = [obs_a ~ A]
@test isequal(obs, obs_true)

pp_wrong = PetabProblem(parameter_df, [condition_df], [measurement_df], [observable_df],
    [sbmlmod, sbmlmod], DataFrame[], [DataFrame(:simulationConditionId => [])])
@test_throws ErrorException("Multiple files not supported") get_observations(pp_wrong,
    opmod)

# Test get_logprior
neg_logprior = get_neg_logprior(pp_true_prior)
@test_nowarn neg_logprior(LVector((a0 = 1.0, b0 = 1.0, k1 = 1.0, k2 = 1.0)))  # Todo: figure out what that test should actually return or wait for https://github.com/PEtab-dev/petab_test_suite/issues/60
@test isequal(neg_logprior(LVector((a0 = 3.0, b0 = 1.0, k1 = 1.0, k2 = 1.0))), Inf)

# Test get_noiseformulas
odf_noise = DataFrame(observableId = ["obs_a"],
    observableFormula = ["A"],
    noiseFormula = ["k1 * obs_a"])
pp_noise = PetabProblem(parameter_df, [condition_df], [measurement_df], [odf_noise],
    [sbmlmod], DataFrame[], [DataFrame(:simulationConditionId => [])])
noiseformula = get_noiseformulas(pp_noise, opmod)

@variables __sigma_obs_a(t)
noiseformula_true = [__sigma_obs_a ~ k1 * obs_a]

@test isequal(noiseformula, noiseformula_true)

# Test parameterScale
pars = Dict(
    "a0" => Parameter(parameterScale = "lin",
        lowerBound = 0.0,
        upperBound = 2.0,
        nominalValue = 1.0,
        estimate = true),
    "b0" => Parameter(parameterScale = "lin",
        lowerBound = 0.0,
        upperBound = 2.0,
        nominalValue = 0.0,
        estimate = true),
    "k1" => Parameter(parameterScale = "log",
        lowerBound = 0.0,
        upperBound = 2.0,
        nominalValue = 0.8,
        estimate = true),
    "k2" => Parameter(parameterScale = "log10",
        lowerBound = 0.0,
        upperBound = 2.0,
        nominalValue = 0.6,
        estimate = true))
ppm_true = PetabProblemMTK(odesys,
    [obs_a ~ A],
    Dict("c1" => Dict("u0" => Vector{Pair{Num, Any}}(),
        "params" => Pair{Num, Any}[a0 => 1.0])),
    l2loss,
    (x) -> zero(eltype(x)),
    pars,
    Dict("c1" => select(measurement_df, Not(:simulationConditionId))),
    Dict{String, DataFrame}())
invprob_scale = InverseProblem(ppm_true)
scale = Set(get_search_space(invprob_scale))
scale_true = Set([b0 => (0.0, 2.0, :identity)
                  k1 => (0.0, 2.0, :log)
                  a0 => (0.0, 2.0, :identity)
                  k2 => (0.0, 2.0, :log10)])
@test isequal(scale, scale_true)

# Test get_experiments
trial_true = Experiment(data_true,
    odesys,
    tspan = (0.0, 10.0),
    loss_func = l2loss,
    saveat = [0.0, 10.0],
    save_names = [obs_a],
    name = "c1",
    params = [a0 => 1.0])
ppm = PetabProblemMTK(odesys,
    [obs_a ~ A],
    Dict("c1" => Dict("u0" => Vector{Pair{Num, Any}}(),
        "params" => Pair{Num, Any}[a0 => 1.0])),
    l2loss,
    (x) -> zero(eltype(x)),
    Dict("a0" => par),
    Dict("c1" => select(measurement_df, Not(:simulationConditionId))),
    Dict{String, DataFrame}())
trials = get_experiments(ppm)
@test collect(trials[1].config.aliased_u0) == collect(trial_true.config.aliased_u0)
@test collect(trials[1].config.aliased_params) == collect(trial_true.config.aliased_params)
@test isequal_except(trials[1], trial_true, [:config])

# Test create_par
x = create_par(convert(String7, "x"))
@test x isa Num
@test x.val.name == :x

# Test str2sim
term1 = "3*A + a0 * notinsbml"
symterm = str2sim(term1, opmod)

@parameters notinsbml
symterm_true = 3 * A + a0 * notinsbml

@test isequal(symterm, symterm_true)

# Test get_petab_simulation_df
res = calibrate(invprob, SingleShooting(maxiters = 1))
x0 = DyadModelOptimizer.initial_state(Any, invprob)
defres = CalibrationResult(x0, res.prob, nothing, x0, res.alg, [], missing)
sim_df = get_petab_simulation_df(defres, pp_true)
@test isequal(select(sim_df, Not(:simulation)), select(measurement_df, Not(:measurement)))

res2 = calibrate(invprob2, SingleShooting(maxiters = 1))
@test isequal(res2.u, res.u)

# Test petab_export
vp = parametric_uq(invprob,
    StochGlobalOpt(method = SingleShooting(maxiters = 1,
        optimizer = BBO_adaptive_de_rand_1_bin_radiuslimited())),
    sample_size = 2)
DyadModelOptimizer.export_petab(yaml, vp)
pp = get_petab_problem(joinpath(dir, "_0001_optimized.yaml"))
@test pp.parameter_df.sol_1 isa Vector{Float64}
@test length(pp.parameter_df.sol_2) == 4

# Test get_experiment
experiment = get_experiment("c0", invprob_true)
@test isequal(experiment, trials_true[1])

# Test get_defaults
def = DyadModelOptimizer.get_default(PetabOverride(ppm_true),
    ModelingToolkit.defaults(odesys),
    a0,
    nothing)
@test isequal(def, 1.0)

# Test problem_chi2
chi2 = problem_chi2(defres)
chi2_true = 0.79183798368486  # Taken from _0001_solution.yaml
@test isapprox(chi2, chi2_true, atol = 1e-3)

# Test setup_problem
invprob = InverseProblem(petab2mtk(pp_true_defaults))
prob = setup_problem(only(get_experiments(invprob)), invprob, [1.0, 0.8])
@test isequal(prob.u0, [0.0, 0.8])  # For some reason the order is switched
@test isequal(prob.p, [0.0, 1.0, 0.8, 0.6, 1.0])

# Test split_replicates
df = DataFrame(:a => [1, 2, 3, 3, 4, 4, 4], :measurement => [1, 2, 3, 4, 5, 6, 7])
df1 = DataFrame(:a => [1, 2, 3, 4], :measurement => [1, 2, 3, 5])
df2 = DataFrame(:a => [3, 4], :measurement => [4, 6])
df3 = DataFrame(:a => [4], :measurement => [7])
dfs_true = [df1, df2, df3]
dfs_split = split_replicates(df)
@test isequal(dfs_split, dfs_true)

# Test create_petab_template
outdir = joinpath(@__DIR__, "template_tmp")
create_petab_template(outdir)
for fn in ["conditions.tsv", "measurements.tsv", "model.xml", "observables.tsv",
    "parameters.tsv", "petab.yaml"]
    @test isfile(joinpath(outdir, fn))
end
create_petab_template(outdir, force = true)

invprob = import_petab(joinpath(outdir, "petab.yaml"))
invprob = remake_experiments(invprob, alg = TRBDF2())
costfun = objective(invprob, SingleShooting(maxiters = 1))

costval_true = -0.9657722809371183  # Just a regression test as ground truth for this case is not known. Correct costval calculation is tested in petab_test_suite.jl.
costval = costfun()

@test isapprox(costval, costval_true)

r = calibrate(invprob, SingleShooting(maxiters = 10^3, maxtime = 100))
@test length(r) == 14

m = SingleShooting(maxiters = 10^3, maxtime = 100,
    optimizer = BBO_adaptive_de_rand_1_bin_radiuslimited())
ps = parametric_uq(invprob, StochGlobalOpt(method = m), sample_size = 2)

@test length(ps) == 2

# Test collapse_timepoint_overrides!
odf = DataFrame("observableId" => ["obs_A", "obs_B"],
    "observableFormula" => ["observableParameter1_obs_A + observableParameter2_obs_A",
        "observableParameter1_obs_B + observableParameter2_obs_B"])
mdf = DataFrame("observableId" => ["obs_A", "obs_A", "obs_B"],
    "observableParameters" => ["op1a; op2a",
        "op1a; op2a",
        "op1b; op2b"])
odf_true = DataFrame("observableId" => ["obs_A", "obs_B"],
    "observableFormula" => ["op1a + op2a",
        "op1b + op2b"])
collapse_timepoint_overrides!(odf, mdf)
@test isequal(odf, odf_true)

mdf = DataFrame("observableId" => ["obs_A", "obs_A", "obs_B"],
    "observableParameters" => ["op1a; op2a",
        "_op1a; op2a",
        "op1b; op2b"])
@test_throws ErrorException("Multiple different observableParameter overrides not supported") collapse_timepoint_overrides!(
    odf,
    mdf)

odf = DataFrame("observableId" => ["obs_A", "obs_B"],
    "noiseFormula" => ["noiseParameter1_obs_A + noiseParameter2_obs_A",
        "noiseParameter1_obs_B + noiseParameter2_obs_B"])
mdf = DataFrame("observableId" => ["obs_A", "obs_A", "obs_B"],
    "noiseParameters" => ["op1a; op2a",
        "op1a; op2a",
        "op1b; op2b"])
odf_true = DataFrame("observableId" => ["obs_A", "obs_B"],
    "noiseFormula" => ["op1a + op2a",
        "op1b + op2b"])
collapse_timepoint_overrides!(odf, mdf)
@test isequal(odf, odf_true)

# Delete temporary files
rm(outdir, recursive = true)
rm(dir, recursive = true)
