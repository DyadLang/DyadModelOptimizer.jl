Base.@kwdef struct Parameter{PN, PS, LB <: Float64, UB <: Float64, NV <: Float64,
    ES <: Bool, IPT, IPP, OPT, OPP}
    parameterName::PN = nothing
    parameterScale::PS
    lowerBound::LB
    upperBound::UB
    nominalValue::NV
    estimate::ES
    initializationPriorType::IPT = nothing
    initializationPriorParameters::IPP = nothing
    objectivePriorType::OPT = nothing
    objectivePriorParameters::OPP = nothing
end

struct PetabProblemMTK{S <: ODESystem, O <: Vector{Equation},
    C, NLL <: Function, NLP <: Function, PA <: Dict{String, <:Parameter},
    M <: Dict{String, DataFrame}, PE <: Dict{String, DataFrame}}
    sys::S  # already contains observations and noise in sys.observed
    observations::O  # To be supplied to InverseProblem as a new keyword argument
    conditions::C  # To be supplied to Experiment as `params`.
    neg_llh::NLL # take a DataFrame (measurements) and ODESolution (observations) as input and returns a vector of Floats. Goes into Experiment.err
    neg_logprior::NLP  # Supplied to InverseProblem.penalty
    parameters::PA  # To be supplied to InverseProblem as `search_space`
    measurements::M
    # Todo: visualization.
    perturbations::PE
end

struct PetabOverride{P}
    parameters::P

    function PetabOverride(ppm::PetabProblemMTK)
        params = ppm.parameters  # TODO: write accessor functions for PetabProblemMTK.

        new{typeof(params)}(params)
    end
end

Base.broadcastable(o::PetabOverride) = (o,)

function get_default(po::PetabOverride, model_defaults, model_var, v)
    key = string(model_var)
    if haskey(po.parameters, key)
        val = po.parameters[key].nominalValue
        @debug "override for $(key): $val"
        val
    else
        model_defaults[model_var]
    end
end
