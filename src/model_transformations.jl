"""
    AbstractModelTransformation

The supertype of all model transformations.
"""
abstract type AbstractModelTransformation end

abstract type AbstractModelWrapper end

# model transformations can change the model behaviour either via DiffEq-style callbacks
# or via wrapping the problem function in another function
# (or more concretely a callable struct <: AbstractModelWrapper).
# More generally, model transformations can arbitrarily change the model & the problem that gets cached by
# the Experiment and can theoretically be arbitrarily nested.
# The model transformation can have parameters that can be added to the search space automatically by get_tunable_params.

abstract type AbstractPredictionErrorMethod <: AbstractModelTransformation end

callback_form_transformation(::AbstractModelTransformation, data, model) = nothing

setup_wrapper(::AbstractModelTransformation, experiment, invprob, x, f) = f

"""
    augment_model(mt::AbstractModelTransformation, model)

A model transformation can optionally change the user given model.
Note that the user given model is structurally simplified at this stage.
This function should return a new model of the same type as the user given one.
Examples of use for this are adding extra parameters to a model.
"""
augment_model(::AbstractModelTransformation, model) = model

"""
    augment_prob(mt::AbstractModelTransformation, prob)

A model transformation can optionally change the problem that
is compiled from a given model. One possible application for this
is changing the problem function.
"""
augment_prob(::AbstractModelTransformation, prob) = prob

"""
    get_tunable_params(mt::AbstractModelTransformation, experiment)

A model transformation can add tunable parameters (added to the model via [`augment_model`](@ref))
to the search space of the inverse problem. This should return a vector of pairs
from the symbolic parameter variable to a tuple of lower and upper bounds for the parameter.
"""
get_tunable_params(::AbstractModelTransformation, experiment) = Pair[]

"""
    callback_form_transformation(::AbstractModelTransformation, data, model)

Model transformations that have associated callbacks need to implement this. This function
should return the corresponding callback that is passed to `solve` via `callback`.
If the model transformation does not have an associated callback, this fallback returns `nothing`.
"""
function callback_form_transformation(mt::AbstractPredictionErrorMethod, data, model)
    prediction_error_callback(mt, data, model)
end
