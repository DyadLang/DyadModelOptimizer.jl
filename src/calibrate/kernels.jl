# Derived from https://github.com/SciML/DiffEqFlux.jl/blob/af5fd5edbbaad38e92315d7ffff14877b15c5774/src/collocation.jl,
# MIT licensed, see https://github.com/SciML/DiffEqFlux.jl for details.

abstract type CollocationKernel end
struct EpanechnikovKernel <: CollocationKernel end
struct UniformKernel <: CollocationKernel end
struct TriangularKernel <: CollocationKernel end
struct QuarticKernel <: CollocationKernel end
struct TriweightKernel <: CollocationKernel end
struct TricubeKernel <: CollocationKernel end
struct GaussianKernel <: CollocationKernel end
struct CosineKernel <: CollocationKernel end
struct LogisticKernel <: CollocationKernel end
struct SigmoidKernel <: CollocationKernel end
struct SilvermanKernel <: CollocationKernel end

# Reference - https://en.wikipedia.org/wiki/Kernel_(statistics)

function calckernel(::EpanechnikovKernel, t)
    if abs(t) > 1
        return zero(t)
    else
        return 0.75 * (1 - t^2)
    end
end

function calckernel(::UniformKernel, t)
    if abs(t) > 1
        return 0.0
    else
        return 0.5
    end
end

function calckernel(::TriangularKernel, t)
    if abs(t) > 1
        return zero(t)
    else
        return (1 - abs(t))
    end
end

function calckernel(::QuarticKernel, t)
    if abs(t) > 1
        return zero(t)
    else
        return (15 * (1 - t^2)^2) / 16
    end
end

function calckernel(::TriweightKernel, t)
    if abs(t) > 1
        return zero(t)
    else
        return (35 * (1 - t^2)^3) / 32
    end
end

function calckernel(::TricubeKernel, t)
    if abs(t) > 1
        return zero(t)
    else
        return (70 * (1 - abs(t)^3)^3) / 81
    end
end

function calckernel(::GaussianKernel, t)
    exp(-0.5 * t^2) / (sqrt(2 * π))
end

function calckernel(::CosineKernel, t)
    if abs(t) > 1
        return zero(t)
    else
        return (π * cos(π * t / 2)) / 4
    end
end

function calckernel(::LogisticKernel, t)
    1 / (exp(t) + 2 + exp(-t))
end

function calckernel(::SigmoidKernel, t)
    2 / (π * (exp(t) + exp(-t)))
end

function calckernel(::SilvermanKernel, t)
    sin(abs(t) / 2 + π / 4) * 0.5 * exp(-abs(t) / sqrt(2))
end

function construct_t1(t, tpoints)
    hcat(ones(eltype(tpoints), length(tpoints)), tpoints .- t)
end

function construct_t2(t, tpoints)
    hcat(ones(eltype(tpoints), length(tpoints)), tpoints .- t, (tpoints .- t) .^ 2)
end

function construct_w(t, tpoints, h, kernel)
    W = @. calckernel((kernel,), ((tpoints - t) / abs(tpoints[end] - tpoints[begin])) / h) /
           h
    Diagonal(W)
end
