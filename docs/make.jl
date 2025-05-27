using DyadModelOptimizer
using Documenter
using PkgAuthentication
using DyadInterface
using DyadData

ENV["GKSwstype"] = "100"
ENV["JULIA_DEBUG"] = "Documenter"
PkgAuthentication.authenticate()

DocMeta.setdocmeta!(DyadModelOptimizer, :DocTestSetup, :(using DyadModelOptimizer);
    recursive = true)

makedocs(;
    modules = [DyadInterface, DyadData, DyadModelOptimizer],
    authors = "JuliaHub",
    repo = "https://github.com/JuliaComputing/DyadModelOptimizer.jl",
    sitename = "DyadModelOptimizer",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://JuliaComputing.github.io/DyadModelOptimizer.jl",
        assets = String[],
        edit_link = nothing),
    linkcheck = true,
    warnonly = [:missing_docs],
    pagesonly = true,
    pages = [
        "index.md",
        "getting_started.md",
        "Topic Tutorials" => [
            "tutorials/data.md",
            "tutorials/loss.md",
            # "tutorials/juliahub.md",
            "tutorials/ballandbeam.md",
            "tutorials/pendulum_pem.md",
            # "tutorials/autocomplete.md",
            "tutorials/dcmotor.md",
            "tutorials/cstr.md"
        ],
        "Workflow Examples" => [
            "examples/TerminalVelocity.md",
            "examples/ThermalConduction.md",
            "examples/CoupledExperiments.md",
            "examples/ChuaCircuit.md",
            # "examples/CalibrateFMU.md",
            # "examples/UncertaintyAnalysisFMUs.md",
            "examples/gsa.md"
        ],
        "Manual and APIs" => [
            "manual/experiments.md",
            "manual/inverseproblem.md",
            "manual/calibrate.md",
            "manual/parametric_uq.md",
            "manual/results.md",
            # "manual/subsample.md",
            "manual/plot.md",
            "manual/collocation.md",
            "manual/misc.md",
            "manual/juliahub.md"
        ]
    ])

deploydocs(;
    repo = "github.com/JuliaComputing/DyadModelOptimizer.jl",
    branch = "gh-pages", # gh-pages is the default branch, just making it explicit
    push_preview = true)
