using Downloads
using SHA

const dir = joinpath(@__DIR__, "petabimport_tmp")

function check_case(dir, case, hash)  # Derived from https://github.com/LCSB-BioCore/SBML.jl, Apache 2.0 licensed, see repository for details
    temp_file = joinpath(dir, "temp.txt")

    for file in readdir(dir)
        open(temp_file, "a") do f
            write(f, read(joinpath(dir, file)))
        end
    end

    cksum = bytes2hex(open(io -> sha256(io), temp_file))
    rm(temp_file)
    if cksum != hash
        @warn "The downloaded test case `$case' has probably been modified in the petab_test_suite. New hash is $cksum."
    end
    nothing
end

function download_case(dir, case, hash)
    url = "https://raw.githubusercontent.com/PEtab-dev/petab_test_suite/main/petabtests/cases/v1.0.0/sbml/$(case)"

    mkpath(dir)
    filenames = [case * ".py",
        "README.md",
        "_" * case * ".yaml",
        "_" * case * "_solution.yaml",
        "_conditions.tsv",
        "_measurements.tsv",
        "_model.xml",
        "_observables.tsv",
        "_parameters.tsv",
        "_simulations.tsv"]

    for name in filenames
        Downloads.download(url * "/" * name, joinpath(dir, name))
    end
    check_case(dir, case, hash)
    nothing
end
