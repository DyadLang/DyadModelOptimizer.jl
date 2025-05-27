function export_petab(yamlfile::String, res::AbstractParametricUncertaintyEnsemble)
    petab_yaml = YAML.load_file(yamlfile)
    if typeof(petab_yaml["parameter_file"]) == String
        parameter_path = joinpath(dirname(yamlfile), petab_yaml["parameter_file"])
        out_df = CSV.read(parameter_path, DataFrame)
    else
        error("Multiple files not supported")
    end
    res_df = DataFrame(res)  # Todo: create a DataFrame(res:CalibrationResult) method

    length(names(res_df)) == sum(out_df.estimate) ||
        error("Number of estimated parameters does not match")
    for i in 1:size(res_df, 1)
        out_df[!, "sol_$i"] .= zero(eltype(out_df[1, "nominalValue"]))
        for (j, p) in enumerate(out_df.parameterId)
            if out_df[j, "estimate"] == 1
                out_df[j, "sol_$i"] = res_df[i, p]
            else
                in(p, names(res_df)) &&
                    error("Parameter $p is not estimated, but appears in the optimization result.")
                out_df[j, "sol_$i"] = out_df.nominalValue[j]
            end
        end
    end

    yaml_fn = splitext(yamlfile)[1] * "_optimized.yaml"
    tsv_fn = splitext(petab_yaml["parameter_file"])[1] * "_optimized.tsv"
    petab_yaml["parameter_file"] = tsv_fn
    CSV.write(joinpath(dirname(yamlfile), tsv_fn), out_df, delim = '\t')
    YAML.write_file(yaml_fn, petab_yaml)
end
