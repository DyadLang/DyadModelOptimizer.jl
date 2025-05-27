using SnoopCompile, SnoopCompileCore

invalidations = @snoop_invalidations using DyadModelOptimizer

using Test
using Aqua

@testset "Aqua" begin
    #TODO: check `find_persistent_tasks_deps` failing spuriously
    # Aqua.find_persistent_tasks_deps(DyadModelOptimizer)
    Aqua.test_ambiguities(DyadModelOptimizer, recursive = false)
    Aqua.test_deps_compat(DyadModelOptimizer)
    Aqua.test_piracies(DyadModelOptimizer)
    Aqua.test_project_extras(DyadModelOptimizer)
    Aqua.test_stale_deps(DyadModelOptimizer, ignore = Symbol[])
    Aqua.test_unbound_args(DyadModelOptimizer)
    Aqua.test_undefined_exports(DyadModelOptimizer)
end

# no non-const globals
non_const_names = filter(x -> !isconst(DyadModelOptimizer, x),
    names(DyadModelOptimizer, all = true))
# filter out gensymed names
filter!(x -> !startswith(string(x), "#"), non_const_names)
@test isempty(non_const_names)

trees = invalidation_trees(invalidations);

# no invelidations in the package
@test isempty(filtermod(DyadModelOptimizer, trees))
