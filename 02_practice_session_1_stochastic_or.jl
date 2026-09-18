### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 4853369b-c13c-40ff-b5e2-ec74fe882f1d
begin
    import Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
end

# ╔═╡ ffb5b23f-da28-4eea-b1a5-edee6911dd25
begin
    using StochasticTailAssignment
    using StochasticTailAssignment.AircraftRoutingBase
    using StochasticTailAssignment.InstanceGenerator
    using StochasticTailAssignment.FlightDelayModel
    using Dates
    using JLD2
    using PlutoUI
    using PlutoTeachingTools
    using ProgressLogging
    using WGLMakie # If this doesn't work on your browser, delte this line and uncomment the line below
    # using CairoMakie
end

# ╔═╡ eeb13b61-d1d8-4824-9253-581a74450d3e
ChooseDisplayMode()

# ╔═╡ eaaf49ba-fa58-4b27-a1a0-86a45c12e74a
md"""
# Practice session 1: stochastic OR
"""

# ╔═╡ 102fa231-68fd-4a23-8bd6-c2c85ab05b35
md"""
## Julia in a few lines, coming from Python

No Julia to write here, just some specific syntax worth recognizing in the code cells below.

- `(; a, b)` builds a **NamedTuple**, attribute access `nt.a` instead of `nt["a"]`.
- Arguments after `;` in a call are **keyword arguments**, e.g. `f(x; nb_scenarios=30)`.
- A dot before an operator or after a function name means **broadcasting**, element-wise, like NumPy.
- Julia allows **unicode identifiers**, so `λ`, `μ`, `ξ` are valid variable names.
"""

# ╔═╡ 34fc4248-9418-4376-b91c-6de852386d79
md"""
## Problem glossary

A **flight leg** is one scheduled flight between two airports.
An **aircraft** flies a sequence of legs (field `immat`, the tail number).
A **route** is an ordered sequence of compatible legs assigned to one aircraft.
A **delay scenario** is one random draw of delays, propagated forward through an aircraft's rotation.
A **column** is a route as a decision variable in the master problem.
The objective combines operational cost with expected delay cost over all scenarios.
"""

# ╔═╡ 9ac34107-c99d-4e46-adf5-c35ab1b146a1
md"""
## Concept: the master problem is a set-partitioning problem

Each route has a precomputed cost, from simulating delay propagation over all scenarios.
The master problem is

```math
\min_{y} \sum_{a} \sum_{r \in \mathcal{R}(a)} c_r^a \, y_r^a
\quad \text{s.t.} \quad
\sum_{a} \sum_{r \ni \ell} y_r^a = 1 \ \ \forall \ell, \qquad
\sum_{r \in \mathcal{R}(a)} y_r^a = 1 \ \ \forall a.
```

The first set of constraints covers each leg exactly once, the second assigns exactly one route per aircraft.
Routes per aircraft grow exponentially with the number of legs, so we generate only the useful ones through column generation.
"""

# ╔═╡ a26e1d41-4689-4dfe-a19c-72886dfe0cdd
keyconcept(
    "Set-partitioning master problem",
    md"Once routes are the unit of decision, the master problem picks one route per aircraft so that every leg is covered exactly once.",
)

# ╔═╡ 3ffe51f2-6c05-4456-9464-d5b04fa65052
md"""
## Concept: pricing is a stochastic shortest path

Column generation adds a route only when it improves the master objective.
The master's dual prices (one per leg, one per aircraft) turn this into a search for the route with the smallest reduced cost:

```math
\min_{r \in \mathcal{R}(a)} \ c_r^a - \sum_{\ell \in r} \lambda_\ell - \mu_a.
```

Labels on this shortest path are piecewise linear functions, one per scenario, compared by dominance and pruned instead of enumerated.
A new column is added whenever the minimum reduced cost is negative, until no aircraft can find one.
"""

# ╔═╡ 6df4c710-f90f-4cae-803a-eff3eb1d4efb
keyconcept(
    "Pricing is a stochastic shortest path",
    md"Because the route cost takes delay propagation and expected delay into account, minimizing reduced cost is a resource-constrained shortest path problem on the schedule graph.",
)

# ╔═╡ 7c35904a-617f-42a7-b96c-4e130371ab03
md"""
## Concept: scenarios multiply the pricing cost

Each label has one coordinate per scenario, so more scenarios means costlier labels and weaker dominance (a label must win on every coordinate to dominate another):

```math
\text{pricing cost per aircraft} \ \approx \ O\bigl(S \times \text{number of surviving labels}(S)\bigr).
```

The scaling experiment below measures this directly.
"""

# ╔═╡ ad00114d-cc92-4d4c-8699-23fea3b2ec88
md"""
## Concept: diving turns the fractional solution into an integer one

Column generation stops at a fractional LP optimum, a lower bound on the true cost.
Diving repairs it with a depth-first search: fix the largest-fraction route, shrink the graph, re-run column generation, backtrack on infeasibility.

```math
\text{diving runtime} \ \approx \ (\text{number of aircraft}) \times (\text{one column generation run on a shrinking graph}).
```

The graph shrinks at every step, so diving typically costs about one to two full column generation runs in total.
"""

# ╔═╡ bd2b7df3-ca14-4672-bfd8-f5aed2969ab2
keyconcept(
    "Column generation dominates the total runtime",
    md"Column generation dominates the total runtime you will see below at these instance sizes, diving adds one shrinking column generation re-solve per aircraft on top of the first, full-size run.",
)

# ╔═╡ 21ef4150-020b-418b-b157-3c33b7c93fb6
md"""
legs: $(@bind selected_legs PlutoUI.Slider(20:10:80; default=50, show_value=true))

scenarios: $(@bind selected_scenarios PlutoUI.Slider(5:5:30; default=30, show_value=true))

seed: $(@bind selected_seed PlutoUI.Slider(1:1:100; default=13, show_value=true))

display limit: $(@bind display_limit PlutoUI.Slider(5:5:40; default=20, show_value=true))
"""

# ╔═╡ 9670de72-5357-4067-b838-bf0eda1b0830
md"The solve button takes a few seconds."

# ╔═╡ 3e9f123f-cf67-416b-b579-7b3eac3d1309
tip(
    md"Runtimes below were measured on the instructor's machine, a laptop may well be slower.",
)

# ╔═╡ 5411c3ab-b5a4-4334-9d41-883e808b171c
md"""run deterministic + stochastic solve: $(@bind solve_click PlutoUI.CounterButton("Run deterministic + stochastic solve"))"""

# ╔═╡ fbea2d43-5577-4e31-912f-66b2ce80ae5e
question_box(
    md"""
What does the optimality gap above tell you about how close diving's integer solution is to the column generation lower bound?
""",
)

# ╔═╡ 19824c74-077a-4fd3-ae54-a1eab59850d1
question_box(
    md"""
Does the in-sample improvement hold up on the fresh out-of-sample scenarios?
Why might a small scenario count make this comparison noisy?
""",
)

# ╔═╡ 8d27c291-cef1-4a2e-8b94-d8e6264bced9
md"""
The "Show delays" bar coloring is only interactive in a live Pluto session, on GLMakie or WGLMakie.
"""

# ╔═╡ 4d89f83a-4baa-4119-8cba-23c2de5f9991
md"""
Both curves converge to `lp_lower_bound`, the LP optimum, a lower bound on the integer (diving) solution's cost.
"""

# ╔═╡ 8ba1c220-0488-46c6-9e5c-66322d2caa5c
md"""
## Scaling experiment

Repeats the solve at several instance sizes, live: a legs sweep (30, 50, 65, 80, up to your chosen maximum) at fixed scenarios, and a scenario sweep (5, 15, 30) at 40 legs fixed.
"""

# ╔═╡ 400669f2-cb60-467f-9b7d-492fb2079063
tip(md"Both sweeps run behind the same button, a larger maximum takes longer.")

# ╔═╡ 645dc443-a89c-4696-a09e-446f1b8e4f79
md"""
largest instance in the sweep: $(@bind scale_max_legs PlutoUI.Slider([80, 100, 120, 150]; default=100, show_value=true))

run scaling experiment: $(@bind scale_click PlutoUI.CounterButton("Run scaling experiment"))
"""

# ╔═╡ 64be96b2-7016-4a70-b8fa-8aea5193770d
md"The scaling experiment takes under a minute up to 80 legs, about two minutes at 100, about five at 120, about ten at 150."

# ╔═╡ 0f78170c-b897-4c8b-b009-7591ebed0a6e
# Summarizes one raw sweep point: the number of legs, the number of scenarios, the total
# runtime in seconds, the number of generated columns, and the LP objective.
function build_scaleup_row(raw)
    return (
        nb_legs = raw.nb_legs,
        nb_scenarios = raw.nb_scenarios,
        runtime = raw.total_time,
        columns = raw.nb_columns,
        objective = raw.objective,
    )
end

# ╔═╡ 3a6f3005-fa01-4644-a6c6-dc749c776891
question_box(
    md"""
How does total runtime grow when you double the number of legs, versus when you double the number of scenarios?
Which of the two dominates at these instance sizes, and why?
""",
)

# ╔═╡ d0a52953-1a3d-4e0e-93a9-f4ed4fbd4cbf
md"""
## Build a labeled dataset

Part 2 trains a neural network to imitate the solver you just ran, from labeled examples: instances paired with the routes this part's solver chose.
The button below labels a chosen number of instances (50 legs, 30 scenarios, one per seed), fixed so part 2 always sees this size.
Each saved example holds the schedule, its delay scenarios, and the expert routes.
"""

# ╔═╡ 6d079949-624a-46f3-b73e-4922ca4ced45
tip(md"About a minute total for the default 30, spent once, offline.")

# ╔═╡ 75ffc900-562a-4d24-9bbb-a9cffab30228
question_box(
    md"""
`expert_routes` below is the expert solver the dataset button uses to label every instance.
Identify its three stages, warm start, column generation, diving, each feeding the next.
""",
)

# ╔═╡ 7dcd4eb4-b302-40c6-9af3-0b6788f9d007
# Optional tweak: change this value and re-press the dataset or solve button to see the effect
# on the number of skipped instances and the average expert time per instance. k is the paper's
# taboo list size, the list of fixed columns that led to an infeasible branch, the dive
# backtracks and stops once that list exceeds k. The paper uses 15. Lower it to see more
# instances skipped (dive gives up sooner) but each attempt run faster.
const TABOO_LIST_SIZE = 15

# ╔═╡ a145e517-1476-4065-9894-9168b08e1ffa
"Chain warm start, column generation and one dive, return the expert `Vector{Route}` or `nothing` if the dive is infeasible."
function expert_routes(schedule, root_delays, delay_cost_function)
    warm_start, _, _ = solve_aircraft_routing(schedule; silent = true)
    isempty(warm_start) && error("deterministic warm start was infeasible")
    cg = stochastic_column_generation(
        schedule,
        warm_start;
        root_delays,
        delay_cost_function,
        max_nb_columns = 10_000,
        tol = 1e-6,
        silent = true,
    )
    cg.feasible || error("stochastic column generation was infeasible")
    dived, feasible = diving_heuristic_with_backtracking!(
        schedule,
        cg.columns,
        root_delays,
        cg.dual_values,
        TABOO_LIST_SIZE;
        model_builder = highs_model,
        delay_cost_function,
        silent = true,
    )
    feasible && return Vector{Route}(dived)
    return nothing
end

# ╔═╡ eb0f60d7-c69c-4937-a432-46de60d55484
warning_box(
    md"At k = 15 the dive rarely fails, 0 of 30 instances were skipped when this dataset was last labeled, but `expert_routes` can still return `nothing`, and the dataset button skips and counts those instances as a safety net.",
)

# ╔═╡ 81ead6c7-6530-4168-ae67-75dc5f4be8f1
md"""
number of instances: $(@bind dataset_size PlutoUI.Slider(10:5:40; default=30, show_value=true))

build the labeled dataset: $(@bind dataset_click PlutoUI.CounterButton("Build and save the labeled dataset"))

The instance size itself is not a slider here, every instance is generated at 50 legs and 30 scenarios so that practice session 2 always sees practice session 1's default size.
"""

# ╔═╡ 18c72211-47b7-4ed4-8edb-32de944452c2
md"The dataset button takes about a minute for the default 30 instances."

# ╔═╡ 45b7bfe0-be18-4486-802b-5b6b8338ce2e
md"""
## Appendix: helpers

Plotting, caching, and formatting code that supports the sections above.
Nothing here needs reading to follow the notebook, it is safe to skip.
"""

# ╔═╡ 95c7ee9b-e6f1-415b-9b85-5c78c6216060
begin
    const OOS_SEED_SHIFT = 1000
    # Dataset labeling uses seeds DATASET_SEED_BASE+1 .. DATASET_SEED_BASE+40 (2001 to
    # 2040), disjoint from the live-instance seed slider (1:100) and the out-of-sample shift
    # (1000+selected_seed) in this notebook. Practice session 2 does not reuse this range: each of its
    # instances draws its own schedule seed with `rand(rng, 1:1_000_000)`, so an exact
    # collision with practice session 1's 2001 to 2040 is possible in principle but negligible in
    # practice (about 40 in 1,000,000 per draw).
    const DATASET_SEED_BASE = 2000
    control_int(value, default, lower, upper) =
        clamp(value isa Integer ? value : default, lower, upper)
    control_choice(value, default, choices) = value in choices ? value : default
    valid_route_collection(routes) =
        routes isa AbstractVector &&
        !isempty(routes) &&
        all(route -> route isa AbstractRoute, routes)
    # Legs sweep for the scaling experiment: the four small points always run, extended up
    # to the chosen maximum (80, 100, 120 or 150 legs) with the "largest instance in the
    # sweep" slider.
    const SCALEUP_BASE_LEGS = (30, 50, 65, 80)
    const SCALEUP_EXTRA_LEGS = (100, 120, 150)
    scaleup_legs_cases(max_legs) =
        (SCALEUP_BASE_LEGS..., (nb for nb in SCALEUP_EXTRA_LEGS if nb <= max_legs)...)
    format_seconds(x) = ismissing(x) ? "n/a" : string(round(x; digits = 2), " s")
    format_cost(x) = ismissing(x) ? "n/a" : string(round(x; digits = 1))
    format_pct(x) = ismissing(x) ? "n/a" : string(round(x; digits = 2), " %")
end

# ╔═╡ 76d7068f-0b6c-4ba7-848a-67e7e28c793e
begin
    schedule, root_delays, delay_cost_fn = generate_benchmark_instance(
        control_int(selected_legs, 50, 20, 80);
        nb_scenarios = control_int(selected_scenarios, 30, 5, 30),
        seed = control_int(selected_seed, 13, 1, 100),
    )
    Markdown.parse(
        "**Current instance**: $(nb_legs(schedule)) legs, $(nb_immats(schedule)) aircraft, $(size(root_delays, 1)) delay scenarios.",
    )
end

# ╔═╡ dbd10df9-195e-4830-857a-9283d7e37e15
current_request = (
    legs = control_int(selected_legs, 50, 20, 80),
    scenarios = control_int(selected_scenarios, 30, 5, 30),
    seed = control_int(selected_seed, 13, 1, 100),
)

# ╔═╡ 3b80d3d6-a3fe-4101-97b5-7f899f8a37a9
current_scale_request = (
    legs_cases = scaleup_legs_cases(control_choice(scale_max_legs, 100, (80, 100, 120, 150))),
    scenario_cases = (5, 15, 30),
    scenario_fixed_legs = 40,
    scenarios = control_int(selected_scenarios, 30, 5, 30),
    seed = control_int(selected_seed, 13, 1, 100),
)

# ╔═╡ 2ac3e869-99b1-47d3-95bf-92441c67f22d
function scale_point(nb, nb_scenarios_value, seed)
    try
        s, d, c = generate_benchmark_instance(nb; nb_scenarios = nb_scenarios_value, seed)
        det_timed = @timed solve_aircraft_routing(s; silent = true)
        det_routes, _, _ = det_timed.value
        valid_route_collection(det_routes) || error("deterministic solve was infeasible")
        cg_timed = @timed stochastic_column_generation(
            s,
            det_routes;
            root_delays = d,
            delay_cost_function = c,
            max_nb_columns = 10_000,
            tol = 1e-6,
            silent = true,
        )
        cg_timed.value.feasible || error("stochastic column generation was infeasible")
        dive_timed = @timed diving_heuristic_with_backtracking!(
            s,
            cg_timed.value.columns,
            d,
            cg_timed.value.dual_values,
            TABOO_LIST_SIZE; # taboo list size
            model_builder = highs_model,
            delay_cost_function = c,
            silent = true,
        )
        _, dive_feasible = dive_timed.value
        dive_feasible || error("diving was infeasible")
        (
            nb_legs = nb,
            nb_scenarios = nb_scenarios_value,
            status = :ok,
            error = nothing,
            total_time = det_timed.time + cg_timed.time + dive_timed.time,
            nb_columns = length(cg_timed.value.columns),
            objective = cg_timed.value.obj,
        )
    catch caught_error
        (
            nb_legs = nb,
            nb_scenarios = nb_scenarios_value,
            status = :error,
            error = sprint(showerror, caught_error),
            total_time = missing,
            nb_columns = missing,
            objective = missing,
        )
    end
end

# ╔═╡ 29e92ee1-763b-463b-8c6d-726e7e7c4fc4
begin
    solve_request_cache = Ref{Any}((click = 0, request = nothing))
    solve_result_cache = Ref{Any}((click = 0, request = nothing, result = nothing))
    scale_request_cache = Ref{Any}((click = 0, request = nothing))
    scale_result_cache = Ref{Any}((click = 0, request = nothing, result = nothing))
    dataset_request_cache = Ref{Any}((click = 0, request = nothing))
    dataset_result_cache = Ref{Any}((click = 0, request = nothing, result = nothing))
end

# ╔═╡ 1b996341-fbb7-4ac0-9384-8c73eea0f428
begin
    if solve_click > solve_request_cache[].click
        solve_request_cache[] = (click = solve_click, request = current_request)
    end
    solve_request = solve_request_cache[].request
end

# ╔═╡ 44dc579c-8a19-4911-9325-258168356306
begin
    if solve_request === nothing
        solve_bundle = (status = :idle, request = nothing, error = nothing)
    elseif solve_result_cache[].request == solve_request &&
           solve_result_cache[].click == solve_click
        solve_bundle = solve_result_cache[].result
    else
        computed_result = try
            run_schedule, run_root_delays, run_delay_cost_fn = generate_benchmark_instance(
                solve_request.legs;
                nb_scenarios = solve_request.scenarios,
                seed = solve_request.seed,
            )
            deterministic = let timed = @timed solve_aircraft_routing(run_schedule; silent = true)
                routes, objective, _ = timed.value
                valid_route_collection(routes) || error("deterministic solve was infeasible")
                (; routes, objective, runtime = timed.time)
            end
            column_generation = let timed = @timed stochastic_column_generation(
                    run_schedule,
                    deterministic.routes;
                    root_delays = run_root_delays,
                    delay_cost_function = run_delay_cost_fn,
                    max_nb_columns = 10_000,
                    tol = 1e-6,
                    silent = true,
                )
                (; value = timed.value, runtime = timed.time)
            end
            column_generation.value.feasible ||
                error("stochastic column generation was infeasible")
            diving = let timed = @timed diving_heuristic_with_backtracking!(
                    run_schedule,
                    column_generation.value.columns,
                    run_root_delays,
                    column_generation.value.dual_values,
                    TABOO_LIST_SIZE;
                    model_builder = highs_model,
                    delay_cost_function = run_delay_cost_fn,
                    silent = true,
                )
                routes, feasible = timed.value
                (; routes, feasible, runtime = timed.time)
            end
            (;
                status = :complete,
                request = solve_request,
                error = nothing,
                schedule = run_schedule,
                root_delays = run_root_delays,
                delay_cost_fn = run_delay_cost_fn,
                deterministic,
                column_generation,
                diving,
            )
        catch caught_error
            (; status = :failed, request = solve_request, error = sprint(showerror, caught_error))
        end
        solve_result_cache[] =
            (click = solve_click, request = solve_request, result = computed_result)
        solve_bundle = computed_result
    end
    (; status = solve_bundle.status, request = solve_bundle.request)
end

# ╔═╡ 1abfdfe2-aa67-4c6b-8423-8665b7868e2b
begin
    deterministic_result =
        hasproperty(solve_bundle, :deterministic) ? solve_bundle.deterministic : nothing
    column_generation_result =
        hasproperty(solve_bundle, :column_generation) ? solve_bundle.column_generation :
        nothing
    diving_result = hasproperty(solve_bundle, :diving) ? solve_bundle.diving : nothing
    solved_schedule = hasproperty(solve_bundle, :schedule) ? solve_bundle.schedule : nothing
    solved_root_delays =
        hasproperty(solve_bundle, :root_delays) ? solve_bundle.root_delays : nothing
    solved_delay_cost_fn =
        hasproperty(solve_bundle, :delay_cost_fn) ? solve_bundle.delay_cost_fn : nothing
    nothing
end

# ╔═╡ fbe91f9c-cd40-460c-abfe-a17e41f022e6
begin
    lp_lower_bound =
        column_generation_result === nothing ? missing : column_generation_result.value.obj
    diving_cost =
        (diving_result === nothing || !diving_result.feasible) ? missing :
        full_cost(
            diving_result.routes,
            solved_root_delays,
            solved_schedule;
            delay_cost_function = solved_delay_cost_fn,
        )
    deterministic_cost_insample =
        deterministic_result === nothing ? missing :
        full_cost(
            deterministic_result.routes,
            solved_root_delays,
            solved_schedule;
            delay_cost_function = solved_delay_cost_fn,
        )
    nothing
end

# ╔═╡ b2c5100c-6353-4dee-97fb-25d3816a8687
# The optimality gap between diving's feasible integer cost and the column generation lower
# bound, in percent. `missing` propagates automatically through arithmetic, so this stays a
# single expression whether or not the solve above has run yet.
optimality_gap_pct = 100 * (diving_cost - lp_lower_bound) / lp_lower_bound

# ╔═╡ 57a58f70-f350-4a06-a535-93227c56f2d0
# Draw a fresh, independent delay scenario matrix for the same schedule, to evaluate the
# solve out-of-sample. Stays missing until the solve above has run.
oos_root_delays = let
    result = missing
    if solved_schedule !== nothing
        result = generate_root_delays(
            solved_schedule;
            nb_scenarios = size(solved_root_delays, 1),
            seed = solve_bundle.request.seed + OOS_SEED_SHIFT,
        )
    end
    result
end

# ╔═╡ 9a8b4395-0f33-4c4a-a33f-4e130f07a593
begin
    det_oos_cost =
        (deterministic_result === nothing || ismissing(oos_root_delays)) ? missing :
        full_cost(
            deterministic_result.routes,
            oos_root_delays,
            solved_schedule;
            delay_cost_function = solved_delay_cost_fn,
        )
    stoch_oos_cost =
        (diving_result === nothing || !diving_result.feasible || ismissing(oos_root_delays)) ?
        missing :
        full_cost(
            diving_result.routes,
            oos_root_delays,
            solved_schedule;
            delay_cost_function = solved_delay_cost_fn,
        )
    nothing
end

# ╔═╡ eb28a39f-a19b-4211-9581-c61c81a3b09e
# The relative improvement of the stochastic (diving) cost over the deterministic cost,
# evaluated out-of-sample, in percent.
oos_improvement_pct = 100 * (det_oos_cost - stoch_oos_cost) / det_oos_cost

# ╔═╡ 49021f2b-305f-4cd4-b960-e5f3c8ec9bcb
begin
    status_text = if solve_bundle.status == :idle
        "**Status: idle.** Press \"Run deterministic + stochastic solve\" above to solve the current instance."
    elseif solve_bundle.status == :failed
        "**Status: failed.** " * solve_bundle.error
    elseif solve_bundle.request != current_request
        "**Status: stale.** Controls changed since the last solve, press the run button again to update this result."
    elseif diving_result === nothing || !diving_result.feasible
        "**Status: no feasible integer solution.** Diving could not find one for this instance, try a different seed or scenario count."
    else
        """
**Status: complete.**

Timing split: deterministic MIP $(format_seconds(deterministic_result.runtime)), column generation $(format_seconds(column_generation_result.runtime)), diving $(format_seconds(diving_result.runtime)), $(length(column_generation_result.value.columns)) columns generated.

In-sample (same scenarios the stochastic solve optimized against): deterministic full cost $(format_cost(deterministic_cost_insample)), diving full cost $(format_cost(diving_cost)), LP lower bound $(format_cost(lp_lower_bound)), optimality gap $(format_pct(optimality_gap_pct)).

Out-of-sample (a fresh scenario draw, shifted seed): deterministic full cost $(format_cost(det_oos_cost)), stochastic full cost $(format_cost(stoch_oos_cost)), improvement $(format_pct(oos_improvement_pct)).
"""
    end
    Markdown.parse(status_text)
end

# ╔═╡ bbfd0d86-88ad-4057-8681-8295ef9c9b27
if solve_bundle.status == :complete &&
   solve_bundle.request == current_request &&
   diving_result !== nothing &&
   diving_result.feasible
    plot_gantt(
        solved_schedule,
        deterministic_result.routes;
        root_delays = solved_root_delays,
        delay_cost_function = solved_delay_cost_fn,
        comparison_routes = diving_result.routes,
        primary_label = "Deterministic",
        comparison_label = "Stochastic (diving)",
        show_delays = true,
    )
else
    md"*Run the solve above to see the deterministic vs stochastic route comparison.*"
end

# ╔═╡ e6c684a8-7f6e-46cb-bde3-5d7e283b4199
begin
    if scale_click > scale_request_cache[].click
        scale_request_cache[] = (click = scale_click, request = current_scale_request)
    end
    scale_request = scale_request_cache[].request
end

# ╔═╡ 2f2fb971-efdb-4529-9c69-f8d986dd890a
begin
    if scale_request !== nothing && (
        scale_result_cache[].request != scale_request ||
        scale_result_cache[].click < scale_click
    )
        req = scale_request
        legs_rows = Any[]
        @progress "legs sweep" for nb in req.legs_cases
            @info "scaling experiment (legs sweep): solving $nb legs at $(req.scenarios) scenarios"
            push!(legs_rows, scale_point(nb, req.scenarios, req.seed))
        end
        scenario_rows = Any[]
        @progress "scenario sweep" for sc in req.scenario_cases
            @info "scaling experiment (scenario sweep): solving $(req.scenario_fixed_legs) legs at $sc scenarios"
            push!(scenario_rows, scale_point(req.scenario_fixed_legs, sc, req.seed))
        end
        scale_result_cache[] =
            (click = scale_click, request = scale_request, result = (; legs_rows, scenario_rows))
    end
    scale_bundle = scale_result_cache[].result
end

# ╔═╡ df4393fe-4196-4269-bbc5-9f4235bee90b
# Cheap: applies build_scaleup_row to the cached raw timings, so this re-renders the table
# and plots below without re-running the sweep above.
scale_table_rows =
    scale_bundle === nothing ? nothing :
    (
        legs_rows = [
            r.status == :ok ? build_scaleup_row(r) : missing for r in scale_bundle.legs_rows
        ],
        scenario_rows = [
            r.status == :ok ? build_scaleup_row(r) : missing for
            r in scale_bundle.scenario_rows
        ],
    )

# ╔═╡ ef9b449c-8726-4e54-87a5-c03e4fb6b816
begin
    table_text = if scale_bundle === nothing
        "*Press \"Run scaling experiment\" above.*"
    else
        header = "| legs | scenarios | runtime | columns | LP objective |\n|---|---|---|---|---|"
        body = join(
            (
                if raw_r.status == :error
                    "| $(raw_r.nb_legs) | $(raw_r.nb_scenarios) | **error**: $(raw_r.error) | | |"
                else
                    "| $(built_r.nb_legs) | $(built_r.nb_scenarios) | $(format_seconds(built_r.runtime)) | $(built_r.columns) | $(format_cost(built_r.objective)) |"
                end for
                (raw_r, built_r) in zip(scale_bundle.legs_rows, scale_table_rows.legs_rows)
            ),
            "\n",
        )
        header * "\n" * body
    end
    Markdown.parse(table_text)
end

# ╔═╡ 9ebcf7e7-de5b-426a-845a-f82c6fb2bffb
begin
    current_dataset_request =
        (size = control_int(dataset_size, 30, 10, 40), nb_legs = 50, nb_scenarios = 30)
    if dataset_click > dataset_request_cache[].click
        dataset_request_cache[] = (click = dataset_click, request = current_dataset_request)
    end
    dataset_request = dataset_request_cache[].request
end

# ╔═╡ 4b4077fa-0e04-4523-b71e-9989848cc1fd
try
    warmup_schedule, warmup_delays, warmup_cost_fn =
        generate_benchmark_instance(15; nb_scenarios = 3, seed = 1)
    warmup_routes, _, _ = solve_aircraft_routing(warmup_schedule; silent = true)
    warmup_cg = stochastic_column_generation(
        warmup_schedule,
        warmup_routes;
        root_delays = warmup_delays,
        delay_cost_function = warmup_cost_fn,
        max_nb_columns = 2_000,
        tol = 1e-6,
        silent = true,
    )
    warmup_cg.feasible && diving_heuristic_with_backtracking!(
        warmup_schedule,
        warmup_cg.columns,
        warmup_delays,
        warmup_cg.dual_values,
        TABOO_LIST_SIZE; # taboo list size
        model_builder = highs_model,
        delay_cost_function = warmup_cost_fn,
        silent = true,
    )
    "JIT warm-up done (15 legs, 3 scenarios)."
catch caught_error
    "JIT warm-up skipped (" * sprint(showerror, caught_error) * "), the first real solve below will absorb that one-time JIT cost instead."
end

# ╔═╡ 25a2e1f7-6097-4395-8abb-9d842920a023
"""
Plot the first `limit` legs of `schedule` on a single unlabeled row, before any solve has assigned them to an aircraft.
"""
function plot_unassigned_legs(schedule; limit = 20)
    shown = schedule.legs[1:min(limit, length(schedule.legs))]
    t0 = minimum(departure_time(l) for l in schedule.legs)
    fig = Figure(; size = (900, 220))
    ax = Axis(
        fig[1, 1];
        xlabel = "Time (hours since first departure)",
        yticks = ([1], ["unassigned"]),
        title = "$(length(shown)) legs before solving, no aircraft assigned yet",
    )
    rects = Rect2f[]
    for l in shown
        dep_h = Dates.value(departure_time(l) - t0) / 3_600_000
        arr_h = Dates.value(arrival_time(l) - t0) / 3_600_000
        push!(rects, Rect2f(dep_h, 0.6, max(arr_h - dep_h, 0.02), 0.8))
    end
    poly!(ax, rects; color = (:gray, 0.6))
    ylims!(ax, 0, 2)
    fig
end

# ╔═╡ b3dfbc0d-534a-4070-bf1e-d564b5e7a412
plot_unassigned_legs(schedule; limit = display_limit)

# ╔═╡ 4557d8a5-f8e3-4f9a-b049-c6928a63ba50
"""
Plot signed root delay bars (in minutes, around a zero line) for the first `nb_scenarios_shown` scenarios and the first `limit` legs of `root_delays`.
"""
function plot_root_delay_bars(root_delays; limit = 20, nb_scenarios_shown = 5)
    S, L = size(root_delays)
    shown_s = min(S, nb_scenarios_shown)
    shown_l = min(L, limit)
    fig = Figure(; size = (900, 320))
    ax = Axis(
        fig[1, 1];
        xlabel = "Leg index",
        ylabel = "Root delay (minutes)",
        title = "Root delays for the first $shown_s scenarios (negative delays are early legs)",
    )
    hlines!(ax, [0.0]; color = :black, linewidth = 1)
    width = 0.8 / shown_s
    for s = 1:shown_s
        xs = (1:shown_l) .+ (s - 1) * width .- 0.4
        ys = Float64.(root_delays[s, 1:shown_l])
        barplot!(ax, xs, ys; width, label = "scenario $s")
    end
    axislegend(ax; position = :rb)
    fig
end

# ╔═╡ 972df272-ae29-4927-a8bd-1e1ba8185695
plot_root_delay_bars(root_delays; limit = display_limit)

# ╔═╡ f7c86311-95cb-4d32-86a2-2ac88916e05b
if ismissing(oos_root_delays)
    md"*Run the solve above to see the out-of-sample delay scenarios.*"
else
    plot_root_delay_bars(oos_root_delays; limit = display_limit)
end

# ╔═╡ 537ac5a5-2302-42c4-872a-3100fe1928ba
"""
Plot root delay vs propagated arrival delay along `route`, one line pair per scenario, for the first `nb_scenarios_shown` scenarios.
"""
function plot_delay_propagation(route, root_delays, schedule; nb_scenarios_shown = 5)
    arrival_delays = propagate_delays_from_root_delays(route, root_delays, schedule)
    leg_positions =
        [idx for (idx, activity) in enumerate(route) if !is_maintenance(schedule, activity)]
    leg_ids = [route[idx] for idx in leg_positions]
    S = size(root_delays, 1)
    shown_s = min(S, nb_scenarios_shown)
    xs = 1:length(leg_positions)
    fig = Figure(; size = (900, 320))
    ax = Axis(
        fig[1, 1];
        xlabel = "Position along the route",
        ylabel = "Delay (minutes)",
        title = "Root delay vs propagated arrival delay along one route ($(length(xs)) legs, first $shown_s scenarios)",
    )
    for s = 1:shown_s
        lines!(ax, xs, Float64.(root_delays[s, leg_ids]); linestyle = :dash, label = "scenario $s root")
        lines!(ax, xs, Float64.(arrival_delays[s, leg_positions]); label = "scenario $s arrival")
    end
    axislegend(ax; position = :lt, nbanks = 2)
    fig
end

# ╔═╡ 320b03db-b38e-412e-899a-016338f8d96e
if solve_bundle.status == :complete &&
   solve_bundle.request == current_request &&
   diving_result !== nothing &&
   diving_result.feasible
    chosen_route = diving_result.routes[argmax(length.(diving_result.routes))]
    plot_delay_propagation(chosen_route, solved_root_delays, solved_schedule)
else
    md"*Run the solve above to see delay propagation along one route.*"
end

# ╔═╡ 10d9d026-e1af-49bb-b025-83812f3da80e
"""
Plot the lower bound and master LP upper bound history returned by `stochastic_column_generation`.
"""
function plot_cg_convergence(lb_history, ub_history)
    fig = Figure(; size = (700, 320))
    ax = Axis(
        fig[1, 1];
        xlabel = "Column generation iteration",
        ylabel = "Objective (cost units)",
        title = "Column generation convergence",
    )
    lines!(ax, 1:length(lb_history), lb_history; label = "lower bound")
    lines!(ax, 1:length(ub_history), ub_history; label = "upper bound (master LP)")
    axislegend(ax; position = :rb)
    fig
end

# ╔═╡ d3ece587-5270-4d21-ba86-76ccdbdc0f6f
if column_generation_result !== nothing &&
   column_generation_result.value.feasible &&
   !isempty(column_generation_result.value.lb_history)
    plot_cg_convergence(
        column_generation_result.value.lb_history, column_generation_result.value.ub_history
    )
else
    md"*Run the solve above to see the column generation convergence curve.*"
end

# ╔═╡ e352aa56-83bc-4fae-89b0-4de9190905eb
"""
Plot total runtime vs number of legs on a log y axis, from `live_rows` (from the legs sweep, field `.runtime`), any number of points.
"""
function plot_runtime_scaling(live_rows)
    fig = Figure(; size = (700, 380))
    ax = Axis(
        fig[1, 1];
        xlabel = "Number of legs",
        ylabel = "Total runtime (s, log scale)",
        yscale = log10,
        title = "Runtime vs instance size (deterministic + column generation + diving)",
    )
    if !isempty(live_rows)
        xs = [r.nb_legs for r in live_rows]
        ys = [r.runtime for r in live_rows]
        scatter!(ax, xs, ys; markersize = 16, label = "live (this machine, now)")
        lines!(ax, xs, ys)
        axislegend(ax; position = :lt)
    end
    fig
end

# ╔═╡ 46429172-4b57-40ef-98ba-e9cbe862379f
begin
    live_valid =
        scale_table_rows === nothing ? NamedTuple[] :
        [r for r in scale_table_rows.legs_rows if !ismissing(r)]
    if isempty(live_valid)
        md"*No runtime points yet, press \"Run scaling experiment\" above.*"
    else
        plot_runtime_scaling(live_valid)
    end
end

# ╔═╡ 85c700cd-7279-4db9-bc70-8e830c01a10c
"""
Plot total runtime vs scenario count, from the fixed-legs scenario sweep.
"""
function plot_scenario_scaling(rows)
    fig = Figure(; size = (700, 320))
    ax = Axis(
        fig[1, 1];
        xlabel = "Number of scenarios",
        ylabel = "Total runtime (s)",
        title = "Runtime vs scenario count (40 legs fixed)",
    )
    if !isempty(rows)
        xs = [r.nb_scenarios for r in rows]
        ys = [r.runtime for r in rows]
        scatter!(ax, xs, ys; markersize = 16)
        lines!(ax, xs, ys)
    end
    fig
end

# ╔═╡ c8727ee8-3919-44a4-8a06-059cf6689a3e
begin
    scenario_valid =
        scale_table_rows === nothing ? NamedTuple[] :
        [r for r in scale_table_rows.scenario_rows if !ismissing(r)]
    plot_scenario_scaling(scenario_valid)
end

# ╔═╡ a3ba3b1a-8ac4-4531-9e79-f45b944c74a0
"""
Build one labeled dataset entry for practice session 2: generate an instance with arc indices stored,
split its root delays into their departure and arrival components (the two halves practice session 2
needs to compute connection-slack features), run the expert solver, and return a named
tuple of StochasticTailAssignment objects.

The returned tuple carries two extra fields, `expert_seconds` and `delay_cost_fn`, used
only by the in-notebook summary above, they are stripped before the entries are saved to
`data/practice_session_1_dataset.jld2`.

Returns `nothing` if the expert's dive was infeasible for this instance (an expected,
recoverable outcome, not an error).
"""
function build_dataset_entry(seed; nb_legs_value = 50, nb_scenarios_value = 30)
    # store_arc_index=true is required, without it practice session 2 cannot compute features nor decode
    # arc labels from the saved routes.
    schedule, root_delays, delay_cost_fn = generate_benchmark_instance(
        nb_legs_value;
        nb_scenarios = nb_scenarios_value,
        seed,
        store_arc_index = true,
    )
    config = FeaturesConfig(; airports = schedule_airports(schedule))
    delay_model = build_delay_model(schedule; config)
    # seed + 100 is exactly generate_benchmark_instance's default delay_seed, which is what
    # makes departure_root_delays .+ arrival_root_delays == root_delays hold below.
    scenarios = DelayScenarios(
        schedule; nb_scenarios = nb_scenarios_value, config, seed = seed + 100
    )
    unmerged = sample_root_scenarios_unmerged(delay_model, scenarios)
    timed = @timed expert_routes(schedule, root_delays, delay_cost_fn)
    isnothing(timed.value) && return nothing
    return (;
        instance = schedule,
        departure_root_delays = unmerged.departure,
        arrival_root_delays = unmerged.arrival,
        routes = timed.value,
        seed,
        nb_legs = nb_legs_value,
        nb_scenarios = nb_scenarios_value,
        expert_seconds = timed.time,
        delay_cost_fn,
    )
end

# ╔═╡ 99e911b9-e16d-41e0-b088-bebab8df21b2
begin
    if dataset_request === nothing
        dataset_bundle = (status = :idle, request = nothing, error = nothing)
    elseif dataset_result_cache[].request == dataset_request &&
           dataset_result_cache[].click == dataset_click
        dataset_bundle = dataset_result_cache[].result
    else
        dataset_bundle = try
            req = dataset_request
            entries = Any[]
            nb_skipped = 0
            timed = @timed @progress "labeled dataset" for i = 1:req.size
                @info "labeled dataset: solving instance $i of $(req.size)"
                entry = build_dataset_entry(
                    DATASET_SEED_BASE + i;
                    nb_legs_value = req.nb_legs,
                    nb_scenarios_value = req.nb_scenarios,
                )
                if entry === nothing
                    nb_skipped += 1
                else
                    push!(entries, entry)
                end
            end
            (;
                status = :complete,
                request = req,
                error = nothing,
                entries,
                nb_skipped,
                runtime = timed.time,
            )
        catch caught_error
            (;
                status = :failed,
                request = dataset_request,
                entries = [],
                nb_skipped = 0,
                runtime = missing,
                error = sprint(showerror, caught_error),
            )
        end
        dataset_result_cache[] =
            (click = dataset_click, request = dataset_request, result = dataset_bundle)
    end
    (; status = dataset_bundle.status)
end

# ╔═╡ 77e6e356-705b-4e0d-a6b6-1438baace61e
begin
    dataset_path = joinpath(@__DIR__, "data", "practice_session_1_dataset.jld2")
    dataset_status_text = if dataset_bundle.status == :idle
        "**Status: idle.** Press \"Build and save the labeled dataset\" above."
    elseif dataset_bundle.status == :failed
        "**Status: failed.** " * dataset_bundle.error
    else
        mkpath(dirname(dataset_path))
        saved_entries = [
            (;
                instance = e.instance,
                departure_root_delays = e.departure_root_delays,
                arrival_root_delays = e.arrival_root_delays,
                routes = e.routes,
                seed = e.seed,
                nb_legs = e.nb_legs,
                nb_scenarios = e.nb_scenarios,
            ) for e in dataset_bundle.entries
        ]
        jldsave(dataset_path; dataset = saved_entries)
        n = length(saved_entries)
        mean_expert_seconds =
            n == 0 ? missing : sum(e.expert_seconds for e in dataset_bundle.entries) / n
        """
**Status: complete.** Solved $n instance$(n == 1 ? "" : "s") in $(format_seconds(dataset_bundle.runtime)) ($(format_seconds(mean_expert_seconds)) average per instance), $(dataset_bundle.nb_skipped) skipped (no feasible diving solution).

Saved to `data/practice_session_1_dataset.jld2` ($(round(filesize(dataset_path) / 1024^2; digits = 1)) MB).
"""
    end
    Markdown.parse(dataset_status_text)
end

# ╔═╡ Cell order:
# ╠═4853369b-c13c-40ff-b5e2-ec74fe882f1d
# ╠═ffb5b23f-da28-4eea-b1a5-edee6911dd25
# ╟─eeb13b61-d1d8-4824-9253-581a74450d3e
# ╟─eaaf49ba-fa58-4b27-a1a0-86a45c12e74a
# ╟─102fa231-68fd-4a23-8bd6-c2c85ab05b35
# ╟─34fc4248-9418-4376-b91c-6de852386d79
# ╟─9ac34107-c99d-4e46-adf5-c35ab1b146a1
# ╟─a26e1d41-4689-4dfe-a19c-72886dfe0cdd
# ╟─3ffe51f2-6c05-4456-9464-d5b04fa65052
# ╟─6df4c710-f90f-4cae-803a-eff3eb1d4efb
# ╟─7c35904a-617f-42a7-b96c-4e130371ab03
# ╟─ad00114d-cc92-4d4c-8699-23fea3b2ec88
# ╟─bd2b7df3-ca14-4672-bfd8-f5aed2969ab2
# ╟─21ef4150-020b-418b-b157-3c33b7c93fb6
# ╟─9670de72-5357-4067-b838-bf0eda1b0830
# ╟─3e9f123f-cf67-416b-b579-7b3eac3d1309
# ╠═76d7068f-0b6c-4ba7-848a-67e7e28c793e
# ╠═b3dfbc0d-534a-4070-bf1e-d564b5e7a412
# ╠═972df272-ae29-4927-a8bd-1e1ba8185695
# ╟─5411c3ab-b5a4-4334-9d41-883e808b171c
# ╟─dbd10df9-195e-4830-857a-9283d7e37e15
# ╟─1b996341-fbb7-4ac0-9384-8c73eea0f428
# ╟─44dc579c-8a19-4911-9325-258168356306
# ╟─1abfdfe2-aa67-4c6b-8423-8665b7868e2b
# ╟─fbe91f9c-cd40-460c-abfe-a17e41f022e6
# ╠═b2c5100c-6353-4dee-97fb-25d3816a8687
# ╠═57a58f70-f350-4a06-a535-93227c56f2d0
# ╟─f7c86311-95cb-4d32-86a2-2ac88916e05b
# ╟─9a8b4395-0f33-4c4a-a33f-4e130f07a593
# ╠═eb28a39f-a19b-4211-9581-c61c81a3b09e
# ╟─49021f2b-305f-4cd4-b960-e5f3c8ec9bcb
# ╟─fbea2d43-5577-4e31-912f-66b2ce80ae5e
# ╟─19824c74-077a-4fd3-ae54-a1eab59850d1
# ╟─8d27c291-cef1-4a2e-8b94-d8e6264bced9
# ╟─bbfd0d86-88ad-4057-8681-8295ef9c9b27
# ╟─320b03db-b38e-412e-899a-016338f8d96e
# ╟─d3ece587-5270-4d21-ba86-76ccdbdc0f6f
# ╟─4d89f83a-4baa-4119-8cba-23c2de5f9991
# ╟─8ba1c220-0488-46c6-9e5c-66322d2caa5c
# ╟─400669f2-cb60-467f-9b7d-492fb2079063
# ╟─645dc443-a89c-4696-a09e-446f1b8e4f79
# ╟─64be96b2-7016-4a70-b8fa-8aea5193770d
# ╠═0f78170c-b897-4c8b-b009-7591ebed0a6e
# ╟─3b80d3d6-a3fe-4101-97b5-7f899f8a37a9
# ╟─e6c684a8-7f6e-46cb-bde3-5d7e283b4199
# ╟─2ac3e869-99b1-47d3-95bf-92441c67f22d
# ╟─2f2fb971-efdb-4529-9c69-f8d986dd890a
# ╟─df4393fe-4196-4269-bbc5-9f4235bee90b
# ╟─ef9b449c-8726-4e54-87a5-c03e4fb6b816
# ╟─46429172-4b57-40ef-98ba-e9cbe862379f
# ╟─c8727ee8-3919-44a4-8a06-059cf6689a3e
# ╟─3a6f3005-fa01-4644-a6c6-dc749c776891
# ╟─d0a52953-1a3d-4e0e-93a9-f4ed4fbd4cbf
# ╟─6d079949-624a-46f3-b73e-4922ca4ced45
# ╟─75ffc900-562a-4d24-9bbb-a9cffab30228
# ╠═7dcd4eb4-b302-40c6-9af3-0b6788f9d007
# ╠═a145e517-1476-4065-9894-9168b08e1ffa
# ╟─eb0f60d7-c69c-4937-a432-46de60d55484
# ╟─81ead6c7-6530-4168-ae67-75dc5f4be8f1
# ╟─18c72211-47b7-4ed4-8edb-32de944452c2
# ╟─9ebcf7e7-de5b-426a-845a-f82c6fb2bffb
# ╟─99e911b9-e16d-41e0-b088-bebab8df21b2
# ╟─77e6e356-705b-4e0d-a6b6-1438baace61e
# ╟─45b7bfe0-be18-4486-802b-5b6b8338ce2e
# ╟─95c7ee9b-e6f1-415b-9b85-5c78c6216060
# ╟─29e92ee1-763b-463b-8c6d-726e7e7c4fc4
# ╟─4b4077fa-0e04-4523-b71e-9989848cc1fd
# ╟─25a2e1f7-6097-4395-8abb-9d842920a023
# ╟─4557d8a5-f8e3-4f9a-b049-c6928a63ba50
# ╟─537ac5a5-2302-42c4-872a-3100fe1928ba
# ╟─10d9d026-e1af-49bb-b025-83812f3da80e
# ╟─e352aa56-83bc-4fae-89b0-4de9190905eb
# ╟─85c700cd-7279-4db9-bc70-8e830c01a10c
# ╟─a3ba3b1a-8ac4-4531-9e79-f45b944c74a0
