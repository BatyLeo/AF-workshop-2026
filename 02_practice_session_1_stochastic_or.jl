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

# ╔═╡ d4ba8059-b450-43ea-b5b0-57b03359c7c0
TableOfContents()

# ╔═╡ eaaf49ba-fa58-4b27-a1a0-86a45c12e74a
md"""
# Practice session 1: stochastic OR
"""

# ╔═╡ 102fa231-68fd-4a23-8bd6-c2c85ab05b35
md"""
No Julia to write here, just some specific syntax worth recognizing in the code cells below.

- `(; a, b)` builds a **NamedTuple**, attribute access `nt.a` instead of `nt["a"]`.
- Arguments after `;` in a call are **keyword arguments**, e.g. `f(x; nb_scenarios=30)`.
- A dot before an operator or after a function name means **broadcasting**, element-wise, like NumPy.
- Julia allows **unicode identifiers**, so `λ`, `μ`, `ξ` are valid variable names.
"""

# ╔═╡ 20c61a29-b659-4088-8ad9-ea80524cbc1c
md"## 1. Summary of the part 1 slides"

# ╔═╡ 34fc4248-9418-4376-b91c-6de852386d79
md"""
### The tail assignment problem

- A **flight leg** is one scheduled flight between two airports.
- An **aircraft** flies a sequence of legs (field `immat`, the tail number).
- A **route** is an ordered sequence of compatible legs assigned to one aircraft, feasible only if turn times, mandatory maintenance checks, and mandatory connections are respected.
- A **delay scenario** is one random draw of delays, propagated forward through an aircraft's rotation.
- A **column** is a route as a decision variable in the master problem, its cost combines operational cost (fuel, connections) with the average delay cost simulated over all scenarios.
- The tail assignment problem can be written as a compact leg-level MIP, or reformulated with one variable per route, the formulation this notebook uses.
"""

# ╔═╡ a4b95842-d3be-4a97-9824-caa0bf542eb6
md"""
### Delay propagation and the stochastic problem

Each leg has two intrinsic **root delays**, one at departure and one at arrival, both exogenous (weather, crew, air traffic), not caused by the aircraft's own rotation.
Delay propagates forward along a route: an arrival delay adds the arrival root delay to the departure delay, and the next leg's departure delay adds its own departure root delay on top of whatever survives the connection slack $\omega$ (scheduled connection time minus the minimum turn time):

```math
\xi^a_\ell = \xi^d_\ell + \varepsilon^a_\ell, \qquad
\xi^d_{\ell_j} = \max\bigl(\xi^a_{\ell_{j-1}} - \omega_{\ell_{j-1}, \ell_j}, 0\bigr) + \varepsilon^d_{\ell_j}.
```

A **scenario** is one joint draw of every leg's root delays, propagated through every aircraft's route.
The package merges the departure and arrival root delays into a single root delay per leg, which is what `root_delays` holds.
The stochastic objective averages the resulting delay cost over all scenarios and adds it to the fixed operational cost, this sample-average approximation is what each route's precomputed cost captures.
"""

# ╔═╡ 6a0422e4-24e5-4e30-8120-962178c8aacc
md"""
### Modeling intrinsic delays

Root delays are modeled as (log-)normal distributions whose mean and standard deviation come from a neural network applied to each leg's departure or arrival features.
The network is fit by maximizing the log-likelihood of the observed delays under this model, so ordinary gradient descent training applies.
In this notebook `build_delay_model` substitutes a handcrafted synthetic version of this model, its parameters are set rather than learned.
"""

# ╔═╡ b3544b6d-f56d-4837-9eaf-985c4b1d1275
md"""
### Column generation

Column generation generates only the routes worth adding to the master problem, instead of enumerating every feasible route.
"""

# ╔═╡ 9ac34107-c99d-4e46-adf5-c35ab1b146a1
md"""
#### The master problem is a set-partitioning problem

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
#### Pricing is a stochastic shortest path

Column generation adds a route only when it improves the master objective.
The master's dual prices (one per leg, one per aircraft) turn this into a search for the route with the smallest reduced cost:

```math
\min_{r \in \mathcal{R}(a)} \ c_r^a - \sum_{\ell \in r} \lambda_\ell - \mu_a.
```

A new column is added whenever the minimum reduced cost is negative, until no aircraft can find one.
"""

# ╔═╡ 6df4c710-f90f-4cae-803a-eff3eb1d4efb
keyconcept(
    "Pricing is a stochastic shortest path",
    md"Because the route cost takes delay propagation and expected delay into account, minimizing reduced cost is a resource-constrained shortest path problem on the schedule graph.",
)

# ╔═╡ 7c35904a-617f-42a7-b96c-4e130371ab03
md"""
#### Scenarios multiply the pricing cost

Each label has one coordinate per scenario, so more scenarios means costlier labels and weaker dominance (a label must win on every coordinate to dominate another):

```math
\text{pricing cost per aircraft} \ \approx \ O\bigl(S \times \text{number of surviving labels}(S)\bigr).
```

The scaling experiment below measures this directly.
"""

# ╔═╡ ad00114d-cc92-4d4c-8699-23fea3b2ec88
md"""
### From fractional to integer: diving

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
    md"Column generation dominates the total runtime, compare the three `@time` outputs in section 2 below: diving adds one shrinking column generation re-solve per aircraft on top of the first, full-size run.",
)

# ╔═╡ 964a2456-a9d3-4372-a6b3-0ca39a52fef5
md"## 2. The StochasticTailAssignment.jl package"

# ╔═╡ 0b5aa7a8-f3df-48ee-8a52-5e48ed5820c8
md"""
The [StochasticTailAssignment.jl](https://github.com/BatyLeo/StochasticTailAssignment.jl) package implements the methods from part 1.
It generates synthetic benchmark instances together with a delay model.
It solves the deterministic tail assignment MILP.
It also solves the stochastic version with column generation, a restricted master heuristic, and a diving heuristic.
This session only calls its functions, practice session 2 builds on top of it.
"""

# ╔═╡ 0dc82786-5981-44ca-8b00-25e741d425c8
md"### Generating an instance"

# ╔═╡ 21ef4150-020b-418b-b157-3c33b7c93fb6
@bind instance_params confirm(
    PlutoUI.combine() do Child
        md"""
        Number of legs: $(Child(:legs, Slider(20:10:100; default = 80, show_value = true)))

        Number of scenarios: $(Child(:scenarios, Slider(5:5:30; default = 30, show_value = true)))

        Seed: $(Child(:seed, Slider(1:100; default = 13, show_value = true)))
        """
    end;
    label = "Generate and solve",
)

# ╔═╡ cac3f496-1eaf-4ed4-9543-fe75ef89dc77
instance_params

# ╔═╡ c1f092b6-45d3-4541-8748-f44026d50154
md"Move the sliders, then click the button: the instance is generated and solved below, which takes a few seconds."

# ╔═╡ afb27cae-8369-4418-8642-1b39ffa92393
md"""
display limit: $(@bind display_limit PlutoUI.Slider(5:5:40; default=20, show_value=true))
"""

# ╔═╡ 6d9c9ea0-3443-4d63-a847-509872c5544e
md"The `generate_benchmark_instance` method allows to build a random instance:"

# ╔═╡ 76d7068f-0b6c-4ba7-848a-67e7e28c793e
schedule, root_delays, delay_cost_fn = generate_benchmark_instance(
    instance_params.legs;
    nb_scenarios=instance_params.scenarios,
    seed=instance_params.seed,
)

# ╔═╡ 4fe844a3-a4fb-4cc2-bf1b-3d1b4b2a25a6
md"### Inspecting an instance"

# ╔═╡ d146c758-83ce-4c5c-becb-8552f674da6e
nb_legs(schedule)

# ╔═╡ e507a7b4-eb56-489e-bf6d-46d2037b7bb8
nb_immats(schedule)

# ╔═╡ cf56ba51-2c6b-4514-98bf-82d012732a18
root_delays

# ╔═╡ 3b6b6dbe-a760-4012-9e7c-ec7cd325b4df
md"### Solving the deterministic problem"

# ╔═╡ dbd10df9-195e-4830-857a-9283d7e37e15
current_instance = (;
    schedule,
    root_delays,
    delay_cost_fn,
    params = (;
        legs = instance_params.legs,
        scenarios = instance_params.scenarios,
        seed = instance_params.seed,
    ),
)

# ╔═╡ af1a6ddb-e5eb-4134-a0e1-9f5125c3a857
deterministic_routes, _, _ = @time solve_aircraft_routing(current_instance.schedule; silent = true)

# ╔═╡ c9a57106-e1eb-415e-b70b-f7c29f5d93ba
deterministic_cost_insample =
    isempty(deterministic_routes) ? missing :
    full_cost(
        deterministic_routes,
        current_instance.root_delays,
        current_instance.schedule;
        delay_cost_function = current_instance.delay_cost_fn,
    )

# ╔═╡ aed62408-3b9a-425d-b64c-365b5eca9307
if isempty(deterministic_routes)
    md"*No deterministic routes: the deterministic MIP was infeasible for this instance.*"
else
    plot_gantt(
        current_instance.schedule,
        deterministic_routes;
        root_delays = current_instance.root_delays,
        delay_cost_function = current_instance.delay_cost_fn,
        show_delays = true,
    )
end

# ╔═╡ 8d27c291-cef1-4a2e-8b94-d8e6264bced9
md"""
The "Show delays" bar coloring is only interactive in a live Pluto session, on GLMakie or WGLMakie.
"""

# ╔═╡ 7bda92d3-1afc-42ca-99ac-9d640f130feb
md"### Solving with column generation"

# ╔═╡ 99bc39c6-154f-498f-b6a1-065cf1e352de
column_generation_result =
    isempty(deterministic_routes) ? missing :
    @time stochastic_column_generation(
        current_instance.schedule,
        deterministic_routes;
        root_delays = current_instance.root_delays,
        delay_cost_function = current_instance.delay_cost_fn,
        max_nb_columns = 10_000,
        tol = 1e-6,
        silent = true,
    )

# ╔═╡ fbe91f9c-cd40-460c-abfe-a17e41f022e6
lp_lower_bound =
    (ismissing(column_generation_result) || !column_generation_result.feasible) ? missing :
    column_generation_result.obj

# ╔═╡ 4d89f83a-4baa-4119-8cba-23c2de5f9991
md"""
Both curves converge to `lp_lower_bound`, the LP optimum, a lower bound on the integer (diving) solution's cost.
"""

# ╔═╡ 3e877022-150b-4b7d-8f12-158de176b2a7
md"""
### Solving with the restricted master heuristic

Once column generation stops, the simplest way to get an integer solution is to solve the restricted master problem over the generated columns as a MILP.
This is the restricted master heuristic.
It is exact on the column pool, but the MILP can become slow on large instances, which the scaling experiment below shows.
"""

# ╔═╡ 5ceac5c4-4c41-48b2-ab72-d02a69c7ec7a
restricted_routes, restricted_objective =
    (ismissing(column_generation_result) || !column_generation_result.feasible) ?
    (Route[], missing) :
    @time stochastic_column_heuristic(
        current_instance.schedule,
        column_generation_result.columns,
        current_instance.root_delays;
        model_builder = highs_model,
        delay_cost_function = current_instance.delay_cost_fn,
        silent = true,
        warm_start = true,
    )

# ╔═╡ 47ec0049-8fc5-41fe-8558-2633125d59c3
restricted_cost =
    isempty(restricted_routes) ? missing :
    full_cost(
        restricted_routes,
        current_instance.root_delays,
        current_instance.schedule;
        delay_cost_function = current_instance.delay_cost_fn,
    )

# ╔═╡ 4391ff9d-4076-4bed-8be8-d1dbe97b531b
restricted_gap_pct = 100 * (restricted_cost - lp_lower_bound) / lp_lower_bound

# ╔═╡ b09f091b-57ed-424b-beff-7fadd6e6abf2
md"### Solving with diving"

# ╔═╡ 7dcd4eb4-b302-40c6-9af3-0b6788f9d007
const TABOO_LIST_SIZE = 15

# ╔═╡ fa3a3282-50da-4dae-8331-5bf1f16b70fd
diving_routes, diving_feasible =
    (ismissing(column_generation_result) || !column_generation_result.feasible) ?
    (Route[], false) :
    @time diving_heuristic_with_backtracking!(
        current_instance.schedule,
        column_generation_result.columns,
        current_instance.root_delays,
        column_generation_result.dual_values,
        TABOO_LIST_SIZE;
        model_builder = highs_model,
        delay_cost_function = current_instance.delay_cost_fn,
        silent = true,
    )

# ╔═╡ fd5fc431-7d01-4e90-9a1e-923d721510c3
diving_cost =
    diving_feasible ?
    full_cost(
        diving_routes,
        current_instance.root_delays,
        current_instance.schedule;
        delay_cost_function = current_instance.delay_cost_fn,
    ) : missing

# ╔═╡ b2c5100c-6353-4dee-97fb-25d3816a8687
# The optimality gap between diving's feasible integer cost and the column generation lower
# bound, in percent. `missing` propagates automatically through arithmetic, so this stays a
# single expression whether or not diving found an integer solution.
optimality_gap_pct = 100 * (diving_cost - lp_lower_bound) / lp_lower_bound

# ╔═╡ fbea2d43-5577-4e31-912f-66b2ce80ae5e
question_box(
    md"""
Which integer method, restricted master or diving, is closer to the LP lower bound on this instance?
Which one was faster, based on the `@time` output above?
""",
)

# ╔═╡ ae78d44d-524d-4f94-b955-6073d090193f
md"""
### Out-of-sample evaluation

The routes chosen above are re-evaluated on a fresh set of delay scenarios that the solver never saw during optimization.
"""

# ╔═╡ 19824c74-077a-4fd3-ae54-a1eab59850d1
question_box(
    md"""
Does the in-sample improvement hold up on the fresh out-of-sample scenarios?
Why might a small scenario count make this comparison noisy?
""",
)

# ╔═╡ bbfd0d86-88ad-4057-8681-8295ef9c9b27
if diving_feasible
    plot_gantt(
        current_instance.schedule,
        deterministic_routes;
        root_delays = current_instance.root_delays,
        delay_cost_function = current_instance.delay_cost_fn,
        comparison_routes = diving_routes,
        primary_label = "Deterministic",
        comparison_label = "Stochastic (diving)",
        show_delays = true,
    )
else
    md"*No integer solution: diving failed for this instance, so there is no stochastic route set to compare.*"
end

# ╔═╡ 8ba1c220-0488-46c6-9e5c-66322d2caa5c
md"""
### Optional: scaling experiment

Repeats the solve at several instance sizes, live: a legs sweep (30, 50, 65, 80, up to your chosen maximum) at fixed scenarios, and a scenario sweep (5, 15, 30) at 40 legs fixed.
Each point compares three methods: the deterministic MILP, column generation followed by the restricted master heuristic, and column generation followed by diving.
The table and plots below report runtime and solution quality (gap to the LP bound) for each.
"""

# ╔═╡ 400669f2-cb60-467f-9b7d-492fb2079063
tip(md"Both sweeps run behind the same button, a larger maximum takes longer.")

# ╔═╡ 645dc443-a89c-4696-a09e-446f1b8e4f79
md"""
largest instance in the sweep: $(@bind scale_max_legs PlutoUI.Slider([80, 100, 120, 150]; default=100, show_value=true))

run scaling experiment: $(@bind scale_click PlutoUI.CounterButton("Run scaling experiment"))
"""

# ╔═╡ 64be96b2-7016-4a70-b8fa-8aea5193770d
md"""
The scaling experiment takes under a minute up to 80 legs, about two minutes at 100, about five at 120, about ten at 150.
The restricted master MILP is capped at 60 seconds per instance, so the sweep can take longer than before at 120 and 150 legs.
"""

# ╔═╡ 0f78170c-b897-4c8b-b009-7591ebed0a6e
# Summarizes one raw sweep point: per-method wall time, the number of generated columns,
# and each method's gap to the column generation LP bound, in percent.
function build_scaleup_row(raw)
    return (
        nb_legs = raw.nb_legs,
        nb_scenarios = raw.nb_scenarios,
        det_time = raw.det_time,
        stochastic_restricted_time = raw.cg_time + raw.restricted_time,
        stochastic_diving_time = raw.cg_time + raw.diving_time,
        columns = raw.nb_columns,
        det_gap_pct = 100 * (raw.det_cost - raw.lp_bound) / raw.lp_bound,
        restricted_gap_pct = 100 * (raw.restricted_cost - raw.lp_bound) / raw.lp_bound,
        diving_gap_pct = 100 * (raw.diving_cost - raw.lp_bound) / raw.lp_bound,
    )
end

# ╔═╡ 3a6f3005-fa01-4644-a6c6-dc749c776891
question_box(
    md"""
Compare the three methods above in runtime and solution quality as instances grow.
Which method would you pick at 150 legs, and why?
""",
)

# ╔═╡ d0a52953-1a3d-4e0e-93a9-f4ed4fbd4cbf
md"""
## 3. Building a labeled dataset for practice session 2

Practice session 2 trains a neural network to imitate the solver you just ran, from labeled examples: instances paired with the routes this part's solver chose.
The button below labels a chosen number of instances (50 legs, 30 scenarios, one per seed), fixed so practice session 2 always sees this size.
Each saved example holds the schedule, its delay scenarios, and the expert routes.
"""

# ╔═╡ 75ffc900-562a-4d24-9bbb-a9cffab30228
question_box(
    md"""
`expert_routes` below is the expert solver the dataset button uses to label every instance.
Identify its three stages, warm start, column generation, diving, each feeding the next.
""",
)

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
    # (1000+instance_params.seed) in this notebook. Practice session 2 does not reuse this range: each of its
    # instances draws its own schedule seed with `rand(rng, 1:1_000_000)`, so an exact
    # collision with practice session 1's 2001 to 2040 is possible in principle but negligible in
    # practice (about 40 in 1,000,000 per draw).
    const DATASET_SEED_BASE = 2000
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

# ╔═╡ f2fd2a8d-3a1a-400e-9ef4-af7a32590a3c
md"""
| method | in-sample cost | gap to LP bound |
|---|---|---|
| Deterministic | $(format_cost(deterministic_cost_insample)) | $(format_pct(100 * (deterministic_cost_insample - lp_lower_bound) / lp_lower_bound)) |
| Restricted master | $(format_cost(restricted_cost)) | $(format_pct(restricted_gap_pct)) |
| Diving | $(format_cost(diving_cost)) | $(format_pct(optimality_gap_pct)) |
"""

# ╔═╡ 57a58f70-f350-4a06-a535-93227c56f2d0
# Draw a fresh, independent delay scenario matrix for the same schedule, to evaluate the
# solve out-of-sample.
oos_root_delays = generate_root_delays(
    current_instance.schedule;
    nb_scenarios = size(current_instance.root_delays, 1),
    seed = current_instance.params.seed + OOS_SEED_SHIFT,
)

# ╔═╡ 8374d7c7-6b0d-4e7e-b0b8-c1ffd1de1d42
det_oos_cost =
    isempty(deterministic_routes) ? missing :
    full_cost(
        deterministic_routes,
        oos_root_delays,
        current_instance.schedule;
        delay_cost_function = current_instance.delay_cost_fn,
    )

# ╔═╡ 220fd06c-dd1b-4533-9dad-3b1f32fcdd47
stoch_oos_cost =
    !diving_feasible ? missing :
    full_cost(
        diving_routes,
        oos_root_delays,
        current_instance.schedule;
        delay_cost_function = current_instance.delay_cost_fn,
    )

# ╔═╡ eb28a39f-a19b-4211-9581-c61c81a3b09e
# The relative improvement of the stochastic (diving) cost over the deterministic cost,
# evaluated out-of-sample, in percent.
oos_improvement_pct = 100 * (det_oos_cost - stoch_oos_cost) / det_oos_cost

# ╔═╡ 62cc2c7c-3e84-476a-8df4-f2f3c86c85e1
restricted_oos_cost =
    isempty(restricted_routes) ? missing :
    full_cost(
        restricted_routes,
        oos_root_delays,
        current_instance.schedule;
        delay_cost_function = current_instance.delay_cost_fn,
    )

# ╔═╡ 49021f2b-305f-4cd4-b960-e5f3c8ec9bcb
begin
    status_text = if !diving_feasible
        "**Status: no integer solution.** The deterministic MIP, column generation, or the dive failed on this instance, try a different seed or scenario count."
    else
        """
**Status: complete.** $(length(column_generation_result.columns)) columns generated.

In-sample (same scenarios the stochastic solve optimized against): deterministic full cost $(format_cost(deterministic_cost_insample)), restricted master full cost $(format_cost(restricted_cost)), diving full cost $(format_cost(diving_cost)), LP lower bound $(format_cost(lp_lower_bound)), optimality gap $(format_pct(optimality_gap_pct)).

Out-of-sample (a fresh scenario draw, shifted seed): deterministic full cost $(format_cost(det_oos_cost)), restricted master full cost $(format_cost(restricted_oos_cost)), diving full cost $(format_cost(stoch_oos_cost)), improvement (deterministic vs diving) $(format_pct(oos_improvement_pct)).
"""
    end
    Markdown.parse(status_text)
end

# ╔═╡ 3b80d3d6-a3fe-4101-97b5-7f899f8a37a9
current_scale_request = (
    legs_cases = scaleup_legs_cases(scale_max_legs),
    scenario_cases = (5, 15, 30),
    scenario_fixed_legs = 40,
    scenarios = instance_params.scenarios,
    seed = instance_params.seed,
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
        restricted_timed = @timed stochastic_column_heuristic(
            s,
            cg_timed.value.columns,
            d;
            model_builder = highs_model,
            delay_cost_function = c,
            silent = true,
            warm_start = true,
            time_limit = 60,
        )
        restricted_routes_point, _ = restricted_timed.value
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
        dive_routes, dive_feasible = dive_timed.value
        dive_feasible || error("diving was infeasible")
        cost_of(routes) = full_cost(routes, d, s; delay_cost_function = c)
        (
            nb_legs = nb,
            nb_scenarios = nb_scenarios_value,
            status = :ok,
            error = nothing,
            det_time = det_timed.time,
            cg_time = cg_timed.time,
            restricted_time = restricted_timed.time,
            diving_time = dive_timed.time,
            nb_columns = length(cg_timed.value.columns),
            lp_bound = cg_timed.value.obj,
            det_cost = cost_of(det_routes),
            restricted_cost = cost_of(restricted_routes_point),
            diving_cost = cost_of(dive_routes),
        )
    catch caught_error
        (
            nb_legs = nb,
            nb_scenarios = nb_scenarios_value,
            status = :error,
            error = sprint(showerror, caught_error),
            det_time = missing,
            cg_time = missing,
            restricted_time = missing,
            diving_time = missing,
            nb_columns = missing,
            lp_bound = missing,
            det_cost = missing,
            restricted_cost = missing,
            diving_cost = missing,
        )
    end
end

# ╔═╡ 29e92ee1-763b-463b-8c6d-726e7e7c4fc4
begin
    scale_request_cache = Ref{Any}((click = 0, request = nothing))
    scale_result_cache = Ref{Any}((click = 0, request = nothing, result = nothing))
    dataset_request_cache = Ref{Any}((click = 0, request = nothing))
    dataset_result_cache = Ref{Any}((click = 0, request = nothing, result = nothing))
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
        header = "| legs | scenarios | deterministic | CG + restricted master | CG + diving | columns | gap deterministic | gap restricted master | gap diving |\n|---|---|---|---|---|---|---|---|---|"
        body = join(
            (
                if raw_r.status == :error
                    "| $(raw_r.nb_legs) | $(raw_r.nb_scenarios) | **error**: $(raw_r.error) | | | | | | |"
                else
                    "| $(built_r.nb_legs) | $(built_r.nb_scenarios) | $(format_seconds(built_r.det_time)) | $(format_seconds(built_r.stochastic_restricted_time)) | $(format_seconds(built_r.stochastic_diving_time)) | $(built_r.columns) | $(format_pct(built_r.det_gap_pct)) | $(format_pct(built_r.restricted_gap_pct)) | $(format_pct(built_r.diving_gap_pct)) |"
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
        (size = dataset_size, nb_legs = 50, nb_scenarios = 30)
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
plot_unassigned_legs(schedule; limit=display_limit)

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
plot_root_delay_bars(root_delays; limit=display_limit)

# ╔═╡ f7c86311-95cb-4d32-86a2-2ac88916e05b
plot_root_delay_bars(oos_root_delays; limit = display_limit)

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
if diving_feasible
    chosen_route = diving_routes[argmax(length.(diving_routes))]
    plot_delay_propagation(chosen_route, current_instance.root_delays, current_instance.schedule)
else
    md"*No integer solution: diving failed for this instance, so there is no route to show delay propagation along.*"
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
if !ismissing(column_generation_result) &&
   column_generation_result.feasible &&
   !isempty(column_generation_result.lb_history)
    plot_cg_convergence(
        column_generation_result.lb_history, column_generation_result.ub_history
    )
else
    md"*No column generation result: the deterministic solve was infeasible for this instance.*"
end

# ╔═╡ e352aa56-83bc-4fae-89b0-4de9190905eb
"""
Plot runtime vs number of legs on a log y axis, one series per method, from `live_rows`
(from the legs sweep), any number of points.
"""
function plot_runtime_scaling(live_rows)
    fig = Figure(; size = (700, 380))
    ax = Axis(
        fig[1, 1];
        xlabel = "Number of legs",
        ylabel = "Runtime (s, log scale)",
        yscale = log10,
        title = "Runtime vs instance size",
    )
    if !isempty(live_rows)
        xs = [r.nb_legs for r in live_rows]
        series = (
            ("Deterministic", [r.det_time for r in live_rows]),
            ("CG + restricted master", [r.stochastic_restricted_time for r in live_rows]),
            ("CG + diving", [r.stochastic_diving_time for r in live_rows]),
        )
        for (label, ys) in series
            scatter!(ax, xs, ys; markersize = 16, label)
            lines!(ax, xs, ys)
        end
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

# ╔═╡ baa9dee5-9010-43a2-9cea-eee1424ecea2
"""
Plot the gap to the LP bound (percent) vs number of legs, one series per method, from
`live_rows` (from the legs sweep), linear y axis.
"""
function plot_gap_scaling(live_rows)
    fig = Figure(; size = (700, 380))
    ax = Axis(
        fig[1, 1];
        xlabel = "Number of legs",
        ylabel = "Gap to LP bound (%)",
        title = "Solution quality vs instance size",
    )
    if !isempty(live_rows)
        xs = [r.nb_legs for r in live_rows]
        series = (
            ("Deterministic", [r.det_gap_pct for r in live_rows]),
            ("CG + restricted master", [r.restricted_gap_pct for r in live_rows]),
            ("CG + diving", [r.diving_gap_pct for r in live_rows]),
        )
        for (label, ys) in series
            scatter!(ax, xs, ys; markersize = 16, label)
            lines!(ax, xs, ys)
        end
        axislegend(ax; position = :lt)
    end
    fig
end

# ╔═╡ 265b4419-9ee1-41e4-b056-e1efe7bc62ec
begin
    live_valid_gap =
        scale_table_rows === nothing ? NamedTuple[] :
        [r for r in scale_table_rows.legs_rows if !ismissing(r)]
    if isempty(live_valid_gap)
        md"*No gap points yet, press \"Run scaling experiment\" above.*"
    else
        plot_gap_scaling(live_valid_gap)
    end
end

# ╔═╡ 85c700cd-7279-4db9-bc70-8e830c01a10c
"""
Plot runtime vs scenario count, one series per method, from the fixed-legs scenario sweep.
"""
function plot_scenario_scaling(rows)
    fig = Figure(; size = (700, 320))
    ax = Axis(
        fig[1, 1];
        xlabel = "Number of scenarios",
        ylabel = "Runtime (s)",
        title = "Runtime vs scenario count (40 legs fixed)",
    )
    if !isempty(rows)
        xs = [r.nb_scenarios for r in rows]
        series = (
            ("Deterministic", [r.det_time for r in rows]),
            ("CG + restricted master", [r.stochastic_restricted_time for r in rows]),
            ("CG + diving", [r.stochastic_diving_time for r in rows]),
        )
        for (label, ys) in series
            scatter!(ax, xs, ys; markersize = 16, label)
            lines!(ax, xs, ys)
        end
        axislegend(ax; position = :lt)
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
# ╟─d4ba8059-b450-43ea-b5b0-57b03359c7c0
# ╟─eaaf49ba-fa58-4b27-a1a0-86a45c12e74a
# ╟─102fa231-68fd-4a23-8bd6-c2c85ab05b35
# ╟─20c61a29-b659-4088-8ad9-ea80524cbc1c
# ╟─34fc4248-9418-4376-b91c-6de852386d79
# ╟─a4b95842-d3be-4a97-9824-caa0bf542eb6
# ╟─6a0422e4-24e5-4e30-8120-962178c8aacc
# ╟─b3544b6d-f56d-4837-9eaf-985c4b1d1275
# ╟─9ac34107-c99d-4e46-adf5-c35ab1b146a1
# ╟─a26e1d41-4689-4dfe-a19c-72886dfe0cdd
# ╟─3ffe51f2-6c05-4456-9464-d5b04fa65052
# ╟─6df4c710-f90f-4cae-803a-eff3eb1d4efb
# ╟─7c35904a-617f-42a7-b96c-4e130371ab03
# ╟─ad00114d-cc92-4d4c-8699-23fea3b2ec88
# ╟─bd2b7df3-ca14-4672-bfd8-f5aed2969ab2
# ╟─964a2456-a9d3-4372-a6b3-0ca39a52fef5
# ╟─0b5aa7a8-f3df-48ee-8a52-5e48ed5820c8
# ╟─0dc82786-5981-44ca-8b00-25e741d425c8
# ╟─21ef4150-020b-418b-b157-3c33b7c93fb6
# ╠═cac3f496-1eaf-4ed4-9543-fe75ef89dc77
# ╟─c1f092b6-45d3-4541-8748-f44026d50154
# ╟─afb27cae-8369-4418-8642-1b39ffa92393
# ╟─6d9c9ea0-3443-4d63-a847-509872c5544e
# ╠═76d7068f-0b6c-4ba7-848a-67e7e28c793e
# ╟─4fe844a3-a4fb-4cc2-bf1b-3d1b4b2a25a6
# ╠═d146c758-83ce-4c5c-becb-8552f674da6e
# ╠═e507a7b4-eb56-489e-bf6d-46d2037b7bb8
# ╠═cf56ba51-2c6b-4514-98bf-82d012732a18
# ╠═b3dfbc0d-534a-4070-bf1e-d564b5e7a412
# ╠═972df272-ae29-4927-a8bd-1e1ba8185695
# ╟─3b6b6dbe-a760-4012-9e7c-ec7cd325b4df
# ╟─dbd10df9-195e-4830-857a-9283d7e37e15
# ╠═af1a6ddb-e5eb-4134-a0e1-9f5125c3a857
# ╠═c9a57106-e1eb-415e-b70b-f7c29f5d93ba
# ╟─aed62408-3b9a-425d-b64c-365b5eca9307
# ╟─8d27c291-cef1-4a2e-8b94-d8e6264bced9
# ╟─7bda92d3-1afc-42ca-99ac-9d640f130feb
# ╠═99bc39c6-154f-498f-b6a1-065cf1e352de
# ╟─fbe91f9c-cd40-460c-abfe-a17e41f022e6
# ╟─d3ece587-5270-4d21-ba86-76ccdbdc0f6f
# ╟─4d89f83a-4baa-4119-8cba-23c2de5f9991
# ╟─3e877022-150b-4b7d-8f12-158de176b2a7
# ╠═5ceac5c4-4c41-48b2-ab72-d02a69c7ec7a
# ╠═47ec0049-8fc5-41fe-8558-2633125d59c3
# ╠═4391ff9d-4076-4bed-8be8-d1dbe97b531b
# ╟─b09f091b-57ed-424b-beff-7fadd6e6abf2
# ╠═7dcd4eb4-b302-40c6-9af3-0b6788f9d007
# ╠═fa3a3282-50da-4dae-8331-5bf1f16b70fd
# ╠═fd5fc431-7d01-4e90-9a1e-923d721510c3
# ╠═b2c5100c-6353-4dee-97fb-25d3816a8687
# ╟─f2fd2a8d-3a1a-400e-9ef4-af7a32590a3c
# ╟─fbea2d43-5577-4e31-912f-66b2ce80ae5e
# ╟─ae78d44d-524d-4f94-b955-6073d090193f
# ╠═57a58f70-f350-4a06-a535-93227c56f2d0
# ╟─f7c86311-95cb-4d32-86a2-2ac88916e05b
# ╠═8374d7c7-6b0d-4e7e-b0b8-c1ffd1de1d42
# ╠═220fd06c-dd1b-4533-9dad-3b1f32fcdd47
# ╠═62cc2c7c-3e84-476a-8df4-f2f3c86c85e1
# ╠═eb28a39f-a19b-4211-9581-c61c81a3b09e
# ╟─19824c74-077a-4fd3-ae54-a1eab59850d1
# ╟─49021f2b-305f-4cd4-b960-e5f3c8ec9bcb
# ╟─bbfd0d86-88ad-4057-8681-8295ef9c9b27
# ╟─320b03db-b38e-412e-899a-016338f8d96e
# ╟─8ba1c220-0488-46c6-9e5c-66322d2caa5c
# ╟─400669f2-cb60-467f-9b7d-492fb2079063
# ╟─645dc443-a89c-4696-a09e-446f1b8e4f79
# ╟─64be96b2-7016-4a70-b8fa-8aea5193770d
# ╟─0f78170c-b897-4c8b-b009-7591ebed0a6e
# ╟─3b80d3d6-a3fe-4101-97b5-7f899f8a37a9
# ╟─e6c684a8-7f6e-46cb-bde3-5d7e283b4199
# ╟─2ac3e869-99b1-47d3-95bf-92441c67f22d
# ╟─2f2fb971-efdb-4529-9c69-f8d986dd890a
# ╟─df4393fe-4196-4269-bbc5-9f4235bee90b
# ╟─ef9b449c-8726-4e54-87a5-c03e4fb6b816
# ╟─46429172-4b57-40ef-98ba-e9cbe862379f
# ╟─265b4419-9ee1-41e4-b056-e1efe7bc62ec
# ╟─c8727ee8-3919-44a4-8a06-059cf6689a3e
# ╟─3a6f3005-fa01-4644-a6c6-dc749c776891
# ╟─d0a52953-1a3d-4e0e-93a9-f4ed4fbd4cbf
# ╟─75ffc900-562a-4d24-9bbb-a9cffab30228
# ╠═a145e517-1476-4065-9894-9168b08e1ffa
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
# ╟─baa9dee5-9010-43a2-9cea-eee1424ecea2
# ╟─85c700cd-7279-4db9-bc70-8e830c01a10c
# ╟─a3ba3b1a-8ac4-4531-9e79-f45b944c74a0
