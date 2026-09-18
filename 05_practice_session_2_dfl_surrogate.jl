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

# ╔═╡ 5e9a2000-a1b2-4c3d-8e9f-000000000001
begin
    import Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
end

# ╔═╡ 5e9a2000-a1b2-4c3d-8e9f-000000000002
begin
    using StochasticTailAssignment
    using StochasticTailAssignment.AircraftRoutingBase
    using StochasticTailAssignment.InstanceGenerator
    using StochasticTailAssignment.FlightDelayModel
    using DecisionFocusedLearningBenchmarks
    using DecisionFocusedLearningAlgorithms
    using InferOpt: LinearMaximizer, PerturbedAdditive, FenchelYoungLoss, compute_probability_distribution
    using Flux: Flux, Adam
    using JLD2
    using Random
    using Statistics: mean, median
    using LinearAlgebra: dot
    using Zygote
    using PlutoUI
    using PlutoTeachingTools
    using ProgressLogging
    using WGLMakie # If this doesn't work on your browser, delte this line and uncomment the line below
    # using CairoMakie
end

# ╔═╡ 3e1088c9-3990-4947-a8dd-cc7bd1e0d374
md"""
# Practice session 2: a decision-focused learning policy

$(PlutoUI.TableOfContents(; depth=2))
"""

# ╔═╡ 7dae61cc-9ce2-4319-96ca-1b91fce6c752
tip(
    "This notebook trains with threaded perturbations, it currently sees **$(Threads.nthreads()) thread(s)**. " *
    (Threads.nthreads() == 1 ?
     "Training will still work, but it will be several times slower, restart Pluto with `julia --threads=auto -e 'using Pluto; Pluto.run()'` to use all cores." :
     "Training will use them."),
)

# ╔═╡ 564cb970-9ec4-4a8e-9f2f-7df7d71547d0


# ╔═╡ 8e38e69f-579b-4969-9f88-47fd28705c2e
md"""
## Introduction
### Differentiating through an optimization layer

A tiny two-dimensional example before the real problem: a polygon instead of a schedule graph, one cost vector θ instead of one per arc.
"""

# ╔═╡ fbf14a46-1fe9-441f-8ec8-d67f8647c92f
const TOY_VERTICES = [
    [1.0, 0.2],
    [0.6, 0.9],
    [-0.3, 1.0],
    [-1.0, 0.3],
    [-0.8, -0.6],
    [0.0, -1.0],
    [0.9, -0.5],
]

# ╔═╡ 43e28775-c9b3-4919-947d-e7f228a74e7a
"""
Toy maximizer: the polygon vertex farthest in the direction of θ, `argmax_{y in vertices} θᵀy`.
"""
toy_maximizer(θ; vertices=TOY_VERTICES, kwargs...) = vertices[argmax(dot(θ, v) for v in vertices)]

# ╔═╡ faeda526-988b-433d-8281-fd4ae577d835
md_angle = md"direction angle: $(@bind toy_angle PlutoUI.Slider(0:0.01:6.28; default=0.7, show_value=true))"

# ╔═╡ d99a713c-8f04-4098-afec-faf2967435da
toy_θ = [cos(toy_angle), sin(toy_angle)]

# ╔═╡ 809a18c1-180a-4d0d-8803-0610dbb08953
md"The output is piecewise constant in θ, the selected vertex jumps from one corner to the next as the angle crosses an edge, it never slides."

# ╔═╡ 44eb3212-9db8-470c-bde9-be8ec1557984
Zygote.jacobian(θ -> toy_maximizer(θ; vertices=TOY_VERTICES), toy_θ)[1]

# ╔═╡ 0d275dbb-5373-446d-a9f8-a343b551964a
md"The Jacobian is zero almost everywhere (and undefined exactly at the jumps), so gradient descent gets no signal from the maximizer directly."

# ╔═╡ 631741d6-a2ec-41af-a75d-0b72c282d083
md"""
### Smoothing by perturbation

`PerturbedAdditive` averages the maximizer over several random perturbations of θ, replacing one hard vertex by a weighted cloud of vertices, and their expectation moves smoothly.

perturbation scale ε: $(@bind toy_ε PlutoUI.Slider(0.05:0.05:1.0; default=0.3, show_value=true))

number of perturbation samples: $(@bind toy_nb_samples PlutoUI.Slider(10:10:200; default=100, show_value=true))
"""

# ╔═╡ 30a483d8-733a-40e8-bcc3-384411cb2153
md_angle

# ╔═╡ 4f6ff23a-4401-46f5-ad1c-8551d78234f6
toy_perturbed =
    PerturbedAdditive(toy_maximizer; nb_samples=toy_nb_samples, ε=toy_ε, threaded=false, seed=0)

# ╔═╡ 86e29a7e-7f06-46f4-a85a-670ab4e6d778
md"""Dot area is how often that vertex wins across the perturbed draws, the orange diamond is their weighted average, still inside the polygon."""

# ╔═╡ d36172e6-bd2a-43eb-85c9-8a9e0cb79d4e
let
    fig = Figure(; size=(500, 320))
    ax = Axis(fig[1, 1]; xlabel="θ₁ (θ₂ fixed)", ylabel="cost", title="Piecewise constant vs smooth")
    xs = range(-1.5, 1.5; length=200)
    target = TOY_VERTICES[3]
    toy_cost(y) = -dot(target, y)
    staircase = [toy_cost(toy_maximizer([x, toy_θ[2]])) for x in xs]
    smooth = [toy_cost(mean(compute_probability_distribution(toy_perturbed, [x, toy_θ[2]]))) for x in xs]
    lines!(ax, xs, staircase; label="through the maximizer", linewidth=2)
    lines!(ax, xs, smooth; label="through the perturbed layer", linewidth=2)
    axislegend(ax; position=:rb)
    fig
end

# ╔═╡ 090416db-2bdf-452f-8464-62361bb7c10d
md"""The blue line jumps at each vertex change, the orange one bends smoothly instead."""

# ╔═╡ b7d60f6e-503f-4bc3-bb92-4663fb22e345
md"""
### The Fenchel-Young loss

`FenchelYoungLoss(toy_perturbed)` turns the perturbed layer into a convex loss against a fixed target vertex, smooth enough to have a gradient everywhere.
"""

# ╔═╡ 730cfe08-70ac-429e-a4a8-9cd845a21a01
begin
    toy_fyl = FenchelYoungLoss(toy_perturbed)
    toy_target = TOY_VERTICES[3]
end

# ╔═╡ bc1c5b4f-9941-4289-94d3-89a2b4adc068
let
    grid = range(-1.5, 1.5; length=60)
    Z = [toy_fyl([x, y], toy_target) for x in grid, y in grid]
    fig = Figure(; size=(480, 420))
    ax = Axis(
        fig[1, 1]; xlabel="θ₁", ylabel="θ₂", aspect=DataAspect(),
        title="Fenchel-Young loss around a fixed target vertex",
    )
    contourf!(ax, grid, grid, Z)
    arrow_grid = range(-1.4, 1.4; length=10)
    arrow_x = Float64[]
    arrow_y = Float64[]
    arrow_u = Float64[]
    arrow_v = Float64[]
    for x in arrow_grid, y in arrow_grid
        g = -Zygote.gradient(θ -> toy_fyl(θ, toy_target), [x, y])[1]
        push!(arrow_x, x)
        push!(arrow_y, y)
        push!(arrow_u, g[1] / 8)
        push!(arrow_v, g[2] / 8)
    end
    arrows2d!(ax, arrow_x, arrow_y, arrow_u, arrow_v; color=:white, shaftwidth=1.5, tipwidth=6, tiplength=5)
    scatter!(ax, [toy_target[1]], [toy_target[2]]; color=:red, markersize=16, marker=:star5)
    fig
end

# ╔═╡ d91e8366-ac43-4b95-9302-20cfc715277e
md"The landscape is smooth everywhere and the gradient arrows point toward the region where the target vertex wins, exactly the descent direction training follows."

# ╔═╡ 07fda6d3-52cc-4ef0-807a-a5d3c6e8e59c
keyconcept(
    "From polygon to routing",
    md"Swap the polygon for the routing MIP's feasible set: vertices become arc vectors (one route per aircraft), the maximizer becomes the HiGHS edge MIP, and θ becomes one predicted cost per arc, everything else on this page, the smoothing, the loss, the gradient, is identical.",
)

# ╔═╡ 36dbc492-ba6c-4001-8b25-6533ec88285f
md"""
### From features to a routing decision

Three steps: a 23-number feature vector per interior arc, a small neural network predicting one arc cost θ, and a deterministic edge MIP maximizing θᵀy to pick one path per aircraft.
`generate_context` below leaves `x` as `nothing`, the target policy fills it once the K SAA scenarios exist (Section 4), from those same scenarios, so label and features always describe the same frozen delay draw.
"""

# ╔═╡ 4aedb21d-9dde-461d-965e-a4b517ffec8d
keyconcept(
    "What is still solved at prediction time",
    md"The surrogate replaces the expert's *search*, not the solver: a deterministic edge MIP still runs every prediction, now on learned costs θ instead of stochastic column generation and diving.",
)

# ╔═╡ 2b6a3712-9928-4bdf-9a6f-8c11dbd597a0
md"""
### Why you cannot differentiate through an argmax

The edge MIP is exactly the toy maximizer above, an argmax over a discrete feasible set, so it has the same zero-almost-everywhere Jacobian.
`PerturbedFenchelYoungLossImitation` smooths it the same way, by averaging over random perturbations of θ, the mechanism the polygon figures made concrete.
"""

# ╔═╡ 5028aadb-b2a0-495f-a8bf-3dbb0fcf07fc
keyconcept(
    "Smoothing by perturbation",
    md"A perturbed maximizer responds to how *often* a perturbed θ favors a different route, not to a single hard switch, so it is smooth in θ and has a usable gradient.",
)

# ╔═╡ 1a812d10-7702-49c8-ac98-360ff4ffb7e3
md"""
### The Fenchel-Young loss

Same loss as the contour figure above, now with θ the predicted arc costs and ȳ the expert's arc selection instead of a fixed target vertex.
Its gradient has the same clean form, $\nabla_\theta \mathcal{L}^{\mathrm{FYL}} = \hat y(\theta) - \bar y$, the surrogate's own perturbed arc selection minus the expert's.
"""

# ╔═╡ 52311ed7-a07b-45f8-93c8-a15e69548399
keyconcept(
    "The Fenchel-Young gradient",
    md"Push predicted costs toward the arcs the expert used, and away from the arcs the surrogate picked instead, one gradient step at a time.",
)

# ╔═╡ afab3069-b909-466f-8f3b-5eaeb20333fc
md"""
### SAA and joint labeling

The decision (one route per aircraft) must be made **before** delays are known, and be good on average, not tailored to one draw.
So the expert labels one **sample average approximation** (SAA): K frozen delay scenarios, optimized jointly, one set of routes as the label for the whole instance.
The same K scenarios also produce the features `x`, recomputed from them once fixed, so label and features always describe the same frozen draw.
"""

# ╔═╡ 9edb2355-846f-4e26-91e5-10cd6c9432e2
aside(
    md"""
**Unlike the knapsack demo**, which labeled each scenario independently (its own optimal decision, no aircraft reuse), here one route per aircraft must be fixed before delays are known, so the label is one SAA decision robust to K scenarios at once.
""",
)

# ╔═╡ 2ac77c30-f39b-4370-9868-88a207a1af0b
md"""
## 1. Define the benchmark type

`DecisionFocusedLearningBenchmarks` needs five things: **instance**, **context** (features), **scenario**, **score** (a decision under a scenario), and **direction**, plus a **maximizer** and a **statistical model**.
Delays do not depend on our decisions, so the benchmark is **exogenous**, `AbstractStochasticBenchmark{true}`.
"""

# ╔═╡ c09f83bc-a8fe-4399-8ae7-9d254bf9b731
const DFLB = DecisionFocusedLearningBenchmarks

# ╔═╡ d050236e-e869-4195-9628-f1d24e24cc83
begin
    # Pluto requires a struct and the outer constructor that builds it to live in the same
    # cell, otherwise it sees two cells both defining the global `StochasticTailAssignmentBenchmark`.
    struct StochasticTailAssignmentBenchmark{C} <: AbstractStochasticBenchmark{true}
        "number of flight legs per generated instance"
        nb_legs::Int
        "number of delay scenarios the expert optimizes over, also the scenarios the slack-quantile features are built from (see `expert_target_policy`)"
        nb_scenarios::Int
        "seats per aircraft, scales the delay cost"
        nb_seats::Int
        "piecewise linear delay cost function, shared by every instance"
        delay_cost_function::C
    end

    function StochasticTailAssignmentBenchmark(; nb_legs=50, nb_scenarios=30, nb_seats=180)
        delay_cost_function = DelayCostFunction(;
            slopes=nb_seats .* FlightDelayModel.DELAY_COST_SLOPE_MEDIUM_HAUL
        )
        return StochasticTailAssignmentBenchmark(
            nb_legs, nb_scenarios, nb_seats, delay_cost_function
        )
    end
end

# ╔═╡ 868fcead-a545-43f3-97d7-c4925d50130e
bench = StochasticTailAssignmentBenchmark(; nb_legs=50, nb_scenarios=30)

# ╔═╡ f17db97d-7036-4ccc-b061-a52253e91da2
DFLB.is_minimization_problem(::StochasticTailAssignmentBenchmark) = true

# ╔═╡ 4508eed1-2cf0-4b6d-9056-26e8e4d5d795
md"Costs, not utilities, so this one is a minimization problem, unlike the knapsack demo."

# ╔═╡ 81169384-a0ad-46c7-a0c5-b10bf9e80eaf
"""
Scale each feature row of `x` to [0, 1] within the instance, so the network sees inputs of
comparable magnitude whatever the schedule. Rows that are constant are left at zero.
Per instance rather than per dataset: a fresh instance at prediction time then needs no
statistic fitted on training data, and the arc scores θ are only ever compared inside one
instance anyway.
"""
function scale_features(x)
    mins = minimum(x; dims=2)
    ranges = maximum(x; dims=2) .- mins
    ranges[ranges .== 0] .= 1
    return Float32.((x .- mins) ./ ranges)
end

# ╔═╡ 5e9a03df-1d1b-487d-9ade-3b8b5a02f06f
function DFLB.generate_instance(
    bench::StochasticTailAssignmentBenchmark, rng::Random.AbstractRNG
)
    seed = rand(rng, 1:1_000_000)
    schedule, _, _ = generate_benchmark_instance(
        bench.nb_legs; nb_scenarios=1, seed, store_arc_index=true
    )
    delay_model = build_delay_model(schedule)
    return DataSample(; instance=schedule, delay_model)
end

# ╔═╡ ef891c26-7163-4638-8e75-4c7758adb0b9
md"""
`store_arc_index=true` is required to compute features and decode arc labels.
`delay_model` is cached in the context, built once, since both features and every scenario draw need it.
"""

# ╔═╡ 86d5a09a-40c7-4071-9dbb-d56f8623e74f
function DFLB.generate_context(
    bench::StochasticTailAssignmentBenchmark,
    rng::Random.AbstractRNG,
    instance_sample::DataSample,
)
    # x stays `nothing` here, the target policy below fills it from the K SAA scenarios.
    return instance_sample
end

# ╔═╡ e5b0ac6e-6e0b-4973-b4dc-16321067261f
function DFLB.generate_scenario(
    bench::StochasticTailAssignmentBenchmark,
    rng::Random.AbstractRNG;
    instance,
    delay_model,
    kwargs...,
)
    scenarios = DelayScenarios(instance; nb_scenarios=1, seed=rand(rng, 1:1_000_000))
    unmerged = sample_root_scenarios_unmerged(delay_model, scenarios)
    return (; departure=vec(unmerged.departure), arrival=vec(unmerged.arrival))
end

# ╔═╡ 1cbfa417-8dc1-436e-a305-fea810813720
question_box(
    md"""
`y` is a binary arc-selection vector, `scenario` a `(; departure, arrival)` pair of root delay vectors.
Write the cost of `y` under that scenario: sum the two components, decode the routes, evaluate with `full_cost`, which expects a scenarios-by-legs **matrix**, so reshape the single scenario into one row.
""",
)

# ╔═╡ 85dc4872-6f1f-4aa6-b032-9fc49f7ca849
"Score decision `y` under one delay `scenario`, for `sample`'s instance."
function DFLB.objective_value(
    bench::StochasticTailAssignmentBenchmark, sample::DataSample, y, scenario
)
    result = missing
    instance = sample.instance
    routes = decode_routes_from_arc_solution(y, instance)  # decode the arc selection into routes
    root_delays = scenario.departure .+ scenario.arrival  # combine the two delay components
    result = missing  # TODO: full_cost(routes, reshape(root_delays, 1, :), instance; delay_cost_function=bench.delay_cost_function), reshape needed because full_cost expects a scenarios-by-legs matrix
    return result
end

# ╔═╡ abf0a855-3c55-468a-bc7c-f82a41e8cab1
Foldable(
    "Full solution",
    md"""
```julia
function DFLB.objective_value(bench::StochasticTailAssignmentBenchmark, sample::DataSample, y, scenario)
    instance = sample.instance
    routes = decode_routes_from_arc_solution(y, instance)
    root_delays = scenario.departure .+ scenario.arrival
    return full_cost(
        routes,
        reshape(root_delays, 1, :),
        instance;
        delay_cost_function=bench.delay_cost_function,
    )
end
```
`reshape(v, 1, :)` turns a length-L vector into a 1 by L matrix without copying, exactly the scenarios-by-legs shape `full_cost` expects for a single scenario.
""",
)

# ╔═╡ 460600a5-08b2-463d-ab9b-81971ae151ac
instance_sample = DFLB.generate_instance(bench, Xoshiro(67))

# ╔═╡ 63301851-f92f-47f1-9785-879bc2d11ce8
sample = DFLB.generate_context(bench, Xoshiro(67), instance_sample)

# ╔═╡ e458661a-de1d-4052-9bb4-dc1a1cdde506
sample.x  # nothing: no scenario has been drawn yet, see Section 4

# ╔═╡ f46495af-e2ba-477e-999f-e136ea860779
ξ = DFLB.generate_scenario(bench, Xoshiro(67); sample.context...)

# ╔═╡ a454f17b-b84f-4993-9ec6-b8f9cd2cb2f6
md"""
## 2. The optimization oracle

The oracle is the edge-based MIP from part 1's package: given θ per connection, pick one path per aircraft maximizing θᵀy.
The graph also has source and sink arcs with no prediction, so `LinearMaximizer`'s `g` slices the decision down to interior arcs before it meets θ.
"""

# ╔═╡ 3f72a5c3-cc00-4d58-8679-d4468ec4e0d6
function sta_edge_maximizer(θ::AbstractVector; instance, relaxation=false, kwargs...)
    return aircraft_routing_edge_maximizer(
        θ;
        instance,
        include_chaining_costs=true,
        model_builder=highs_model,
        use_operational_costs=false,
        silent=true,
        relaxation,
    )
end

# ╔═╡ 7af83fe2-0fde-4e98-9feb-bd55a64417d0
interior_arcs(y; instance, kwargs...) = y[1:(instance.nb_interior_arcs)]

# ╔═╡ 21d2593d-d77c-4456-8486-61763c298c5b
function DFLB.generate_maximizer(::StochasticTailAssignmentBenchmark)
    return LinearMaximizer(sta_edge_maximizer; g=interior_arcs)
end

# ╔═╡ 3a27631a-3c48-4d69-94a6-8ad71a36274f
maximizer = DFLB.generate_maximizer(bench)

# ╔═╡ 95f7c78d-302d-4f43-b582-47870a1d027e
md"""
## 3. The statistical model

One small feedforward network, shared by every connection, maps its 23 features to a score.
A single call `model(x)` scores every connection at once, `x` a 23 by (connections) matrix.
"""

# ╔═╡ 3eeeb774-674c-44e7-929a-1e33c9cc5c82
const NB_FEATURES = 23

# ╔═╡ cb9eb966-1d16-478a-8bee-091137b9691a
function DFLB.generate_statistical_model(
    ::StochasticTailAssignmentBenchmark; seed=nothing
)
    isnothing(seed) || Random.seed!(seed)
    return default_model(NB_FEATURES; hidden_dims=[16, 16])
end

# ╔═╡ 164024df-6a15-4ccf-8382-3a61ad0247e8
initial_model = DFLB.generate_statistical_model(bench; seed=67)

# ╔═╡ a8203ace-eeef-4861-9516-f66c3fff40a2
# a one-scenario preview of the feature matrix shape, using the single demo scenario ξ
# above, exactly the same compute_features/scale_features pair the target policy uses
# below on the K frozen SAA scenarios.
θ_demo = let
    x_demo = scale_features(
        compute_features(
            sample.instance, reshape(ξ.departure, 1, :), reshape(ξ.arrival, 1, :)
        ),
    )
    initial_model(x_demo)
end

# ╔═╡ 17869e8d-d8e0-4bf9-a766-22493a3b9e89
y_demo = maximizer(θ_demo; sample.context...)

# ╔═╡ 527eaa81-0677-46ae-a04b-7ef57b294e9a
# Cheap: reuses the already-computed demo decision and scenario above, no new solve.
let
    obj_demo = DFLB.objective_value(bench, sample, y_demo, ξ)
    if ismissing(obj_demo)
        still_missing()
    elseif !isfinite(obj_demo)
        keep_working(md"The returned cost is not a finite number, check the reshape and the delay cost function.")
    else
        correct()
    end
end

# ╔═╡ d7a27f14-bd9c-4012-b6ea-e926ac2936d3
md"""
## 4. The expert: sample average approximation

`SampleAverageApproximation(bench, K)` freezes K scenarios per instance and hands them to a **target policy**, the function that labels the instance.
Our target policy is part 1's solver: column generation over the K scenarios, then diving.
"""

# ╔═╡ 28e465ca-a503-4806-9043-3d3d5b7591d3
saa = SampleAverageApproximation(bench, bench.nb_scenarios)

# ╔═╡ 1d2fde83-9469-4cee-a74e-3c1040998546
question_box(
    md"""
Signature `(ctx_sample, scenarios) -> Vector{DataSample}`, one label per instance from all K scenarios together.
Stack `departure` and `arrival` into two K by L matrices, sum them, call `sta_expert_routes` (chains warm start, column generation, diving), decode the routes with `decode_arc_solution_from_routes`, return a labeled copy of the sample.
If `sta_expert_routes` returns `nothing`, return `DataSample[]` instead.
Recompute `x` from the same stacked matrices with `compute_features` and `scale_features`, and keep the routes, `x_raw`, and the wall-clock time in `extra`.
""",
)

# ╔═╡ 191fa21a-4937-451d-94f6-ab3590f145ce
hint(
    md"`reduce(vcat, (reshape(ξ.departure, 1, :) for ξ in scenarios))` stacks the K departure vectors into a K by L matrix, do the same for `arrival`, then sum the two matrices.",
)


# ╔═╡ 7c7d8f05-6d9b-417e-8ec3-22347b3a7c34
Foldable(
    "Full solution",
    md"""
```julia
function expert_target_policy(ctx_sample::DataSample, scenarios::AbstractVector)
    instance = ctx_sample.instance
    departure_root_delays = reduce(vcat, (reshape(ξ.departure, 1, :) for ξ in scenarios))
    arrival_root_delays = reduce(vcat, (reshape(ξ.arrival, 1, :) for ξ in scenarios))
    root_delays = departure_root_delays .+ arrival_root_delays
    timed = @timed sta_expert_routes(instance, root_delays, bench.delay_cost_function)
    routes = timed.value
    routes === nothing && return DataSample[]
    y = decode_arc_solution_from_routes(routes, instance)
    x_raw = compute_features(instance, departure_root_delays, arrival_root_delays)
    return [
        DataSample(
            ctx_sample;
            x=scale_features(x_raw),
            y,
            extra=(; ctx_sample.extra..., x_raw, scenarios, routes, expert_seconds=timed.time),
        ),
    ]
end
```
`DataSample(sample; x, y, extra=(; sample.extra..., k=v))` copies a sample and overrides only the given fields.
""",
)

# ╔═╡ b6a7faeb-6dd8-4232-a773-070c2276da80
md"""
## 5. The training set: practice session 1's dataset

Part 1 saved about 30 solved instances to `data/practice_session_1_dataset.jld2`, the expert runs once offline, its output is a file.
Each entry becomes one `DataSample`: features, label, and the K delay scenarios kept in `extra`.
Missing the file? This notebook falls back to generating its own set, about a minute of solving.
"""

# ╔═╡ 50e05260-cd17-46fb-aa60-57d6092e98c6
md"""load the training set: $(@bind train_data_click PlutoUI.CounterButton("Load the training set"))"""

# ╔═╡ c2cd6fa1-18c8-4377-9012-21cf0c66241b
Markdown.parse(
    isfile(joinpath(@__DIR__, "data", "practice_session_1_dataset.jld2")) ?
    "Expected runtime: well under a second, reading `data/practice_session_1_dataset.jld2` back from disk." :
    "Expected runtime: about a minute, `data/practice_session_1_dataset.jld2` was not found so this button falls back to solving 30 fresh instances through the benchmark's own expert target policy.",
)

# ╔═╡ fd1a924b-2305-4421-92d3-a39c50608e10
begin
    train_data_request_cache = Ref{Any}((click=0, request=nothing))
    train_data_result_cache = Ref{Any}((click=0, request=nothing, result=nothing))
end

# ╔═╡ b299411f-989b-400e-a090-085c0ddfbf7b
begin
    train_data_current_request = (path=joinpath(@__DIR__, "data", "practice_session_1_dataset.jld2"),)
    if train_data_click > train_data_request_cache[].click
        train_data_request_cache[] =
            (click=train_data_click, request=train_data_current_request)
    end
    train_data_request = train_data_request_cache[].request
end

# ╔═╡ 3bfc7519-2824-40bc-a780-c4a25ff0aace
md"""
## 6. A fresh test set

Eight instances the surrogate never trains on, labeled by the same expert, every number below is measured on this set.
Eight is a compromise between noise (Section 8 shows per-instance spread) and the button's runtime.
"""

# ╔═╡ 3932b390-2f88-4678-8be0-418a03b73134
begin
    session2_test_controls = md"""
    inspection instance: $(@bind session2_inspect_idx PlutoUI.Slider(1:1:8; default=1, show_value=true))

    generate the fresh test set: $(@bind test_click PlutoUI.CounterButton("Generate the fresh test set"))
    """
    session2_test_controls
end

# ╔═╡ ac26961f-0a3f-40e6-8d7f-10713c00191c
tip(
    md"The test-set button takes under a minute, measured on the instructor's machine, a laptop may well be slower.",
)

# ╔═╡ a19e0a28-cc7a-4339-ac3c-c0a430de1df3
begin
    test_request_cache = Ref{Any}((click=0, request=nothing))
    test_result_cache = Ref{Any}((click=0, request=nothing, result=nothing))
end

# ╔═╡ 03a6b02b-787e-4614-88bd-960eb900c8f7
begin
    # n and seed are fixed, not chosen to make the reported gap look good, see Section 8.
    test_current_request = (n=8, seed=90_000)
    if test_click > test_request_cache[].click
        test_request_cache[] = (click=test_click, request=test_current_request)
    end
    test_request = test_request_cache[].request
end

# ╔═╡ 226419ca-751b-4d03-b0fe-30a96040cea4
md"""
## 7. Train the policy

`DFLPolicy` bundles the network with the oracle, `PerturbedFenchelYoungLossImitation` perturbs, averages, and pushes the prediction toward the expert.
`train_policy!` evaluates the validation metric at every epoch, including epoch 0: the **in-sample** SAA gap against the fresh test set.
Training does not restart automatically, regenerate the test set then press "Train the surrogate" again.
"""

# ╔═╡ 876c9a36-5afc-4ea4-8632-f4d565690158
warning_box(
    md"This in-sample gap differs from Section 9's **out-of-sample** gap, which re-scores on delay draws neither side has seen.",
)

# ╔═╡ e8fdce16-5fc7-4893-b29f-29e709aeeb11
md"""
training epochs: $(@bind session2_epochs PlutoUI.Slider(5:5:30; default=20, show_value=true))

perturbed samples per gradient step: $(@bind session2_nb_samples PlutoUI.Slider(2:1:10; default=5, show_value=true))

train the surrogate: $(@bind train_click PlutoUI.CounterButton("Train the surrogate"))
"""

# ╔═╡ 1ef9812b-6b96-4bbd-a947-6b28793c8e12
md"The training button takes a few minutes at the default sliders."

# ╔═╡ 997a8c2c-2d25-477d-ac05-26ad117578b3
begin
    train_request_cache = Ref{Any}((click=0, request=nothing))
    train_result_cache = Ref{Any}((click=0, request=nothing, result=nothing))
end

# ╔═╡ 04a46f0d-62e0-46bf-884d-abbf02280113
md"""
## 8. Results

The deterministic baseline (no stochasticity) sits well above the surrogate, that gap is what decision-focused learning chases.
Every full cost gap here is the mean of each instance's own relative gap, `DecisionFocusedLearningBenchmarks.compute_gap`'s convention.
"""

# ╔═╡ 46484b09-fbb4-43ec-b765-81847383f4cf
warning_box(
    md"The gap varies a lot across the 8 test instances, an occasional outlier can move the mean, see the per-instance table below.",
)

# ╔═╡ 251fdd64-1fc9-42f4-8e63-f0595e48af7a
md"""
### Per-instance gap

A single mean hides how the surrogate performs, one bad instance can move it.
The table below breaks the same gap down instance by instance, read the median and the spread.
"""

# ╔═╡ 8eda63fe-701b-4b22-8d07-05bd9dadea74
question_box(
    md"""
The table above has everything you need: expert time, surrogate time, gap (`session2_table.full_cost_gap_percent[2]`, in percent).
Compute the speedup ratio and pass the gap through unchanged.
""",
)

# ╔═╡ 994d7f21-3472-4a31-9ad8-5b77ac17e77f
"Return `(; speedup, full_cost_gap_percent)` from `expert_time`, `surrogate_time` and `full_cost_gap_percent`."
function speedup_and_gap(expert_time, surrogate_time, full_cost_gap_percent)
    result = missing
    speedup = missing  # TODO: expert_time / surrogate_time
    result = ismissing(speedup) ? missing : (; speedup, full_cost_gap_percent)
    return result
end

# ╔═╡ 9245bd07-4761-49ed-8ec2-01f05bc61927
Foldable(
    "Full solution",
    md"""
```julia
function speedup_and_gap(expert_time, surrogate_time, full_cost_gap_percent)
    return (; speedup=expert_time / surrogate_time, full_cost_gap_percent)
end
```
`(; a=1, b=2)` builds a NamedTuple with named fields, useful here to return both numbers from one function.
""",
)

# ╔═╡ ef411983-077d-4842-ab70-d5f1810e40ee
md"""
## 9. Out-of-sample evaluation

Section 8's gap is **in-sample**, favorable to the expert, scored on the scenarios its features were built from.
This section re-scores both already-fixed decisions on 30 **fresh** scenarios neither side has seen, the fair, decision-quality number.
"""

# ╔═╡ 1a23c62b-3f4b-4c06-b933-d5c1a2b899cc
"Score expert and surrogate decisions on `nb_scenarios` fresh draws, return average costs and the surrogate's gap."
function out_of_sample_gap(sample::DataSample, policy::DFLPolicy; nb_scenarios=30, seed)
    fresh = [
        DFLB.generate_scenario(bench, Xoshiro(seed + k); sample.context...) for
        k in 1:nb_scenarios
    ]
    ŷ = policy(sample.x; sample.context...)
    expert_cost = mean(DFLB.objective_value(bench, sample, sample.y, ξ) for ξ in fresh)
    surrogate_cost = mean(DFLB.objective_value(bench, sample, ŷ, ξ) for ξ in fresh)
    return (;
        expert_cost, surrogate_cost, gap_percent=100 * (surrogate_cost - expert_cost) / abs(expert_cost)
    )
end

# ╔═╡ 92d0c429-a893-4688-a3d5-0f4c2be188ab
md"""
## 11. Expert vs surrogate routes

`plot_gantt` compares each aircraft's rotation, expert routes on top, surrogate routes below, for the inspection instance.
"""

# ╔═╡ 60711b0e-a34c-40bb-8031-6f06628ac011
md"""
## 12. Arc selection agreement

θ is the surrogate's predicted cost per interior arc, ŷ the decoded selection, ȳ the expert's, both binary, one entry per interior arc.
"""

# ╔═╡ f2c131f5-f56d-4a08-b4a0-fc02dc150b94
question_box(
    md"""
Predict θ, decode it into ŷ with the shared maximizer, compute the fraction of arcs where ŷ agrees with ȳ.
`maximizer(θ; instance)` also covers source and sink arcs, slice its output down to `1:instance.nb_interior_arcs` first.
""",
)

# ╔═╡ 193a7467-1002-431c-9d8c-f994e47eb635
"Decode θ into ŷ and compare with ȳ, return `(; ŷ, agreement)`."
function surrogate_agreement(θ, ȳ, maximizer, instance)
    result = missing
    ŷ_full = maximizer(θ; instance)  # one entry per arc, including source and sink
    ŷ = ŷ_full[1:(instance.nb_interior_arcs)]  # slice down to interior arcs, matching ȳ
    ȳ_interior = ȳ[1:(instance.nb_interior_arcs)]
    agreement = missing  # TODO: count(abs.(ŷ .- ȳ_interior) .< 0.5) / length(ȳ_interior)
    result = ismissing(agreement) ? missing : (; ŷ, agreement)
    return result
end

# ╔═╡ cadf672f-d932-4607-96c5-942add9ae86d
Foldable(
    "Full solution",
    md"""
```julia
function surrogate_agreement(θ, ȳ, maximizer, instance)
    ŷ_full = maximizer(θ; instance)
    ŷ = ŷ_full[1:(instance.nb_interior_arcs)]
    ȳ_interior = ȳ[1:(instance.nb_interior_arcs)]
    agreement = count(abs.(ŷ .- ȳ_interior) .< 0.5) / length(ȳ_interior)
    return (; ŷ, agreement)
end
```
`maximizer(θ; instance)` takes θ positionally and the instance as a keyword, `LinearMaximizer` forwards both straight to `sta_edge_maximizer`.
""",
)

# ╔═╡ 8d6ddc95-d241-4e08-8d1c-d4ba38abff8b
md"""
## Appendix: helpers

Plotting, caching, and formatting code, plus the expert solver and the dataset loader, safe to skip.
"""

# ╔═╡ e46c11eb-8102-4f7b-9e2c-d6895b9b4113
"""
Plot the toy polygon, the direction arrow for θ, and the selected vertex highlighted.
`extra!` optionally draws more on the same axis before the vertex marker.
"""
function plot_toy_polygon(θ; vertices=TOY_VERTICES, extra! = ax -> nothing)
    chosen = toy_maximizer(θ; vertices)
    fig = Figure(; size=(420, 420))
    ax = Axis(
        fig[1, 1]; aspect=DataAspect(), xlabel="y₁", ylabel="y₂", title="f(θ) over the polygon"
    )
    poly!(ax, Point2f.(vertices); color=(:steelblue, 0.15), strokecolor=:steelblue, strokewidth=2)
    extra!(ax)
    arrows2d!(ax, [0.0], [0.0], [θ[1]], [θ[2]]; color=:black, shaftwidth=3, tipwidth=14)
    scatter!(ax, [chosen[1]], [chosen[2]]; color=:crimson, markersize=20, strokewidth=1.5, strokecolor=:black)
    limits!(ax, -1.4, 1.4, -1.4, 1.4)
    fig
end

# ╔═╡ cda0b222-ff8e-47d0-b150-d3046e07a69b
plot_toy_polygon(toy_θ)

# ╔═╡ 05434146-9f04-45d5-b4fd-cfb82d7f0312
"""
Merge duplicate atoms of a perturbed-maximizer probability distribution (several perturbed
draws can land on the same vertex), summing their weights, in place.
"""
function compress_toy_distribution!(probadist; atol=0)
    (; atoms, weights) = probadist
    to_delete = Int[]
    for i in length(probadist):-1:1
        for j in 1:(i - 1)
            if isapprox(atoms[i], atoms[j]; atol=atol)
                weights[j] += weights[i]
                push!(to_delete, i)
                break
            end
        end
    end
    deleteat!(atoms, sort(to_delete))
    deleteat!(weights, sort(to_delete))
    return probadist
end

# ╔═╡ ca8004e3-d8c7-45cc-bb03-e39c7c9000f1
"Plot the toy polygon with the perturbed distribution's weighted vertex atoms and their expectation overlaid."
function plot_toy_perturbed(θ; vertices=TOY_VERTICES)
    probadist = compute_probability_distribution(toy_perturbed, θ; vertices)
    compress_toy_distribution!(probadist)
    plot_toy_polygon(
        θ;
        vertices,
        extra! = ax -> begin
            scatter!(
                ax,
                first.(probadist.atoms),
                last.(probadist.atoms);
                markersize=6 .+ 34 .* sqrt.(probadist.weights),
                color=(:steelblue, 0.5),
            )
            ŷ = mean(probadist)
            scatter!(
                ax, [ŷ[1]], [ŷ[2]]; color=:orange, markersize=16, marker=:diamond,
                strokewidth=1.5, strokecolor=:black,
            )
        end,
    )
end

# ╔═╡ b1edca94-2426-47e4-af6d-bf6cb467942a
plot_toy_perturbed(toy_θ)

# ╔═╡ 1c05f7be-5b88-4488-aa75-677598f81aee
begin
    format_seconds(x) = ismissing(x) ? "n/a" : string(round(x; digits=2), " s")
    format_seconds_with_ms(x) =
        ismissing(x) ? "n/a" :
        x < 1 ? string(round(x; digits=2), " s (", round(1000x; digits=1), " ms)") :
        format_seconds(x)
    format_cost(x) = ismissing(x) ? "n/a" : string(round(x; digits=1))
    format_pct(x) = ismissing(x) ? "n/a" : string(round(x; digits=2), " %")
end

# ╔═╡ 513363e4-8f84-475a-b7f2-77494af26bf8
begin
    "Read part 1's dataset into `DataSample`s, `(nothing, nothing)` if missing, `(nothing, message)` on a read error, `(samples, nothing)` on success."
    function session1_samples(path)
        isfile(path) || return (nothing, nothing)
        try
            entries = JLD2.load(path, "dataset")
            return ([session1_sample(entry) for entry in entries], nothing)
        catch caught_error
            message = first(split(sprint(showerror, caught_error), '\n'))
            return (nothing, message)
        end
    end

    "Turn one stored entry into a `DataSample`, see [`session1_samples`](@ref)."
    function session1_sample(entry)
        (; instance, departure_root_delays, arrival_root_delays, routes) = entry
        x_raw = compute_features(instance, departure_root_delays, arrival_root_delays)
        scenarios = [
            (; departure=Vector(departure_root_delays[s, :]), arrival=Vector(arrival_root_delays[s, :]))
            for s in axes(departure_root_delays, 1)
        ]
        # practice session 1's saved file does not carry expert_seconds (it is stripped before saving,
        # that timing only matters inside practice session 1's own notebook), default to NaN here so
        # this loader does not depend on a field practice session 1 does not actually persist.
        expert_seconds = hasproperty(entry, :expert_seconds) ? Float64(entry.expert_seconds) : NaN
        return DataSample(;
            x=scale_features(x_raw),
            y=decode_arc_solution_from_routes(routes, instance),
            instance,
            delay_model=build_delay_model(instance),
            extra=(; x_raw, scenarios, routes, expert_seconds),
        )
    end
end

# ╔═╡ 04238620-69cf-42ab-8965-4f286b0d1c45
"""
Decode the surrogate route for one test sample: run the policy (statistical model then
maximizer) on its features and context, and decode the resulting arc selection back into
routes for `sample.instance`.
"""
function surrogate_routes_for(sample::DataSample, policy::DFLPolicy)
    ŷ = policy(sample.x; sample.context...)
    return decode_routes_from_arc_solution(ŷ, sample.instance)
end

# ╔═╡ 2be101e6-a0d7-41f5-b29e-fcc2398aa0b0
"""
Average full cost of the expert routes stored in each sample of `data`, evaluated on that
sample's own SAA scenarios (`sample.extra.scenarios`, stacked into a scenarios by legs
matrix).
"""
function average_expert_full_cost(data)
    return mean(
        full_cost(
            s.extra.routes,
            reduce(vcat, (reshape(ξ.departure .+ ξ.arrival, 1, :) for ξ in s.extra.scenarios)),
            s.instance;
            delay_cost_function=bench.delay_cost_function,
        ) for s in data
    )
end

# ╔═╡ 1500e1dd-b5b9-4adb-9c2d-3f5c9d0754e2
"""
Average full cost of the routes `policy` decodes for each sample of `data`, evaluated on
that sample's own SAA scenarios.
"""
function average_surrogate_full_cost(data, policy::DFLPolicy)
    return mean(
        full_cost(
            surrogate_routes_for(s, policy),
            reduce(vcat, (reshape(ξ.departure .+ ξ.arrival, 1, :) for ξ in s.extra.scenarios)),
            s.instance;
            delay_cost_function=bench.delay_cost_function,
        ) for s in data
    )
end

# ╔═╡ d857d3d3-3403-4df9-a8f8-7fb943d1dcfa
"""
Solve the deterministic MIP for each sample's instance and return its average full cost,
average runtime, and average full cost gap against the expert routes, evaluated on the same
SAA scenarios.
"""
function deterministic_baseline(data)
    total_cost = 0.0
    total_time = 0.0
    total_gap = 0.0
    nb_gap = 0
    @progress "deterministic baseline" for s in data
        scenarios = reduce(vcat, (reshape(ξ.departure .+ ξ.arrival, 1, :) for ξ in s.extra.scenarios))
        timed = @timed solve_aircraft_routing(s.instance; silent=true)
        det_routes, _, _ = timed.value
        det_full = full_cost(
            Vector{Route}(det_routes),
            scenarios,
            s.instance;
            delay_cost_function=bench.delay_cost_function,
        )
        expert_full = full_cost(
            s.extra.routes, scenarios, s.instance; delay_cost_function=bench.delay_cost_function
        )
        total_cost += det_full
        total_time += timed.time
        if expert_full > 0
            total_gap += (det_full - expert_full) / abs(expert_full)
            nb_gap += 1
        end
    end
    n = length(data)
    return (;
        avg_cost=total_cost / n,
        avg_time=total_time / n,
        full_cost_gap=nb_gap > 0 ? total_gap / nb_gap : NaN,
    )
end

# ╔═╡ a8cdcca5-b5b8-4b69-8c7a-4e9542fbd1b8
"""
Compute the in-sample SAA gap of `policy`'s current weights for each sample of `data`
separately, returning a `NamedTuple` of three vectors (`expert_cost`, `surrogate_cost`,
`gap_percent`), one entry per sample, in the same order as `data`. Powers the per-instance
breakdown table in Section 8, `DFLB.compute_gap` only returns the mean over `data`.
Returns `nothing` instead if `DFLB.objective_value` (Section 1's exercise) is still
unimplemented and yields `missing`, so a caller can show a status line instead of a
`MethodError` from pushing `missing` into a `Vector{Float64}`.
"""
function per_instance_gaps(data, policy::DFLPolicy)
    expert_cost = Float64[]
    surrogate_cost = Float64[]
    gap_percent = Float64[]
    for s in data
        target_obj = DFLB.objective_value(saa, s)
        ismissing(target_obj) && return nothing
        θ = policy.statistical_model(s.x)
        y = policy.maximizer(θ; s.context...)
        obj = DFLB.objective_value(saa, s, y)
        ismissing(obj) && return nothing
        push!(expert_cost, target_obj)
        push!(surrogate_cost, obj)
        push!(gap_percent, 100 * (obj - target_obj) / abs(target_obj))
    end
    return (; expert_cost, surrogate_cost, gap_percent)
end

# ╔═╡ 6c62cf45-7165-48ab-a483-bb14a1c025e3
"""
Plot the per-epoch training loss and validation full cost gap (already in percent) on two
panels, from `epochs`, `loss` and `gap`, exactly the vectors returned by
`get(history, :training_loss)` and `get(history, :val_gap)` after `train_policy!`.
"""
function plot_training_curves(epochs, loss, gap; title="")
    fig = Figure(; size=(900, 340))
    ax1 = Axis(
        fig[1, 1]; xlabel="Epoch", ylabel="Fenchel-Young loss", title="$title training loss"
    )
    lines!(ax1, epochs, loss)
    scatter!(ax1, epochs, loss)
    ax2 = Axis(
        fig[1, 2];
        xlabel="Epoch",
        ylabel="Validation full cost gap (%)",
        title="$title validation gap",
    )
    hlines!(ax2, [0.0]; color=:black, linewidth=1)
    lines!(ax2, epochs, gap)
    scatter!(ax2, epochs, gap)
    return fig
end

# ╔═╡ 3ae0e8f2-3b0e-4e6a-9d8e-3f0a5b9c6d21
"""
Plot a grouped bar chart comparing the expert's `ȳ` and the surrogate's `ŷ` (both binary
arc-selection vectors, one entry per interior arc) for the first `limit` arcs, so agreement
and disagreement are readable arc by arc.
"""
function plot_binary_agreement(ȳ, ŷ; limit=40, title="")
    n = min(limit, length(ȳ))
    fig = Figure(; size=(900, 260))
    ax = Axis(
        fig[1, 1];
        xlabel="Interior arc index (first $n arcs)",
        ylabel="Selected",
        title=title,
        yticks=([0, 1], ["0", "1"]),
    )
    xs = 1:n
    barplot!(ax, xs .- 0.15, Float64.(ȳ[1:n]); width=0.3, label="expert ȳ")
    barplot!(ax, xs .+ 0.15, Float64.(ŷ[1:n]); width=0.3, label="surrogate ŷ")
    axislegend(ax; position=:rt)
    return fig
end

# ╔═╡ 6c2b1e9f-9a4a-4e6a-8e2f-1c7d9a0b3e58
"""
Plot a bar chart of full cost, and a second one of runtime (log scale, seconds), one bar
per method in `methods`, from `costs` (cost units) and `times` (seconds).
"""
function plot_quality_runtime(methods, costs, times)
    fig = Figure(; size=(900, 340))
    ax1 = Axis(
        fig[1, 1];
        xlabel="Method",
        ylabel="Full cost (cost units)",
        xticks=(1:length(methods), methods),
        title="Decision quality",
    )
    barplot!(ax1, 1:length(methods), costs)
    ax2 = Axis(
        fig[1, 2];
        xlabel="Method",
        ylabel="Runtime (s, log scale)",
        yscale=log10,
        xticks=(1:length(methods), methods),
        title="Runtime",
    )
    barplot!(ax2, 1:length(methods), max.(times, 1e-4))
    return fig
end

# ╔═╡ 7f3a2c8e-5b6d-4e91-8a0c-2d9e4b7f1a63
"""
Plot the median slack quantile (feature row 7, the `p=0.5` slack quantile, in minutes)
across the interior arcs of one instance's raw feature matrix `x`, sorted ascending, so
tight and slack connections are easy to tell apart.
"""
function plot_slack_quantiles(x; title="")
    values = sort(x[7, :])
    fig = Figure(; size=(900, 300))
    ax = Axis(
        fig[1, 1];
        xlabel="Arc rank (sorted by slack)",
        ylabel="Median slack quantile (minutes)",
        title=title,
    )
    lines!(ax, 1:length(values), values)
    hlines!(ax, [0.0]; color=:red, linestyle=:dash, label="zero slack")
    axislegend(ax; position=:lt)
    return fig
end

# ╔═╡ 71ca1911-4185-4c84-8fd9-f22a26af44f1
# k, the paper's taboo list size: the dive backtracks and stops once the list of fixed
# columns that led to infeasibility exceeds k. The paper uses 15.
const TABOO_LIST_SIZE = 15

# ╔═╡ 4de274b1-2a31-43aa-99ed-2002850c301d
"Chain warm start, column generation and one dive, exactly part 1's `expert_routes`, returning `nothing` instead of erroring if the dive is infeasible."
function sta_expert_routes(schedule, root_delays, delay_cost_function)
    warm_start, _, _ = solve_aircraft_routing(schedule; silent=true)
    isempty(warm_start) && error("deterministic warm start was infeasible")
    cg = stochastic_column_generation(
        schedule,
        warm_start;
        root_delays,
        delay_cost_function,
        max_nb_columns=10_000,
        tol=1e-6,
        silent=true,
    )
    cg.feasible || error("stochastic column generation was infeasible")
    dived, feasible = diving_heuristic_with_backtracking!(
        schedule,
        cg.columns,
        root_delays,
        cg.dual_values,
        TABOO_LIST_SIZE;
        model_builder=highs_model,
        delay_cost_function,
        silent=true,
    )
    feasible && return Vector{Route}(dived)
    return nothing
end

# ╔═╡ 3ba180e4-734d-4b61-827f-85034801eed2
"Target policy: label one instance from all K SAA `scenarios` at once, or return `DataSample[]` if unlabelable."
function expert_target_policy(ctx_sample::DataSample, scenarios::AbstractVector)
    result = DataSample[]
    instance = ctx_sample.instance
    departure_root_delays = reduce(vcat, (reshape(ξ.departure, 1, :) for ξ in scenarios))  # stack into a K by L matrix
    arrival_root_delays = reduce(vcat, (reshape(ξ.arrival, 1, :) for ξ in scenarios))
    root_delays = departure_root_delays .+ arrival_root_delays  # combine into one root delay matrix
    timed = @timed sta_expert_routes(instance, root_delays, bench.delay_cost_function)
    routes = timed.value
    if routes !== nothing
        y = missing  # TODO: decode_arc_solution_from_routes(routes, instance)
        x_raw = missing  # TODO: compute_features(instance, departure_root_delays, arrival_root_delays)
        if !ismissing(y) && !ismissing(x_raw)
            result = [
                DataSample(
                    ctx_sample;
                    x=scale_features(x_raw),
                    y,
                    extra=(;
                        ctx_sample.extra..., x_raw, scenarios, routes, expert_seconds=timed.time
                    ),
                ),
            ]
        end
    end
    return result
end

# ╔═╡ 515483da-d3d8-40bd-baaa-a84a9981a91f
begin
    train_data_bundle = if train_data_request === nothing
        (; status=:idle, source=nothing, data=DataSample[], runtime=missing, error=nothing, nb_skipped=0, load_error=nothing)
    elseif train_data_result_cache[].request == train_data_request &&
           train_data_result_cache[].click == train_data_click
        train_data_result_cache[].result
    else
        train_data_computed = try
            loaded, load_error = session1_samples(train_data_request.path)
            if loaded === nothing
                train_data_timed = @timed DFLB.generate_dataset(
                    saa, 30; target_policy=expert_target_policy, seed=40_000
                )
                (; status=:complete, source=:fallback, data=train_data_timed.value, runtime=train_data_timed.time, error=nothing, nb_skipped=30 - length(train_data_timed.value), load_error)
            else
                train_data_timed = @timed loaded
                (; status=:complete, source=:file, data=train_data_timed.value, runtime=train_data_timed.time, error=nothing, nb_skipped=0, load_error=nothing)
            end
        catch caught_error
            (; status=:failed, source=nothing, data=DataSample[], runtime=missing,
               error=sprint(showerror, caught_error), nb_skipped=0, load_error=nothing)
        end
        train_data_result_cache[] =
            (click=train_data_click, request=train_data_request, result=train_data_computed)
        train_data_computed
    end
    (; status=train_data_bundle.status)
end

# ╔═╡ 36a4aed5-d6fc-4085-8abf-50ed88a147e7
train_data = train_data_bundle.data

# ╔═╡ 5e319f13-f9f1-431a-a96e-cee2c76da149
if train_data_bundle.status == :complete && isempty(train_data) &&
   train_data_bundle.nb_skipped > 0
    still_missing(md"Every instance was skipped, implement `expert_target_policy` (Section 4).")
else
    Markdown.parse(
        train_data_bundle.status == :idle ?
        "**Status: idle.** Press \"Load the training set\" above." :
        train_data_bundle.status == :failed ?
        "**Status: failed.** " * train_data_bundle.error :
        "**Status: complete.** Loaded $(length(train_data)) sample(s) from " *
        (train_data_bundle.source == :file ?
         "`data/practice_session_1_dataset.jld2`" :
         "a freshly generated fallback set (" *
         (train_data_bundle.load_error === nothing ?
          "the file was not found" :
          "`data/practice_session_1_dataset.jld2` could not be read: " * train_data_bundle.load_error) *
         ")") *
        " in $(format_seconds(train_data_bundle.runtime)), feature matrix size " *
        (isempty(train_data) ? "n/a" : string(size(train_data[1].x))) * "." *
        (train_data_bundle.nb_skipped > 0 ?
         " $(train_data_bundle.nb_skipped) instance$(train_data_bundle.nb_skipped == 1 ? " was" : "s were") skipped: diving found no feasible integer solution at this taboo list size." :
         ""),
    )
end

# ╔═╡ 4ecf80d7-3cf0-4662-8b11-e8df65442065
begin
    test_bundle = if test_request === nothing
        (; status=:idle, data=DataSample[], runtime=missing, error=nothing, nb_skipped=0)
    elseif test_result_cache[].request == test_request &&
           test_result_cache[].click == test_click
        test_result_cache[].result
    else
        test_computed = try
            test_timed = @timed DFLB.generate_dataset(
                saa, test_request.n; target_policy=expert_target_policy, seed=test_request.seed
            )
            (; status=:complete, data=test_timed.value, runtime=test_timed.time, error=nothing, nb_skipped=test_request.n - length(test_timed.value))
        catch caught_error
            (; status=:failed, data=DataSample[], runtime=missing, error=sprint(showerror, caught_error), nb_skipped=0)
        end
        test_result_cache[] = (click=test_click, request=test_request, result=test_computed)
        test_computed
    end
    (; status=test_bundle.status)
end

# ╔═╡ acab1bab-d89f-41de-b089-7b7e184fc016
test_data = test_bundle.data

# ╔═╡ 7c9a1e02-7f3f-4b7f-9a04-1e2f0c6d7a11
# The slider above is drawn before the test set exists and stays at 1:1:8 whatever the
# actual test set size ends up being (unlabelable instances can make it smaller, see
# `expert_target_policy`), so clamp it down to a valid index of `test_data` here, once, and
# use `session2_idx` (not `session2_inspect_idx`) everywhere in sections 10 to 12.
session2_idx = isempty(test_data) ? 1 : clamp(session2_inspect_idx, 1, length(test_data))

# ╔═╡ de459fab-2f3c-4c71-b830-536b3339ca40
Markdown.parse(
    """
    ## 10. Feature inspection

    Median slack quantile (minutes) across interior arcs of **test instance #$(session2_idx)**, sorted, tight connections near or below zero.
    """,
)

# ╔═╡ 002c6c80-8eb6-4dfd-8eb9-5941ff68f2fa
gap_metric = FunctionMetric(:val_gap, test_data) do ctx, data
    isempty(data) && return NaN
    gap = DFLB.compute_gap(saa, data, ctx.policy.statistical_model, ctx.policy.maximizer)
    # compute_gap yields `missing` while `objective_value` (Section 1's exercise) is still
    # unimplemented, NaN keeps the training loop and its Float64 history vectors working
    # instead of failing on a missing value.
    return ismissing(gap) ? NaN : 100 * gap
end

# ╔═╡ 89f02fe9-be13-4c7d-9140-c00d724e083f
begin
    # test_id is included so that a fresh button press after regenerating the test set
    # (Section 6) is recognized as a genuinely new request, not accidentally matched
    # against a validation curve computed for the previous test set. `test_click` (the
    # counter itself) is a more robust identity than `objectid(test_data)`: it strictly
    # increases on every press and does not depend on object identity surviving Pluto's
    # distributed workspace boundary, `length(test_data)` is kept alongside it so a change
    # in how many instances were labeled (Section 6's skip semantics) is also visible here.
    test_id = (length(test_data), test_click)
    train_current_request = (
        epochs=session2_epochs, nb_samples=session2_nb_samples, nb_train=length(train_data), test_id
    )
    if train_click > train_request_cache[].click
        train_request_cache[] = (click=train_click, request=train_current_request)
    end
    train_request = train_request_cache[].request
end

# ╔═╡ cb6fe744-e060-4c1e-9823-912f9f44467e
begin
    train_bundle = if train_request === nothing
        (; status=:idle, policy=nothing, history=nothing, runtime=missing, error=nothing)
    elseif train_result_cache[].request == train_request &&
           train_result_cache[].click == train_click
        train_result_cache[].result
    elseif isempty(train_data)
        train_not_implemented =
            (; status=:not_implemented, policy=nothing, history=nothing, runtime=missing, error=nothing)
        train_result_cache[] =
            (click=train_click, request=train_request, result=train_not_implemented)
        train_not_implemented
    else
        train_computed = try
            # deepcopy initial_model rather than calling generate_statistical_model a second
            # time, exactly the pattern the knapsack demo uses, so training always starts
            # from the same seeded weights shown in Section 3 above.
            policy = DFLPolicy(deepcopy(initial_model), maximizer)
            algo = PerturbedFenchelYoungLossImitation(;
                nb_samples=train_request.nb_samples, ε=0.01, threaded=true, seed=3
            )
            train_timed = @timed train_policy!(
                algo, policy, train_data; epochs=train_request.epochs, metrics=(gap_metric,)
            )
            (; status=:complete, policy=policy, history=train_timed.value, runtime=train_timed.time, error=nothing)
        catch caught_error
            (; status=:failed, policy=nothing, history=nothing, runtime=missing,
               error=sprint(showerror, caught_error))
        end
        train_result_cache[] = (click=train_click, request=train_request, result=train_computed)
        train_computed
    end
    (; status=train_bundle.status)
end

# ╔═╡ 642963eb-6bb3-4c6e-a7f2-0dfcd228d12b
Markdown.parse(
    train_bundle.status == :idle ?
    "**Status: idle.** Press \"Train the surrogate\" above." :
    train_bundle.status == :not_implemented ?
    "**Status: waiting on data.** Load the training set (section 5) and make sure `expert_target_policy` above is implemented, then press the button again." :
    train_bundle.status == :failed ?
    "**Status: failed.** " * train_bundle.error :
    "**Status: complete.** Trained for $(train_request.epochs) epoch(s) in " *
    "$(format_seconds(train_bundle.runtime)).",
)

# ╔═╡ 55a4c6de-0873-461f-a57f-16572782e7fa
train_bundle.status != :complete ? md"*Train the surrogate above to see the loss and gap curves.*" :
isempty(test_data) ? md"*Generate the fresh test set to get a validation curve.*" :
let
    (epochs_logged, losses) = get(train_bundle.history, :training_loss)
    (_, gaps) = get(train_bundle.history, :val_gap)
    plot_training_curves(epochs_logged, losses, gaps)
end

# ╔═╡ fc435c33-5591-4fb2-b90d-d0f807b64090
train_bundle.status != :complete ? md"" :
isempty(test_data) ? md"*Generate the fresh test set to get a validation curve.*" :
let
    (_, gaps) = get(train_bundle.history, :val_gap)
    Markdown.parse(
        "In-sample validation gap: **$(round(gaps[1]; digits=2)) %** before training (epoch 0), " *
        "**$(round(gaps[end]; digits=2)) %** at the final epoch. " *
        "This is the training-time metric (mean over the 8 test instances' own SAA scenarios), the out-of-sample number in Section 9 is the fair comparison. " *
        "Every table below uses the final policy, not the best epoch, so this is the headline number to compare against them. " *
        "The fresh test set doubles as this training curve's monitoring set, a real project would hold out a separate validation set so early stopping does not leak information from the set used for the final reported numbers.",
    )
end

# ╔═╡ f3abed9f-41bb-490f-a41a-78ec6bc936ea
if test_bundle.status == :complete && isempty(test_data) && test_bundle.nb_skipped > 0
    still_missing(md"Every instance was skipped, implement `expert_target_policy` (Section 4).")
else
    Markdown.parse(
        test_bundle.status == :idle ?
        "**Status: idle.** Press \"Generate the fresh test set\" above." :
        test_bundle.status == :failed ?
        "**Status: failed.** " * test_bundle.error :
        "**Status: complete.** Generated $(length(test_data)) fresh test instance(s) in " *
        "$(format_seconds(test_bundle.runtime))." *
        (test_bundle.nb_skipped > 0 ?
         " $(test_bundle.nb_skipped) instance$(test_bundle.nb_skipped == 1 ? " was" : "s were") skipped: diving found no feasible integer solution at this taboo list size." :
         ""),
    )
end

# ╔═╡ 6a1f2b3c-8d4e-4f5a-9b6c-7d8e9f0a1b2c
session2_ready = train_bundle.status == :complete && test_bundle.status == :complete && !isempty(test_data)

# ╔═╡ 258652e8-6624-4c2a-a446-f705d3e07639
session2_per_instance = !session2_ready ? nothing : per_instance_gaps(test_data, train_bundle.policy)

# ╔═╡ 77648d50-0b51-4dc9-8f38-0825a11d2960
begin
    session2_table = !session2_ready || session2_per_instance === nothing ?
        nothing :
        let
            expert_cost = average_expert_full_cost(test_data)
            expert_time = mean(s.extra.expert_seconds for s in test_data)
            surrogate_cost = average_surrogate_full_cost(test_data, train_bundle.policy)
            surrogate_time = mean(
                (@elapsed train_bundle.policy(s.x; s.context...)) for s in test_data
            )
            det = deterministic_baseline(test_data)
            (
                method=[
                    "expert (stochastic OR, practice session 1)",
                    "surrogate (DFL, this practice session)",
                    "deterministic (no stochasticity)",
                ],
                full_cost=[expert_cost, surrogate_cost, det.avg_cost],
                runtime_seconds=[expert_time, surrogate_time, det.avg_time],
                full_cost_gap_percent=[
                    0.0,
                    mean(session2_per_instance.gap_percent),
                    100 * det.full_cost_gap,
                ],
            )
        end
    session2_table
end

# ╔═╡ 6d731846-a2c8-4126-b978-43f34086ae1a
begin
    session2_table_md = if !session2_ready
        "*Generate the fresh test set and train the surrogate above to fill in this table.*"
    elseif session2_table === nothing
        "*Status: waiting on the exercise.* Implement `objective_value` first (Section 1), the comparison table's gap column needs it."
    else
        header = "| method | full cost | runtime | full cost gap |\n|---|---|---|---|"
        rows = join(
            (
                "| $(session2_table.method[i]) | $(format_cost(session2_table.full_cost[i])) | " *
                "$(format_seconds_with_ms(session2_table.runtime_seconds[i])) | " *
                "$(format_pct(session2_table.full_cost_gap_percent[i])) |" for
                i in eachindex(session2_table.method)
            ),
            "\n",
        )
        header * "\n" * rows
    end
    Markdown.parse(session2_table_md)
end

# ╔═╡ 205b19d7-f4db-4d91-9e5f-8c3cd79c5f40
session2_speedup =
    session2_table === nothing ? missing :
    speedup_and_gap(
        session2_table.runtime_seconds[1], session2_table.runtime_seconds[2], session2_table.full_cost_gap_percent[2]
    )

# ╔═╡ b815dc98-485c-4f00-96f6-a3675632b9d0
Markdown.parse(
    """
    ## 13. Map decision-focused learning to your own problem

    Five questions to ask about a problem you actually work on, tied to the numbers you just produced above.

    1. **What is the decision?** Here, one route per aircraft, decoded from a binary arc selection.
       In your problem: whatever a deterministic solver already outputs.
    2. **What is uncertain?** Here, delay scenarios, entering the surrogate as distributional slack features and entering the expert as the scenarios it optimizes over.
       In your problem: whatever varies between the moment you decide and the moment the outcome is known.
    3. **What is the downstream objective?** Here, the full cost gap in the table above$(session2_table === nothing ? "" : " ($(round(session2_table.full_cost_gap_percent[2]; digits=1)) % for the surrogate versus $(round(session2_table.full_cost_gap_percent[3]; digits=1)) % for a deterministic baseline)").
       In your problem: whatever a domain expert would use to judge a decision, not a proxy metric.
    4. **Is there already a solver whose parameters could be learned?** Here, the deterministic edge MIP, unchanged, only its cost coefficients θ are learned$(session2_speedup === missing ? "" : ", giving a $(round(session2_speedup.speedup; digits=1))x speedup over the expert in the table above").
       In your problem: look for a fast deterministic solver next to a slow, high-quality one, DFL turns the slow one's decisions into training labels for the fast one's cost coefficients.
    5. **What would your dataset file look like?** Here, `data/practice_session_1_dataset.jld2`: one row per instance, holding whatever a fresh instance needs to be regenerated plus the expert's routes.
       In your problem: whatever a from-scratch benchmark's `generate_instance` needs, plus the label your `target_policy` would have produced.
    """,
)

# ╔═╡ 0ce675f0-34d1-4b6b-8520-8b3e8373fb63
begin
    session2_per_instance_md = if !session2_ready
        "*Generate the fresh test set and train the surrogate above to fill in this table.*"
    elseif session2_per_instance === nothing
        "*Status: waiting on the exercise.* Implement `objective_value` first (Section 1), the per-instance gap needs it."
    else
        pi_header = "| instance | expert cost | surrogate cost | gap |\n|---|---|---|---|"
        pi_rows = join(
            (
                "| $i | $(format_cost(session2_per_instance.expert_cost[i])) | " *
                "$(format_cost(session2_per_instance.surrogate_cost[i])) | " *
                "$(format_pct(session2_per_instance.gap_percent[i])) |" for
                i in eachindex(session2_per_instance.gap_percent)
            ),
            "\n",
        )
        pi_summary =
            "\n\nMean gap: $(format_pct(mean(session2_per_instance.gap_percent))), " *
            "median gap: $(format_pct(median(session2_per_instance.gap_percent)))."
        pi_header * "\n" * pi_rows * pi_summary
    end
    Markdown.parse(session2_per_instance_md)
end

# ╔═╡ d5f3c1c4-43fc-46d1-a8bc-5a3747a0826f
if session2_speedup !== missing
    if !isfinite(session2_speedup.speedup) || session2_speedup.speedup <= 0
        keep_working(md"The speedup ratio should be a positive, finite number (expert time divided by surrogate time).")
    else
        Markdown.parse(
            "Surrogate speedup over the expert: **$(round(session2_speedup.speedup; digits=1))x**, " *
            "full cost gap: **$(round(session2_speedup.full_cost_gap_percent; digits=2)) %**. " *
            "The surrogate's timed prediction excludes feature computation, which still needs sampled delay scenarios and per-arc slack quantiles, so it is not entirely free in the way this number alone suggests.",
        )
    end
elseif !session2_ready
    md"*Generate the fresh test set and train the surrogate above to see the speedup and gap.*"
elseif session2_table === nothing
    still_missing(md"Implement `objective_value` first (Section 1), the speedup and gap need it.")
else
    still_missing(md"Implement `speedup_and_gap` above, then this will show the speedup and gap.")
end

# ╔═╡ a03f4dcb-c3d8-4705-a009-4eb33610b0b0
!session2_ready ?
md"*Generate the fresh test set and train the surrogate above to see the quality vs runtime chart.*" :
session2_table === nothing ?
md"*Status: waiting on the exercise.* Implement `objective_value` first (Section 1), the quality vs runtime chart needs it." :
plot_quality_runtime(session2_table.method, session2_table.full_cost, session2_table.runtime_seconds)

# ╔═╡ 55462688-f5b1-4363-b1a4-0acf0007720b
session2_oos =
    !session2_ready ? missing :
    let
        results = []
        @progress "out-of-sample evaluation" for (i, s) in enumerate(test_data)
            push!(
                results,
                out_of_sample_gap(s, train_bundle.policy; nb_scenarios=30, seed=95_000 + 1000i),
            )
        end
        (;
            expert_cost=mean(r.expert_cost for r in results),
            surrogate_cost=mean(r.surrogate_cost for r in results),
            gap_percent=mean(r.gap_percent for r in results),
        )
    end

# ╔═╡ cd353de6-ee87-41c8-8711-3e3e60956291
session2_oos === missing ?
md"*Generate the fresh test set and train the surrogate above to see the out-of-sample comparison.*" :
Markdown.parse(
    "Out of sample (30 fresh delay draws per test instance, never seen by either the expert or the surrogate): " *
    "expert full cost $(format_cost(session2_oos.expert_cost)), surrogate full cost $(format_cost(session2_oos.surrogate_cost)), " *
    "surrogate gap **$(format_pct(session2_oos.gap_percent))**. " *
    (ismissing(session2_oos.gap_percent) ?
     "*Status: waiting on the exercise.* Implement `objective_value` first (Section 1), the out-of-sample gap needs it. " :
     "") *
    "This number is the fair one, neither decision has an information advantage on the scenarios it is scored on.",
)

# ╔═╡ d0c8e5b5-9a2f-4684-8061-02f9ca86b8e7
!session2_ready ?
md"*Generate the fresh test set and train the surrogate above to see the route comparison.*" :
let
    s = test_data[session2_idx]
    surrogate_routes = surrogate_routes_for(s, train_bundle.policy)
    plot_gantt(
        s.instance,
        s.extra.routes;
        root_delays=reduce(vcat, (reshape(ξ.departure .+ ξ.arrival, 1, :) for ξ in s.extra.scenarios)),
        delay_cost_function=bench.delay_cost_function,
        comparison_routes=surrogate_routes,
        title="Expert (stochastic OR)",
        comparison_label="Surrogate (DFL)",
    )
end

# ╔═╡ e7263d34-4953-47a7-a0b6-19ad6000ed6d
session2_agreement =
    !session2_ready ? missing :
    let
        s = test_data[session2_idx]
        θ = train_bundle.policy.statistical_model(s.x)
        surrogate_agreement(θ, s.y, maximizer, s.instance)
    end

# ╔═╡ 43452f4a-6e26-4c5f-b1c8-1b806a830b25
session2_agreement === missing ?
md"*Generate the fresh test set and train the surrogate above to see the arc selection comparison.*" :
let
    s = test_data[session2_idx]
    plot_binary_agreement(
        s.y[1:(s.instance.nb_interior_arcs)],
        session2_agreement.ŷ;
        title="test instance #$(session2_idx), agreement $(round(100 * session2_agreement.agreement; digits=1)) %",
    )
end

# ╔═╡ 6c2018e1-2a9f-405e-914f-f74ff3a64ab4
if !session2_ready
    md"*Generate the fresh test set and train the surrogate above first.*"
elseif ismissing(session2_agreement)
    still_missing()
elseif length(session2_agreement.ŷ) != test_data[session2_idx].instance.nb_interior_arcs ||
       !(0 <= session2_agreement.agreement <= 1)
    keep_working(md"`ŷ` should have one entry per interior arc, and `agreement` should be a fraction between 0 and 1.")
else
    correct()
end

# ╔═╡ 8730a999-4b02-4eb3-9e3e-878a05dabecc
(test_bundle.status != :complete || isempty(test_data)) ?
md"*Generate the fresh test set above to see the feature chart for this instance.*" :
plot_slack_quantiles(
    test_data[session2_idx].extra.x_raw; title="test instance #$(session2_idx)"
)

# ╔═╡ a8eb7ae6-a5e2-4399-8ecd-56fe653fa4b5
try
    warm_bench = StochasticTailAssignmentBenchmark(; nb_legs=12, nb_scenarios=3)
    warm_saa = SampleAverageApproximation(warm_bench, 3)
    warm_data = DFLB.generate_dataset(warm_saa, 1; target_policy=expert_target_policy, seed=1)
    if !isempty(warm_data)
        warm_policy = DFLPolicy(
            DFLB.generate_statistical_model(warm_bench; seed=1),
            DFLB.generate_maximizer(warm_bench),
        )
        train_policy!(
            PerturbedFenchelYoungLossImitation(;
                nb_samples=2, ε=0.01, threaded=true, seed=1
            ),
            warm_policy,
            warm_data;
            epochs=1,
        )
    end
    "JIT warm-up done (12 legs, 3 scenarios, 1 epoch)."
catch caught_error
    "JIT warm-up skipped (" * sprint(showerror, caught_error) * "), the first real run below will absorb that one-time JIT cost instead."
end

# ╔═╡ Cell order:
# ╠═5e9a2000-a1b2-4c3d-8e9f-000000000001
# ╠═5e9a2000-a1b2-4c3d-8e9f-000000000002
# ╟─3e1088c9-3990-4947-a8dd-cc7bd1e0d374
# ╟─7dae61cc-9ce2-4319-96ca-1b91fce6c752
# ╠═564cb970-9ec4-4a8e-9f2f-7df7d71547d0
# ╟─8e38e69f-579b-4969-9f88-47fd28705c2e
# ╠═fbf14a46-1fe9-441f-8ec8-d67f8647c92f
# ╠═43e28775-c9b3-4919-947d-e7f228a74e7a
# ╟─faeda526-988b-433d-8281-fd4ae577d835
# ╠═d99a713c-8f04-4098-afec-faf2967435da
# ╠═cda0b222-ff8e-47d0-b150-d3046e07a69b
# ╟─809a18c1-180a-4d0d-8803-0610dbb08953
# ╠═44eb3212-9db8-470c-bde9-be8ec1557984
# ╟─0d275dbb-5373-446d-a9f8-a343b551964a
# ╟─631741d6-a2ec-41af-a75d-0b72c282d083
# ╟─30a483d8-733a-40e8-bcc3-384411cb2153
# ╠═4f6ff23a-4401-46f5-ad1c-8551d78234f6
# ╠═b1edca94-2426-47e4-af6d-bf6cb467942a
# ╟─86e29a7e-7f06-46f4-a85a-670ab4e6d778
# ╠═d36172e6-bd2a-43eb-85c9-8a9e0cb79d4e
# ╟─090416db-2bdf-452f-8464-62361bb7c10d
# ╟─b7d60f6e-503f-4bc3-bb92-4663fb22e345
# ╠═730cfe08-70ac-429e-a4a8-9cd845a21a01
# ╠═bc1c5b4f-9941-4289-94d3-89a2b4adc068
# ╟─d91e8366-ac43-4b95-9302-20cfc715277e
# ╟─07fda6d3-52cc-4ef0-807a-a5d3c6e8e59c
# ╟─36dbc492-ba6c-4001-8b25-6533ec88285f
# ╟─4aedb21d-9dde-461d-965e-a4b517ffec8d
# ╟─2b6a3712-9928-4bdf-9a6f-8c11dbd597a0
# ╟─5028aadb-b2a0-495f-a8bf-3dbb0fcf07fc
# ╟─1a812d10-7702-49c8-ac98-360ff4ffb7e3
# ╟─52311ed7-a07b-45f8-93c8-a15e69548399
# ╟─afab3069-b909-466f-8f3b-5eaeb20333fc
# ╟─9edb2355-846f-4e26-91e5-10cd6c9432e2
# ╟─2ac77c30-f39b-4370-9868-88a207a1af0b
# ╠═c09f83bc-a8fe-4399-8ae7-9d254bf9b731
# ╠═d050236e-e869-4195-9628-f1d24e24cc83
# ╠═868fcead-a545-43f3-97d7-c4925d50130e
# ╠═f17db97d-7036-4ccc-b061-a52253e91da2
# ╟─4508eed1-2cf0-4b6d-9056-26e8e4d5d795
# ╟─81169384-a0ad-46c7-a0c5-b10bf9e80eaf
# ╠═5e9a03df-1d1b-487d-9ade-3b8b5a02f06f
# ╟─ef891c26-7163-4638-8e75-4c7758adb0b9
# ╠═86d5a09a-40c7-4071-9dbb-d56f8623e74f
# ╠═e5b0ac6e-6e0b-4973-b4dc-16321067261f
# ╟─1cbfa417-8dc1-436e-a305-fea810813720
# ╠═85dc4872-6f1f-4aa6-b032-9fc49f7ca849
# ╟─abf0a855-3c55-468a-bc7c-f82a41e8cab1
# ╠═460600a5-08b2-463d-ab9b-81971ae151ac
# ╠═63301851-f92f-47f1-9785-879bc2d11ce8
# ╠═e458661a-de1d-4052-9bb4-dc1a1cdde506
# ╠═f46495af-e2ba-477e-999f-e136ea860779
# ╟─a454f17b-b84f-4993-9ec6-b8f9cd2cb2f6
# ╠═3f72a5c3-cc00-4d58-8679-d4468ec4e0d6
# ╠═7af83fe2-0fde-4e98-9feb-bd55a64417d0
# ╠═21d2593d-d77c-4456-8486-61763c298c5b
# ╠═3a27631a-3c48-4d69-94a6-8ad71a36274f
# ╟─95f7c78d-302d-4f43-b582-47870a1d027e
# ╠═3eeeb774-674c-44e7-929a-1e33c9cc5c82
# ╠═cb9eb966-1d16-478a-8bee-091137b9691a
# ╠═164024df-6a15-4ccf-8382-3a61ad0247e8
# ╠═a8203ace-eeef-4861-9516-f66c3fff40a2
# ╠═17869e8d-d8e0-4bf9-a766-22493a3b9e89
# ╟─527eaa81-0677-46ae-a04b-7ef57b294e9a
# ╟─d7a27f14-bd9c-4012-b6ea-e926ac2936d3
# ╠═28e465ca-a503-4806-9043-3d3d5b7591d3
# ╟─1d2fde83-9469-4cee-a74e-3c1040998546
# ╟─191fa21a-4937-451d-94f6-ab3590f145ce
# ╠═3ba180e4-734d-4b61-827f-85034801eed2
# ╟─7c7d8f05-6d9b-417e-8ec3-22347b3a7c34
# ╟─b6a7faeb-6dd8-4232-a773-070c2276da80
# ╟─50e05260-cd17-46fb-aa60-57d6092e98c6
# ╟─c2cd6fa1-18c8-4377-9012-21cf0c66241b
# ╟─fd1a924b-2305-4421-92d3-a39c50608e10
# ╟─b299411f-989b-400e-a090-085c0ddfbf7b
# ╟─515483da-d3d8-40bd-baaa-a84a9981a91f
# ╠═36a4aed5-d6fc-4085-8abf-50ed88a147e7
# ╟─5e319f13-f9f1-431a-a96e-cee2c76da149
# ╟─3bfc7519-2824-40bc-a780-c4a25ff0aace
# ╟─3932b390-2f88-4678-8be0-418a03b73134
# ╟─ac26961f-0a3f-40e6-8d7f-10713c00191c
# ╟─a19e0a28-cc7a-4339-ac3c-c0a430de1df3
# ╟─03a6b02b-787e-4614-88bd-960eb900c8f7
# ╟─4ecf80d7-3cf0-4662-8b11-e8df65442065
# ╠═acab1bab-d89f-41de-b089-7b7e184fc016
# ╟─7c9a1e02-7f3f-4b7f-9a04-1e2f0c6d7a11
# ╟─f3abed9f-41bb-490f-a41a-78ec6bc936ea
# ╟─226419ca-751b-4d03-b0fe-30a96040cea4
# ╟─876c9a36-5afc-4ea4-8632-f4d565690158
# ╟─e8fdce16-5fc7-4893-b29f-29e709aeeb11
# ╟─1ef9812b-6b96-4bbd-a947-6b28793c8e12
# ╠═002c6c80-8eb6-4dfd-8eb9-5941ff68f2fa
# ╟─997a8c2c-2d25-477d-ac05-26ad117578b3
# ╟─89f02fe9-be13-4c7d-9140-c00d724e083f
# ╟─cb6fe744-e060-4c1e-9823-912f9f44467e
# ╟─642963eb-6bb3-4c6e-a7f2-0dfcd228d12b
# ╠═55a4c6de-0873-461f-a57f-16572782e7fa
# ╟─fc435c33-5591-4fb2-b90d-d0f807b64090
# ╟─04a46f0d-62e0-46bf-884d-abbf02280113
# ╟─46484b09-fbb4-43ec-b765-81847383f4cf
# ╠═6a1f2b3c-8d4e-4f5a-9b6c-7d8e9f0a1b2c
# ╟─77648d50-0b51-4dc9-8f38-0825a11d2960
# ╟─6d731846-a2c8-4126-b978-43f34086ae1a
# ╟─251fdd64-1fc9-42f4-8e63-f0595e48af7a
# ╠═258652e8-6624-4c2a-a446-f705d3e07639
# ╟─0ce675f0-34d1-4b6b-8520-8b3e8373fb63
# ╟─8eda63fe-701b-4b22-8d07-05bd9dadea74
# ╠═994d7f21-3472-4a31-9ad8-5b77ac17e77f
# ╟─9245bd07-4761-49ed-8ec2-01f05bc61927
# ╠═205b19d7-f4db-4d91-9e5f-8c3cd79c5f40
# ╟─d5f3c1c4-43fc-46d1-a8bc-5a3747a0826f
# ╟─a03f4dcb-c3d8-4705-a009-4eb33610b0b0
# ╟─ef411983-077d-4842-ab70-d5f1810e40ee
# ╟─1a23c62b-3f4b-4c06-b933-d5c1a2b899cc
# ╠═55462688-f5b1-4363-b1a4-0acf0007720b
# ╟─cd353de6-ee87-41c8-8711-3e3e60956291
# ╟─de459fab-2f3c-4c71-b830-536b3339ca40
# ╠═8730a999-4b02-4eb3-9e3e-878a05dabecc
# ╟─92d0c429-a893-4688-a3d5-0f4c2be188ab
# ╠═d0c8e5b5-9a2f-4684-8061-02f9ca86b8e7
# ╟─60711b0e-a34c-40bb-8031-6f06628ac011
# ╟─f2c131f5-f56d-4a08-b4a0-fc02dc150b94
# ╠═193a7467-1002-431c-9d8c-f994e47eb635
# ╟─cadf672f-d932-4607-96c5-942add9ae86d
# ╠═e7263d34-4953-47a7-a0b6-19ad6000ed6d
# ╟─6c2018e1-2a9f-405e-914f-f74ff3a64ab4
# ╠═43452f4a-6e26-4c5f-b1c8-1b806a830b25
# ╟─b815dc98-485c-4f00-96f6-a3675632b9d0
# ╟─8d6ddc95-d241-4e08-8d1c-d4ba38abff8b
# ╠═e46c11eb-8102-4f7b-9e2c-d6895b9b4113
# ╠═05434146-9f04-45d5-b4fd-cfb82d7f0312
# ╠═ca8004e3-d8c7-45cc-bb03-e39c7c9000f1
# ╟─a8eb7ae6-a5e2-4399-8ecd-56fe653fa4b5
# ╟─4de274b1-2a31-43aa-99ed-2002850c301d
# ╟─1c05f7be-5b88-4488-aa75-677598f81aee
# ╟─513363e4-8f84-475a-b7f2-77494af26bf8
# ╟─04238620-69cf-42ab-8965-4f286b0d1c45
# ╟─2be101e6-a0d7-41f5-b29e-fcc2398aa0b0
# ╟─1500e1dd-b5b9-4adb-9c2d-3f5c9d0754e2
# ╟─d857d3d3-3403-4df9-a8f8-7fb943d1dcfa
# ╟─a8cdcca5-b5b8-4b69-8c7a-4e9542fbd1b8
# ╟─6c62cf45-7165-48ab-a483-bb14a1c025e3
# ╟─3ae0e8f2-3b0e-4e6a-9d8e-3f0a5b9c6d21
# ╟─6c2b1e9f-9a4a-4e6a-8e2f-1c7d9a0b3e58
# ╟─7f3a2c8e-5b6d-4e91-8a0c-2d9e4b7f1a63
# ╟─71ca1911-4185-4c84-8fd9-f22a26af44f1
