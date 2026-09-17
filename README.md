# Introduction to Decision-Focused Learning for airlines

## Program

The day is split in two parts and six steps.

**Part 1: Operations Research**

1. **The practical problem.** Introduction to the stochastic tail assignment problem with delay costs.
2. **Classical OR: column generation.** How to model delays, and how to solve the problem at scale with column generation, where the pricing subproblem is a stochastic shortest path.
3. **Lab 1: OR.** Hands-on Pluto notebook where you build and run a column generation solver on a toy instance and visualize the resulting routes.

**Part 2: Decision-Focused Learning**

4. **Intro to Decision-Focused Learning.** Why and how to train a machine learning model whose output feeds a combinatorial optimization layer, and the losses and gradients that make this work.
5. **Demo.** Live demonstration of how to build a decision-focused learning pipeline from scratch, using the Julia ecosystem.
6. **Lab 2: Decision-Focused Learning.** Hands-on Pluto notebook where you train a DFL policy for the stochastic tail assignment problem.

## Before the session: install Julia 1.12 and Pluto

The practice sessions will run in [Pluto](https://plutojl.org/) notebooks on Julia.

### 1. Install Julia 1.12 with juliaup

Use `juliaup`, the official Julia version manager.

On Linux and macOS, run in a terminal:

```bash
curl -fsSL https://install.julialang.org | sh
```

On Windows, run in a terminal:

```powershell
winget install --name Julia --id 9NJNWW8PVKMN -e -s msstore
```

Alternatively, download an installer for your platform from <https://julialang.org/downloads/>.

Then open a **new** terminal and make sure version 1.12 is installed and used by default:

```bash
juliaup add 1.12
juliaup default 1.12
julia --version
```

### 2. Install Pluto

In a terminal, run:

```bash
julia -e 'using Pkg; Pkg.add("Pluto")'
```

### 3. Check that Pluto starts

In a terminal, run:

```bash
julia -e 'using Pluto; Pluto.run()'
```

A browser tab with the Pluto welcome page should open (if it does not, copy the URL printed in the terminal into your browser).

## Content of this repository

- `01_or_for_stochastic_tail_assignment.pdf`: slides for part 1 of the workshop.
- `02_or_notebook.jl`: Pluto notebook for part 1 of the workshop.
- `03_decision_focused_learning.pdf`: slides for part 2 of the workshop.
- `04_dfl_demo_knapsack.jl`: Pluto notebook for the live demo in part 2 of the workshop.
- `05_decision_focused_learning_notebook.jl`: Pluto notebook for part 2 of the workshop.
