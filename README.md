# Probabilistic Exponential Integrators - Experiments

<p align="center">
<img alt="Figure 2" src="./figures/1_stability.svg" width="100%">
</p>

This repo contains the experiment code for the paper
"Probabilistic Exponential Integrators" ([arxiv link](https://arxiv.org)).

---

__The functionality of the paper will be made available as part of the [ProbNumDiffEq.jl](https://nathanaelbosch.github.io/ProbNumDiffEq.jl) package.__
It provides contains _fast_ ODE filters for first- and second-order ODEs, and even DAEs.

---

## Running the experiments
The experiments are located in `./experiments/`;
each subfolder has an individual README that explains the individual files.
Since the scripts use the core located in `./src/`, code should be run from the root directory directly.

First open `julia`, activate the local environment, and instantiate it to install all the packages:
```
julia> ]
(v1.9) pkg> activate .
(v1.9) pkg> instantiate
```
and you can quit the `pkg` environment by hitting backspace.

To run a julia script from the Julia REPL,
```
julia> include("experiments/1_logistic/run.jl")
```
