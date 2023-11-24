# Probabilistic Exponential Integrators - Experiments

<p align="center">
<img alt="Figure 2" src="./figures/1_stability.svg" width="100%">
</p>

This repo contains the experiment code for the paper
["Probabilistic Exponential Integrators"](https://openreview.net/forum?id=2dx5MNs2Ip), accepted at NeurIPS 2023.

---

__The functionality of the paper is implemented in the [ProbNumDiffEq.jl](https://nathanaelbosch.github.io/ProbNumDiffEq.jl) package.__
It provides a range of fast probabilistic ODE solvers for first- and second-order ODEs and DAEs - and also probabilistic exponential integrators.
Check out the [getting started](https://nathanaelbosch.github.io/ProbNumDiffEq.jl/stable/getting_started/) and the [probabilistic exponential integrators](https://nathanaelbosch.github.io/ProbNumDiffEq.jl/stable/tutorials/exponential_integrators/) tutorials!

---

## Running the experiments
The experiments are located in `./experiments/`.
Code should be run from the root directory directly.

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

## Reference
```
@inproceedings{bosch2023probabilistic,
  title={Probabilistic Exponential Integrators},
  author={Nathanael Bosch and Philipp Hennig and Filip Tronarp},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023},
  url={https://openreview.net/forum?id=2dx5MNs2Ip}
}
```
