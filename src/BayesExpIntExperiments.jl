module BayesExpIntExperiments

using LinearAlgebra
using SparseArrays
using SimpleUnPack
using ForwardDiff
using SciMLBase
using ProbNumDiffEq
using Statistics
using DiffEqDevTools
using CairoMakie

include("reaction_diffusion.jl")
include("workprecision.jl")
include("plotstuff.jl")

end
