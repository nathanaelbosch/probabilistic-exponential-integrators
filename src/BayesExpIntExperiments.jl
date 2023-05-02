module BayesExpIntExperiments

using LinearAlgebra
using SparseArrays
using SimpleUnPack
using ForwardDiff
using SciMLBase
using ProbNumDiffEq
using Statistics
using DiffEqDevTools

include("reaction_diffusion.jl")
include("workprecision.jl")

end
