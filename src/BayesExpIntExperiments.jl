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
using ProgressMeter

include("problems.jl")
include("workprecision.jl")
include("plotstuff.jl")

end
