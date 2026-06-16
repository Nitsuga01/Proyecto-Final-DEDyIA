"""
En este archivo se hace un agregado de todos los métodos utilizados para simular, inferir
y cualquier otra cosa sobre la trampa. 
"""

## Imports y Usings

using DifferentialEquations
using StochasticDiffEq
using Turing
using Gen

## Includes con métodos

include("trap_v3.jl")
include("data_generation.jl")
include("parameter_estimation.jl")
