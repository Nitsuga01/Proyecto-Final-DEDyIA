"""
En este archivo se hace un agregado de todos los métodos utilizados para simular, inferir
y cualquier otra cosa sobre la trampa. 
"""
## Imports y Usings

import LinearAlgebra: LinearAlgebra, diagm, I, mul!, tr
import Distributions: Distributions, Dirichlet, MvNormal
import DifferentialEquations: DifferentialEquations, init, ODEProblem, solve
import Gen: Gen, choicemap, @gen, get_retval, log_ml_estimate, mvnormal, NoChange, sample_unweighted_traces, select, simulate
import GenParticleFilters: GenParticleFilters, effective_sample_size, pf_initialize, pf_rejuvenate!, pf_resample!, pf_update!
import LowLevelParticleFilters: LowLevelParticleFilters, covariance, forward_trajectory, KalmanFilter, predict!, simulate
import Optimization: Optimization, AutoForwardDiff, OptimizationFunction, OptimizationProblem, solve
import Statistics: mean, std

using OptimizationOptimJL, OptimizationNLopt
using CairoMakie

## Constantes y parámetros de la trampa, la partícula, el feedback, la naturaleza y lo que sea obtenidos del paper

# voy a trabajar en ev, nm, us. pongo acá cosas que de seguro no voy a variar y son constantes de todas las simulaciones.

const ϵ₀ = 0.05526345406        # vacuum permittivity in e^2 / eV / nm
const ħ = 1.054571817 * 1e-34   / 1.602176634e-19 * 1e6 # reduced Planck constant in ev*us
const c = 299792458             * 1e9 / 1e6 # speed of light in nm/us

const λ = 1.55 * 1e-6   * 1e9  # laser wavelength in nm
const k = 2 * π / λ            # laser wavenumber in nm^-1

const NA = 0.6 # numerical aperture of the lens, which determines the waist of the beam
const w₀ = λ / π / NA # Laguerre beam waist in nm

const R = 50 * 1e-9         * 1e9 # particle radius in nm
const m = 1.14 * 1e-18      / 1.602176634e-19 / (1e9/1e6)^2 # mass of the particle in ev (us/nm)^2. Here we assume a silica particle.
const n = 1.45 # refractive index of the particle
const α = 4 * π * ϵ₀ * R^3 * ((n^2 - 1) / (n^2 + 2)) # polarizability of the particle in e^2 / eV *nm^2
const Rα = real(α) # real part of the polarizability of the particle

const P = 70 * 1e-3         / 1.602176634e-19 / 1e6 # total laser power in ev/us
const V_0 = Rα * P / (c * π * w₀^2 * ϵ₀) # potential depth in ev.
const I_G = 0.3 # fractional intensity of gaussian beam, which determines the modulation depth of the potential. I_G + I_L = 1.
const I_L = 1.0 - I_G # fractional intensity of the Laguerre-Gaussian beam. I_G + I_L = 1.

const Ωx = 150 * 1e3 * 2 * π    / 1e6 # gaussian trap frequency equivalent in x in rad/us.
const Ωy = 150 * 1e3 * 2 * π    / 1e6 # gaussian trap frequency equivalent in y in rad/us.
const x_zpf = sqrt(ħ / (2 * m * Ωx)) # zero-point fluctuation of the position in x in nm for gaussian trap.
const y_zpf = sqrt(ħ / (2 * m * Ωy)) # zero-point fluctuation of the position in y in nm for gaussian trap.

const Cx = 1/5 # geometric factor for photon recoil damping due to anisotropy of scattering in x TODO No entiendo muy bien por que se da esta cosa.
const Cy = 2/5 # geometric factor for photon recoil damping due to anisotropy of scattering in y TODO No entiendo muy bien por que se da esta cosa.
const Γx = Cx * I_G * P * k^5 * abs2(α) / (15 * c * m * Ωx * w₀^2 * π^2 * ϵ₀^2) # photon recoil damping rate in x.
const Γy = Cy * I_G * P * k^5 * abs2(α) / (15 * c * m * Ωy * w₀^2 * π^2 * ϵ₀^2) # photon recoil damping rate in y.
const Λx = Γx / x_zpf^2 # effective damping rate for x in lindblad equation
const Λy = Γy / y_zpf^2 # effective damping rate for yin lindblad equation

const ηx = 1.0 # quantum efficiency of the measurements in x, which determines the strength of the measurement backaction. Here we assume perfect measurements.
const ηy = 1.0 # quantum efficiency of the measurements in y, which determines the strength of the measurement backaction. Here we assume perfect measurements.

const n₀ = 0.8 # initial ocupation of CoM mode in x and y
const dlim = w₀ / sqrt(1 + 2 * sqrt(I_L / I_G)) # limit of the harmonic approximation
const Ω₀ = sqrt(2 * V_0 / w₀^2 / m) * sqrt(4 * sqrt(I_G * I_L) + 2 * I_G)
const Ω = 5*Ω₀ # rotational frequency of trap (and thus the trap) in rad/us

## Includes con métodos

include("trap_v3.jl")
include("gen_solvers.jl")
include("kalman_solvers.jl")
