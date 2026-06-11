import DifferentialEquations: ODEProblem, solve
import StochasticDiffEq: SDEFunction, SDEProblem, SOSRA, SKenCarp, solve
using CairoMakie

# voy a trabajar en ev, nm, us

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

const Ωx = 150 * 1e3 * 2 * π    / 1e6 # gaussian trap frequency equivalent in x in rad/us.    TODO fijarse de donde salen estas cosas.
const Ωy = 150 * 1e3 * 2 * π    / 1e6 # gaussian trap frequency equivalent in y in rad/us.    TODO fijarse de donde salen estas cosas.
const x_zpf = sqrt(ħ / (2 * m * Ωx)) # zero-point fluctuation of the position in x in nm for gaussian trap.
const y_zpf = sqrt(ħ / (2 * m * Ωy)) # zero-point fluctuation of the position in y in nm for gaussian trap.

const Cx = 1/5 # geometric factor for photon recoil damping due to anisotropy of scattering in x TODO fijarse de donde salen estas cosas
const Cy = 2/5 # geometric factor for photon recoil damping due to anisotropy of scattering in y TODO fijarse de donde salen estas cosas
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

# OBS, estoy usando el del paper de la trampa óptica en detalle. El que dan en el principal está mal pq falta dividir por w_o ^ 2.
# esa falta de división hace que las unidades no den, y además da un potencial efectivo mucho más chiquito.
# también se comen un 2 multiplicando.
function potential(t,x,y)
    km = I_G - 2 * sqrt(I_L * I_G) * cos(2 * Ω * t)
    kp = I_G + 2 * sqrt(I_L * I_G) * cos(2 * Ω * t)
    kxy = 4 * sqrt(I_L * I_G) * sin(2 * Ω * t)
    c = 2 * V_0 / w₀^2
    return c * (km * x^2 + kp * y^2 - kxy * x * y)
end

function force(t,x,y)
    km = I_G - 2 * sqrt(I_L*I_G) * cos(2 * Ω * t)
    kp = I_G + 2 * sqrt(I_L*I_G) * cos(2 * Ω * t)
    kxy = 4 * sqrt(I_L*I_G)*sin(2 * Ω * t)
    c = 2 * V_0 / w₀^2
    return c * ( -2 * km * x + kxy * y), c * ( -2 * kp * y + kxy * x)
end

function sde_drift!(du,u,p,t)
    x, y, px, py = u
    ux, uy, C_xx, C_yy, C_xy, C_xpx, C_xpy, C_ypx, C_ypy = p

    fx, fy = force(t,x,y) # obs que como la fuerza es lineal, la puedo evaluar directo en los vals medios.

    du[1] = px/m
    du[2] = py/m
    du[3] = fx-ux(t)
    du[4] = fy-uy(t)
end

function sde_diffusion!(du,u,p,t)
    ux, uy, C_xx, C_xy, C_yy, C_xpx, C_xpy, C_ypx, C_ypy = p

    ax = sqrt(2 * Λx * ηx)
    ay = sqrt(2 * Λy * ηy)

    du[1,1] = ax * C_xx(t)
    du[2,1] = ax * C_xy(t)
    du[3,1] = ax * C_xpx(t)
    du[4,1] = ax * C_xpy(t)
    du[1,2] = ay * C_xy(t)
    du[2,2] = ay * C_yy(t)
    du[3,2] = ay * C_ypx(t)
    du[4,2] = ay * C_ypy(t)
end

function ode_drift!(du,u,p,t)
    C_xx, C_xy, C_yy, C_xpx, C_xpy, C_ypx, C_ypy, C_pxpx, C_pxpy, C_pypy = u

    ax = 2 * Λx * ηx
    ay = 2 * Λy * ηy
    bx = 2 * Λx * ħ^2
    by = 2 * Λy * ħ^2
    c = 2 * V_0 / w₀^2

    km = I_G - 2 * sqrt(I_L * I_G) * cos(2 * Ω * t)
    kp = I_G + 2 * sqrt(I_L * I_G) * cos(2 * Ω * t)
    kxy = 4 * sqrt(I_L * I_G) * sin(2 * Ω * t)

    C_dxVx = c * (2 * km * C_xx - kxy * C_xy)
    C_dxVy = c * (2 * km * C_xy - kxy * C_yy)
    C_dyVx = c * (2 * kp * C_xy - kxy * C_xx)
    C_dyVy = c * (2 * kp * C_yy - kxy * C_xy)
    C_dxVpx = c * (2 * km * C_xpx - kxy * C_ypx)
    C_dxVpy = c * (2 * km * C_xpy - kxy * C_ypy)
    C_dyVpx = c * (2 * kp * C_ypx - kxy * C_xpx)
    C_dyVpy = c * (2 * kp * C_ypy - kxy * C_xpy)

    # du :  L_Liouville          + L_Recoil +  L_Diffusion
    du[1] = 2/m * C_xpx                      - ax * C_xx^2 - ay * C_xy^2                # d/dt C_xx
    du[2] = 1/m * (C_xpy + C_ypx)            - ax * C_xx * C_xy - ay * C_yy * C_xy      # d/dt C_xy
    du[3] = 2/m * C_ypy                      - ax * C_xy^2 - ay * C_yy^2                # d/dt C_yy
    du[4] = 1/m * C_pxpx - C_dxVx            - ax * C_xx * C_xpx - ay * C_xy * C_ypx    # d/dt C_xpx
    du[5] = 1/m * C_pxpy - C_dyVx            - ax * C_xx * C_xpy - ay * C_xy * C_ypy    # d/dt C_xpy
    du[6] = 1/m * C_pxpy - C_dxVy            - ax * C_xy * C_xpx - ay * C_yy * C_ypx    # d/dt C_ypx
    du[7] = 1/m * C_pypy - C_dyVy            - ax * C_xy * C_xpy - ay * C_yy * C_ypy    # d/dt C_ypy
    du[8] = - 2 * C_dxVpx        + bx        - ax * C_xpx^2 - ay * C_ypx^2              # d/dt C_pxpx
    du[9] = - (C_dxVpy + C_dyVpx)            - ax * C_xpx * C_xpy - ay * C_ypx * C_ypy  # d/dt C_pxpy
    du[10] = - 2 * C_dyVpy       + by        - ax * C_xpy^2 - ay * C_ypy^2              # d/dt C_pypy
    
end

function thermal_values(n) #TODO chequear
    x = 0
    p = 0
    C_xx = x_zpf^2 * (2 * n + 1) # variance of position in x for thermal state with mean occupation n
    C_yy = y_zpf^2 * (2 * n + 1) # variance of position in y for thermal state with mean occupation n
    C_xy = 0
    C_xpx = 0
    C_xpy = 0
    C_ypx = 0
    C_ypy = 0
    C_pxpx = ħ^2 / (4 * x_zpf^2) * (2 * n + 1) # variance of position in px for thermal state with mean occupation n
    C_pxpy = 0
    C_pypy = ħ^2 / (4 * y_zpf^2) * (2 * n + 1) # variance of position in py for thermal state with mean occupation n
    return [C_xx, C_xy, C_yy, C_xpx, C_xpy, C_ypx, C_ypy, C_pxpx, C_pxpy, C_pypy]
end


initial_state_ode = thermal_values(n₀)

tspan = (0.0,500.0)
prob_ode = ODEProblem(ode_drift!, initial_state_ode, tspan)
sol_ode = solve(prob_ode, abstol=1e-10, reltol=1e-6)

fig1=Figure()
ax1 = Axis(fig1[1,1], xlabel = "t (us)", ylabel = "C_xx (nm^2)")
T=tspan[1]:1:tspan[2]
lines!(ax1, T, sol_ode(T)[1,:], label = "C_xx")
fig1

s=[t -> sol_ode(t)[i] for i in eachindex(initial_state_ode[1:7])]
#s=[t -> 0.0 for i in eachindex(initial_state_ode[1:7])]
p = (x->0.0, x->0.0, s...)

tspan = (0.0,500.0)
initial_state_sde = sqrt.(initial_state_ode[[1,3,8,10]])
func_sde = SDEFunction(sde_drift!, sde_diffusion!)
prob_sde = SDEProblem(func_sde, initial_state_sde, tspan, p, noise_rate_prototype = zeros(4, 2))
sol_sde = solve(prob_sde, reltol=1e-7, abstol=1e-9)

fig2=Figure()
ax2 = Axis(fig2[1,1], xlabel = "t (us)", ylabel = "Pos (nm)")
lines!(ax2, sol_sde.t, sol_sde[1,:], label = "x")
lines!(ax2, sol_sde.t, sol_sde[2,:], label = "y")
ax3 = Axis(fig2[1,2], xlabel = "x (nm)", ylabel = "y (nm)")
lines!(ax3, sol_sde[1,:], sol_sde[2,:])
fig2