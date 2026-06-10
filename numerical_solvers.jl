import DifferentialEquations as DE
import DifferentialEquations: ODEProblem, Tsit5, SDEProblem, SOSRA, solve

const ϵ₀ = 8.854187818 * 1e-12 # vacuum permittivity in F/m
const ħ = 1.054571817 * 1e-34 # reduced Planck constant in J*s
const c = 299792458 # speed of light in m/s

const λ = 1.55 * 1e-6 # laser wavelength in m
const k = 2 * π / λ # laser wavenumber in m^-1
const Ω = 2 * π * c / λ # angular frequency of the laser (and thus the trap) in rad/s

const NA = 0.6 # numerical aperture of the lens, which determines the waist of the beam
const w₀ = λ / π / NA # Laguerre beam waist in m

const R = 50 * 1e-9 # particle radius in m
const m = 1.14 * 1e-18 # mass of the particle in kg. Here we assume a silica particle.
const n = 1.45 # refractive index of the particle
const α = 4 * π * ϵ₀ * R^3 * ((n^2 - 1) / (n^2 + 2)) # polarizability of the particle in F*m^2
const Rα = real(α) # real part of the polarizability of the particle

const P = 70 * 1e-3 # total laser power in W
const V_0 = Rα * P / (c * π * w₀^2 * ϵ₀) # potential depth in J.
const I_G = 0.3 # fractional intensity of gaussian beam, which determines the modulation depth of the potential. I_G + I_L = 1.
const I_L = 1.0 - I_G # fractional intensity of the Laguerre-Gaussian beam. I_G + I_L = 1.

const Ωx = 150 * 1e3 * 2 * π # gaussian trap frequency equivalent in x in rad/s.    TODO fijarse de donde salen estas cosas. 
const Ωy = 150 * 1e3 * 2 * π # gaussian trap frequency equivalent in y in rad/s.    TODO fijarse de donde salen estas cosas.
const x_zpf = sqrt(ħ / (2 * m * Ωx)) # zero-point fluctuation of the position in x for gaussian trap.
const y_zpf = sqrt(ħ / (2 * m * Ωy)) # zero-point fluctuation of the position in y for gaussian trap.

const Cx = 1/5 # geometric factor for photon recoil damping due to anisotropy of scattering in x TODO fijarse de donde salen estas cosas
const Cy = 2/5 # geometric factor for photon recoil damping due to anisotropy of scattering in y TODO fijarse de donde salen estas cosas
const Γx = Cx * I_G * P * k^4 * abs2(α) / (6 * m * Ωx^2 * w₀^2 * π^2 * ϵ₀) # photon recoil damping rate in x.
const Γy = Cy * I_G * P * k^4 * abs2(α) / (6 * m * Ωy^2 * w₀^2 * π^2 * ϵ₀) # photon recoil damping rate in y.
const Λx = Γx / x_zpf^2 # effective damping rate for x in lindblad equation
const Λy = Γy / y_zpf^2 # effective damping rate for x in lindblad equation

const ηx = 1.0 # quantum efficiency of the measurements in x, which determines the strength of the measurement backaction. Here we assume perfect measurements.
const ηy = 1.0 # quantum efficiency of the measurements in y, which determines the strength of the measurement backaction. Here we assume perfect measurements.

const n₀ = 0.8 # initial ocupation of CoM mode in x and y
const dlim = w₀ / sqrt(1 + 2 * sqrt(I_L / I_G)) # limit of the harmonic approximation

#TODO agregar Ω₀ 

function potential(t,x,y)
    km = I_G - 2 * sqrt(I_L * I_G) * cos(2 * Ω * t)
    kp = I_G + 2 * sqrt(I_L * I_G) * cos(2 * Ω * t)
    kxy = 4 * sqrt(I_L * I_G) * sin(2 * Ω * t)
    return V_0 * (km * x^2 + kp * y^2 - kxy * x * y)
end

function force(t,x,y)
    km = I_G - 2 * sqrt(I_L*I_G) * cos(2 * Ω * t)
    kp = I_G + 2 * sqrt(I_L*I_G) * cos(2 * Ω * t)
    kxy = 4 * sqrt(I_L*I_G)*sin(2 * Ω * t)
    return V_0 * ( -2 * km * x + kxy * y), V_0 * ( -2 * kp * y + kxy * x)
end

function sde_drift!(du,u,p,t)
    x, y, px, py = u
    ux, uy, C_xx, C_yy, C_xy, C_xpx, C_xpy, C_ypx, C_ypy = p

    fx, fy = force(t,x,y) # obs que como la fuerza es lineal, la puedo evaluar directo en los vals medios.

    du[1] = px/m
    du[2] = py/m
    du[3] = fx-ux
    du[4] = fy-uy
end

function sde_diffusion!(du,u,p,t)
    ux, uy, C_xx, C_yy, C_xy, C_xpx, C_xpy, C_ypx, C_ypy = p

    ax = sqrt(2 * Λx * ηx)
    ay = sqrt(2 * Λy * ηy)

    du[1,1] = ax * C_xx
    du[2,1] = ax * C_xy
    du[3,1] = ax * C_xpx
    du[4,1] = ax * C_xpy
    du[1,2] = ay * C_xy
    du[2,2] = ay * C_yy
    du[3,2] = ay * C_ypx
    du[4,2] = ay * C_ypy
end

function ode_drift!(du,u,p,t)
    C_xx, C_xy, C_yy, C_xpx, C_xpy, C_ypx, C_ypy, C_pxpx, C_pxpy, C_pypy = u

    ax = 2 * Λx * ηx
    ay = 2 * Λy * ηy
    bx = 2 * Λx * ħ^2
    by = 2 * Λy * ħ^2

    km = I_G - 2 * sqrt(I_L * I_G) * cos(2 * Ω * t)
    kp = I_G + 2 * sqrt(I_L * I_G) * cos(2 * Ω * t)
    kxy = 4 * sqrt(I_L * I_G) * sin(2 * Ω * t)

    C_dxVx = V_0 * (2 * km * C_xx - kxy * C_xy)
    C_dxVy = V_0 * (2 * km * C_xy - kxy * C_yy)
    C_dyVx = V_0 * (2 * kp * C_xy - kxy * C_xx)
    C_dyVy = V_0 * (2 * kp * C_yy - kxy * C_xy)
    C_dxVpx = V_0 * (2 * km * C_xpx - kxy * C_ypx)
    C_dxVpy = V_0 * (2 * km * C_xpy - kxy * C_ypy)
    C_dyVpx = V_0 * (2 * kp * C_ypx - kxy * C_xpx)
    C_dyVpy = V_0 * (2 * kp * C_ypy - kxy * C_xpy)

    # du :  L_Liouville          + L_Recoil +  L_Diffusion
    du[1] = 2/m * C_xpx                      - ax * C_xx^2 - ay * C_xy^2                # d/dt C_xx
    du[2] = 1/m * (C_xpy + C_ypx)            - ax * C_xx * C_xy + ay * C_yy * C_xy      # d/dt C_xy
    du[3] = 2/m * C_ypy                      - ax * C_xy^2 - ay * C_yy^2                # d/dt C_yy
    du[4] = 1/m * C_pxpx - C_dxVx            - ax * C_xx * C_xpx - ay * C_xy * C_ypx    # d/dt C_xpx
    du[5] = 1/m * C_pxpy - C_dyVx            - ax * C_xx * C_xpy - ay * C_xy * C_ypy    # d/dt C_xpy
    du[6] = 1/m * C_pxpy - C_dxVy            - ax * C_xy * C_xpx - ay * C_yy * C_ypx    # d/dt C_ypx
    du[7] = 1/m * C_pypy - C_dyVy            - ax * C_xy * C_xpy - ay * C_yy * C_ypy    # d/dt C_ypy
    du[8] = - 2 * C_dxVpx        + bx        - ax * C_xpx^2 - ay * C_ypx^2              # d/dt C_pxpx
    du[9] = - (C_dxVpy + C_dyVpx)            - ax * C_xpx * C_xpy - ay * C_ypx * C_ypy  # d/dt C_pxpy
    du[10] = - 2 * C_dyVpy       + by        - ax * C_xpy^2 + ay * C_ypy^2              # d/dt C_pypy
    
end

function thermal_values(n) #TODO chequear
    x = 0
    p = 0
    C_xx = x_zpf^2 * (2 * n + 1)
    C_yy = y_zpf^2 * (2 * n + 1)
    C_xy = 0
    C_xpx = 0
    C_xpy = 0
    C_ypx = 0
    C_ypy = 0
    C_pxpx = ħ^2 / (4 * x_zpf^2) * (2 * n + 1)
    C_pxpy = 0
    C_pypy = ħ^2 / (4 * y_zpf^2) * (2 * n + 1)
    return x, p, [C_xx, C_xy, C_yy, C_xpx, C_xpy, C_ypx, C_ypy, C_pxpx, C_pxpy, C_pypy]
end


x0, p0, initial_state = thermal_values(n₀)

tspan = (0.0,1e-3)
prob_ode = ODEProblem(ode_drift!, initial_state, tspan)
sol_ode = solve(prob_ode, Tsit5(), saveat=1e-6)

prob_sde = SDEProblem(sde_drift!, sde_diffusion!, initial_state, tspan)
sol_sde = solve(prob_sde, SOSRA(), saveat=1e-6)