"""
Acá se guardan métodos para generar observaciones a partir del modelo de la trampa
"""

## Funciones pre-procesadas para hacer simulaciones del sistema, teniendo como default parámetros que encontré que funcionaban bien para las simulaciones

"""
    trap_sim(ode_xo, sde_xo, tspan, feedback=(0.0,0.0), ode_abstol=1e-14, ode_reltol=1e-6, sde_abstol=1e-7, sde_reltol=1e-5)

Función que realiza una simulación de la dinámica de la trampa en el intervalo de tiempo 'tspan'.
El sistema parte del estado inicial 'sde_xo' para los primeros momentos: [xo,yo,pxo,pyo],
donde estos son los valores medios de la posición y el momento en las dos dimensiones simuladas.
El sistema parte del estado inicial 'ode_xo' para los segundos momentos: [C_xx,C_xy,C_yy,C_xpx,C_xpy,C_ypx,C_ypy,C_pxpx,C_pxpy,C_pypy]
donde C_ij es <(i-<i>)(j-<j>)>, remplazando en i y j las componentes de la posición o momento de la partícula en la trampa.

El par de funciones 'feedback' da la posibilidad de simular una respuesta agregada al sistema

Los parámetros restantes controlan la tolerancia absoluta y relativa de los solvers numéricos de la ode (para los segundos momentos)
o de la SDE (para los primeros momentos), y tienen como parámetros por default algunos que encontré que funcionaban bien a mano
para el caso que estoy simulando.

La función devuelve las soluciones de ambos sistemas, primero los primeros momentos en 'sol_sde', y luego los segundos en 'sol_ode'.
"""
function trap_sim(tspan, sde_xo=nothing, ode_xo=nothing, feedback_params=(0.0,0.0,0.0,0.0,0.0,0.0); ode_abstol=1e-14, ode_reltol=1e-6, sde_abstol=1e-7, sde_reltol=1e-5, saveat=[], save_idxs=[1,2])
    
    if isnothing(sde_xo) | isnothing(ode_xo)
        ode_xo = thermal_values(n₀)
        sde_xo = [randn(4).*sqrt.(thermal_values(n₀)[[1,3,8,10]])...,0.0,0.0]
    end
    
    # Primero simulo la ODE 
    prob_ode = ODEProblem(ode_drift!, ode_xo, tspan)
    sol_ode = solve(prob_ode, abstol=ode_abstol, reltol=ode_reltol)

    # Luego, basado en los resultados de la ode, preparo los parámetros de la SDE
    s = [t -> sol_ode(t)[i] for i in eachindex(ode_xo[1:7])]
    p = (feedback_params..., s...)

    # Luego simulo la SDE
    prob_sde = SDEProblem(sde_drift!, sde_diffusion!, sde_xo, tspan, p,
                           noise_rate_prototype = zeros(6, 3))
    sol_sde = solve(prob_sde, reltol=sde_abstol, abstol=sde_reltol, alg_hints=[:additive], saveat=saveat, save_idxs=save_idxs)

    return sol_sde, sol_ode
end

"""
    ensemble_trap_sim(tspan, sde_xo=nothing, ode_xo=nothing, feedback_params=(0.0,0.0,0.0,0.0,0.0,0.0), ode_abstol=1e-14, ode_reltol=1e-6, sde_abstol=1e-7, sde_reltol=1e-5)


    
"""
function ensemble_trap_sim(trajectories, tspan, sde_xo=nothing, ode_xo=nothing, feedback_params=(0.0,0.0,0.0,0.0,0.0,0.0); ode_abstol=1e-14, ode_reltol=1e-6, sde_abstol=1e-7, sde_reltol=1e-5, saveat=[], save_idxs=[1,2])
    
    prob_func = (prob,i,repeat)->prob
    if isnothing(sde_xo) | isnothing(ode_xo)
        ode_xo = thermal_values(n₀)
        sde_xo = zeros(6)
        prob_func = (prob,i,repeat) -> begin
            prob.u0 .= (randn(4).*sqrt.(thermal_values(n₀)[[1,3,8,10]]))...,0.0,0.0
            prob
        end
    end
    
    # Primero simulo la ODE 
    prob_ode = ODEProblem(ode_drift!, ode_xo, tspan)
    sol_ode = solve(prob_ode, abstol=ode_abstol, reltol=ode_reltol)

    # Luego, basado en los resultados de la ode, preparo los parámetros de la SDE
    s = [t -> sol_ode(t)[i] for i in eachindex(ode_xo[1:7])]
    p = (feedback_params..., s...)

    # Luego simulo la SDE
    prob_sde = SDEProblem(sde_drift!, sde_diffusion!, sde_xo, tspan, p,
                           noise_rate_prototype = zeros(6, 3))
    
    ensemble_prob_sde = EnsembleProblem(prob_sde, prob_func=prob_func, output_func = (sol, i) -> (sol[:, save_idxs], false))

    sol_sde = solve(ensemble_prob_sde, trajectories=trajectories, reltol=sde_abstol, abstol=sde_reltol, alg_hints=[:additive], saveat=saveat)

    return sol_sde, sol_ode
end

function gen_observations(sol_sde, dt, measure_err)
    to=sol_sde.t[1]
    tf=sol_sde.t[end]

    pos = sol_sde(to:dt:tf)[1:2,:]

    pos + measure_err.*randn(size(pos))
end
