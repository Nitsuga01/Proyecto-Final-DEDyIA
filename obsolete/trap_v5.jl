"""
Acá se implementan cosas para simular la trampa con algún mecanismo de feedback agregado con la intención de enfriarla
"""

include("base_trap.jl")
import Random: randn

# Esta función determina, dado el momento estimado del sistema, la fuerza que se aplica a modo de feedback. e da la proporcionalidad entre el momento y la fuerza
function feedback_force(px,py,e)
    return -e*px, -e*py
end

"""
    measure_position(sol_sde, sol_ode)

Esta función se utiliza para generar una medición de la posición final a partir de 
las soluciones de la SDE y ODE. La SDE determina el valor medio de la medición
mientras la ODE, junto a 'err', el error de la medición, van a determinar la varianza de la misma
"""
function measure_position(mu_pos, C_pos, err)
    # error final en la medición, se agrega el error del sistma de medición.
    pos_cvar = C_pos .+ [err, 0, err] 

    # descomposición de cholesky de la matriz de covarianza de x e y
    ch_mat = [sqrt(pos_cvar[1]) 0; pos_cvar[2]/sqrt(pos_cvar[1]) sqrt(pos_cvar[3]-pos_cvar[2]^2/pos_cvar[1])]

    # vector normal aleatorio
    r = randn(2)

    # vector con distribución gaussiana de valor medio mu_pos y varianza dada por pos_std
    pos_measure = mu_pos + ch_mat*r
end

"""
    state_estimate(position_measure, error_estimate, old_state_estimate, old_state_error)

Esta función indica como se actualiza el estado tiempo a tiempo. Eventualmente estaría bueno cambiarla por algo mejor
para lo que habría que buscar sobre kalman filters, hidden markov models, y otras cosas que también están citadas y muy relacionadas
al problema de inferencia en SDEs. Por ahora, dejo lo más simple y casi peor que podés hacer.
"""
function state_estimate(position_measure, old_state_estimate, t, T, e, weight)

    # Lo que se obtendría propagando el estado mediante la parte ODE de la SDE en un paso de euler de tamaño T  
    propag_state = propag_euler(old_state_estimate, t, T, e)

    # Lo que se obtiene haciendo una resta naíf de las cantidades 
    measured_moment = (position_measure-old_state_estimate[1:2])*m/T

    new_state_estimate = weight*[position_measure..., measured_moment...] .+ (1-weight)*propag_state
end

"""
Esta función propaga un estimativo del estado y su error usando el método de euler explicito en un paso de tamaño T
t es el tiempo de en que se realiza el paso y e es la proporcionalidad de la fuerza de feedback
"""
function propag_euler(state_estimate, t, T, e)
    km = I_G - 2 * sqrt(I_L*I_G) * cos(2 * Ω * t)
    kp = I_G + 2 * sqrt(I_L*I_G) * cos(2 * Ω * t)
    kxy = 4 * sqrt(I_L*I_G)*sin(2 * Ω * t)
    c = 2 * V_0 / w₀^2
    fxx, fxy, fyy = c * -2 * km, c * kxy, c * -2 * kp
    
    A=[1     0     -T/m  0
       0     1     0     -T/m
       T*fxx T*fxy 1-e*T 0
       T*fxy T*fyy 0     1-e*T]
    
    propag = A*state_estimate
    #propag_euler_error = A*old_state_error*(A')
    
    return propag
end

"""
Esta función simula una trampa con feedback 
"""
function trap_with_filter(tspan, p, sde_xo=nothing, ode_xo=nothing, ini_state_estimate=[0.0,0.0,0.0,0.0])
    pos_series = Vector{Float64}[]
    T, err, e, weight = p

    t=tspan[1]

    if isnothing(sde_xo) | isnothing(ode_xo)
        ode_xo = thermal_values(n₀)
        sde_xo = randn(4).*sqrt.(thermal_values(n₀)[[1,3,8,10]])
    end
    
    sol_sde, sol_ode = trap_sim(sde_xo, ode_xo, (t,t+T)) 
    state_sde=sol_sde(t+T)
    state_ode=sol_ode(t+T)
    pos = measure_position(state_sde[1:2], state_ode[1:3], err)
    state = state_estimate(pos,ini_state_estimate, t, T, e, weight)

    push!(pos_series,pos)
    while t<tspan[2] 
        t+=T

        sol_sde, sol_ode = trap_sim(state_sde, state_ode, (t,t+T), feedback_force(state[3],state[4],e))
        state_sde=sol_sde(t+T)
        state_ode=sol_ode(t+T)
    
        pos = measure_position(state_sde[1:2], state_ode[1:3], err)

        state = state_estimate(pos, state, t, T, e, weight)

        push!(pos_series,pos)
    end

    return pos_series, state_sde, state_ode
end