"""
    SDEObs(tpoints, sde_xo, ode_xo, feedback_params=(0.0,0.0,0.0,0.0,0.0,0.0,0.0); kwargs=(;))

Implementación en Gen de solver numérico / generador de observaciones. El solver numérico toma los parámetros del feedback y medición del sistema
junto a un estado inicial para los primeros y segundos momentos y un conjunto de puntos en el tiempo en que se quieren datos.

En base a eso usa un método partido para generar datos observacionales de acuerdo a la SDE según un Euler-Mauryama modificado.
- Se usan solvers numéricos de DifferentialEquations para simular la parte determinista de la evolución del sistma,
  así aprovechando todas sus ventajas.
- Aprovechando que el ruido es pequeño y aditivo, se agrega ruido estocástico al sistema al estilo de Euler-Mauryama.
  Eventualmente, esto se podría mejorar para conseguir algo tipo Milstein, pero ojalá que no tenga que hacerlo pq no es diagonal el ruido.

El sistema se simula y se retornan datos observacionales (agregando error de medición) sobre los puntos temporales especificados.

- Como algunas cuestiones, creo que esto va a funcionar bien solo si los pasos temporales son todos bastante pequeños e idealmente equiespaciados.
- nsteps es el número de pasos del solver numérico entre cada punto de tpoints.
- ode_abstol y ode_reltol controlan la precisión de los solvers de DifferentialEquations
"""
@gen function SDEObs(tpoints, feedback_params=SA[0.0,0.0,0.0,0.0,0.0,0.0,0.0], kwargs=(;))
    kwarg_keys=SA[:sde_xo, :ode_xo, :nssteps, :ode_abstol, :ode_reltol]
    kwarg_vals = Dict()
    for k in kwarg_keys
        if k in keys(kwargs)
            kwarg_vals[k]=kwargs[k]
        else
            kwarg_vals[k]=default_kwarg_value(k,feedback_params)
        end
    end
    sde_xo, ode_xo, nssteps, ode_abstol, ode_reltol = getindex.(Ref(kwarg_vals),kwarg_keys)
    σ = feedback_params[7]

    state_vec=Matrix{Float64}(undef, length(tpoints), 6)
    obs_vec=Matrix{Float64}(undef, length(tpoints), 2)

    state_sde = sde_xo
    state_ode = ode_xo

    prob_ode = ODEProblem(ode_drift!, state_ode, (tpoints[1],tpoints[end]))
    integrator_ode = init(prob_ode, abstol=ode_abstol, reltol=ode_reltol, save_everystep=false)
    prob_splitsde = ODEProblem(sde_drift!, state_sde, (tpoints[1],tpoints[end]),feedback_params)
    integrator_sde = init(prob_splitsde, abstol=ode_abstol, reltol=ode_reltol, save_everystep=false)
    sde_noise_matrix = zeros(length(sde_xo),3)
    sde_scaled_vector = zeros(6)

    y_obs = {1 => :y_obs} ~ mvnormal(state_sde[1:2], σ^2*I(2))
    state_vec[1,:] .= state_sde
    obs_vec[1,:] .= y_obs

    for i in 2:length(tpoints)

        dt=(tpoints[i]-tpoints[i-1])/nssteps

        for j in 1:nssteps

            # Simulo la parte aleatoria
            W = {i => Symbol(:SDERandom,j)} ~ mvnormal(zeros(3),I(3))
            sde_diffusion!(sde_noise_matrix,feedback_params,state_ode[1:7])
            dur = sqrt(dt)*mul!(sde_scaled_vector,sde_noise_matrix,W)

            # luego simulo la parte determinista de la dinámica
            step!(integrator_sde, dt, true)
            step!(integrator_ode, dt, true)

            # ahora agrego el ruido
            set_u!(integrator_sde, integrator_sde.u .+ dur)
        end

        state_sde=integrator_sde.u
        y_obs = {i => :y_obs} ~ mvnormal(state_sde[1:2], σ^2*I(2))
        state_vec[i,:] .= state_sde
        obs_vec[i,:] .= y_obs
    end

    return state_vec, obs_vec
end

# función para obtener valores por default de los kwargs de SDEObs, si no fueron dados.
@gen function default_kwarg_value(k::Symbol,params)
    if k == :sde_xo
        ux,uy,vx,vy,errx,erry,σ = params
        s = thermal_values(n₀)
        covmat = diagm(s[[1,3,8,10,8,10]])
        covmat[3,5] = covmat[5,3] = covmat[3,3]
        covmat[4,6] = covmat[6,4] = covmat[4,4]
        covmat[6,6] += errx^2/2/vx
        covmat[5,5] += erry^2/2/vy
        sde_xo = {:sde_xo} ~ mvnormal(zeros(6), covmat)
        return sde_xo
    end
    if k == :ode_xo
        return thermal_values(n₀)
    end
    if k == :nssteps
        return 1
    end
    if k == :ode_abstol
        return 1e-10
    end
    if k == :ode_reltol
        return 1e-6
    end
end