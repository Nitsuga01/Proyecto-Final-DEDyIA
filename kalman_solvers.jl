## Numerics

function sde_drift_jacobian(u,input,p,i,t0,Δt)
    # tiempo
    t = t0+i*Δt

    # parametros de feedback de la trampa
    ux, uy, vx, vy, errx, erry, σ = p
    
    # parametros de fuerza en trampa
    km = I_G - 2 * sqrt(I_L*I_G) * cos(2 * Ω * t)
    kp = I_G + 2 * sqrt(I_L*I_G) * cos(2 * Ω * t)
    kxy = 4 * sqrt(I_L*I_G)*sin(2 * Ω * t)
    c = 2 * V_0 / w₀^2

    # Jacobiano de la trampa, es todo lineal, así que ganamos.
    J=zeros(typeof(ux),(6,6))
    J[1,3] = 1/m
    J[2,4] = 1/m
    J[3,1] = -2 * c * km
    J[3,2] = c * kxy
    J[3,5] = ux
    J[3,1] = c * kxy
    J[3,1] = -2 * c * kp
    J[4,6] = uy
    J[5,3] = vx
    J[5,5] = -vx
    J[6,4] = vy
    J[6,6] = -vy

    return J
end

function sde_noise_matrix(u,input,p,t,second_moments)
    # fuerza de backaction, medición y segundos momentos de posición y momento en la trampa
    ux, uy, vx, vy, errx, erry, σ = p

    C_xx, C_xy, C_yy, C_xpx, C_xpy, C_ypx, C_ypy = second_moments

    # constantes que gobiernan la magnitud del ruido
    ax = sqrt(2 * Λx * ηx)
    ay = sqrt(2 * Λy * ηy)

    G=[ax * C_xx  ay * C_xy  0      ;
       ax * C_xy  ay * C_yy  0      ;
       ax * C_xpx ay * C_ypx 0      ;
       ax * C_xpy ay * C_ypy 0      ;
       0          0          vx*errx;
       0          0          vy*erry]
    
    Q = 1/2 .* G*G'
    
end

function trap_kalman_filter(d0,to,tf,Δt,feedback_params;ode_xo=thermal_values(n₀),ode_abstol=1e-10,ode_reltol=1e-6)
    prob_ode = ODEProblem(ode_drift!, ode_xo, (to,tf))
    sol_ode = solve(prob_ode, abstol=ode_abstol, reltol=ode_reltol, saveat=to:Δt:tf)
    
    R1 = (u,input,p,i) -> sde_noise_matrix(u,input,p,i,sol_ode[i+1]).*Δt
    R2 = (u,input,p,i) -> I(2)*p[7]^2

    A = (u,input,p,i)->I(6).+sde_drift_jacobian(u,input,p,i,to,Δt).*Δt
    C = (u,input,p,i)->[1 0 0 0 0 0; 0 1 0 0 0 0]

    kf=KalmanFilter(A, zeros(6,0), C, zeros(2,0), R1, R2, d0; p = feedback_params, α = 1.0, check = true, nu=0, nx=6, ny=2, Ts=1)
end

## Inference

function loglik_dataset(dataset,filter)
    type = typeof(filter.p[1])
    ll_vec = Vector{type}(undef, N)
    u=fill([],length(dataset[1]))

    for i in 1:N
        sol = forward_trajectory(filter, u, dataset[i])
        ll_vec[i] = sol.ll
    end

    return ll_vec
end

loss_function_sum(dataset,filter) = -sum(loglik_dataset(dataset,filter))/length(dataset)

loss_function_entropy(dataset,filter) = begin
    ll_vec = loglik_dataset(dataset,filter,opt_params)
    p_vec = exp.(ll_vec.-maximum(ll_vec))
    s_path = p_vec./sum(p_vec).*ll_vec
    -sum(s_path)
end

loss_function_combined(dataset, filter, alpha=0.5) = (1-alpha)*loss_function_sum(dataset,filter)+alpha*loss_function_entropy(dataset,filter)