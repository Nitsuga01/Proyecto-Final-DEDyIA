## Funciones necesarias para la solución de las ecuaciones diferenciales. Como las necesito en un formato distinto al de DifferentialEquations o el que usé en Gen.

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

function sde_drift(u,input,p,i,t0,Δt)
    # tiempo
    t = t0+i*Δt

    # valores del estado
    x, y, px, py, qx, qy = u

    # parametros de feedback de la trampa
    ux, uy, vx, vy, errx, erry, σ = p

    # parametros de fuerza en trampa
    km = I_G - 2 * sqrt(I_L*I_G) * cos(2 * Ω * t)
    kp = I_G + 2 * sqrt(I_L*I_G) * cos(2 * Ω * t)
    kxy = 4 * sqrt(I_L*I_G)*sin(2 * Ω * t)
    c = 2 * V_0 / w₀^2

    # Jacobiano de la trampa, es todo lineal, así que ganamos.
    return SA[px/m,py/m,c*(-2*km*x+kxy*y)+ux*qx,c*(-2*kp*y+kxy*x)+uy*qy,vx*(px-qx),vy*(py-qy)]
end

function sde_noise_matrix(u,input,p,t,second_moments)
    # fuerza de backaction, medición y segundos momentos de posición y momento en la trampa
    ux, uy, vx, vy, errx, erry, σ = p

    C_xx, C_xy, C_yy, C_xpx, C_xpy, C_ypx, C_ypy = second_moments

    # constantes que gobiernan la magnitud del ruido
    ax = sqrt(2 * Λx * ηx)
    ay = sqrt(2 * Λy * ηy)

    G=SA[ax * C_xx  ay * C_xy  0   ;
         ax * C_xy  ay * C_yy  0   ;
         ax * C_xpx ay * C_ypx 0   ;
         ax * C_xpy ay * C_ypy 0   ;
         0          0          errx;
         0          0          erry]
    
    Q = 1/2 .* G*G'
    
end

## Función que genera un filtro de Kalman precargado con las ecuaciones dinámicas de la trampa.

function trap_kalman_filter(to,tf,Δt,feedback_params,d0;ode_xo=thermal_values(n₀),ode_abstol=1e-10,ode_reltol=1e-6, supersample=1)
    prob_ode = ODEProblem(ode_drift!, ode_xo, (to,tf))
    sol_ode = solve(prob_ode, abstol=ode_abstol, reltol=ode_reltol, saveat=to:Δt:tf)
    
    # Matriz que da ruido de proceso al estado
    R1 = (u,input,p,i) -> sde_noise_matrix(u,input,p,i,sol_ode[i+1]).*Δt
    # Matriz de covarianza para el ruido de medición sobre el estado
    R2 = SA[feedback_params[7]^2 0.0; 0.0 feedback_params[7]^2]

    # Evolución
    continuous_dynamics = (u,input,p,i)->sde_drift(u,input,p,i,to,Δt)
    discrete_dynamics = Rk4(continuous_dynamics,Δt;supersample=supersample)
    measurement = (u,input,p,i)->SA[u[1],u[2]]

    kf=ExtendedKalmanFilter(discrete_dynamics, measurement, R1, R2, d0; p = feedback_params, α = 1.0, check = true, nu=0, nx=6, ny=2, Ts=1)
end

# Función que calcula los parámetros de la gaussiana no normalizada obenida al multiplicar dos gaussianas multivariadas. útil para bayes.
function gaussian_product(mu1, sigma1, mu2, sigma2)
    inv1 = inv(sigma1)
    inv2 = inv(sigma2)

    sigma = inv(inv1+inv2)
    mu = sigma*inv1*mu1 .+ sigma*inv2*mu2
    return mu, sigma
end

# Función que calcula el prior del sistema estocástico. Obs que son los valores térmicos de la covarianza salvo por
# el momento estimado, que estará en la misma posición que el momento real a menos de un error que va como el 
# error estacionario de su proceso de aproximación.
function d0(params=SArray{Tuple{7},Float64}(0.0,0.0,0.0,0.0,0.0,0.0,0.0),datapoint=false,y=SArray{Tuple{2},Float64}(0.0,0.0))
    # psrepara la distribución de estados iniciales
    type = typeof(params[1])
    s = type.(thermal_values(n₀))
    C1 = diagm(s[[1,3,8,10,8,10]])
    C1[3,5] = C1[5,3] = C1[3,3]
    C1[4,6] = C1[6,4] = C1[4,4]
    C1[5,5] += params[5]^2/2/params[3]
    C1[6,6] += params[6]^2/2/params[4]

    # si no hay dato inicial
    if !(datapoint)
        return MvNormal(SArray{Tuple{6},type}(0.0,0.0,0.0,0.0,0.0,0.0),SArray{Tuple{6,6},type}(C1...))
    end

    # prepara la distribución correspondiente a la medición
    Q = C1[1:2,1:2] .+ I(2)*params[7]^2
    
    # devuelve el posterior
    mu, sigma = gaussian_product(zeros(type,2),C1[1:2,1:2],y,Q)
    C1[1:2,1:2] .= sigma

    MvNormal(SArray{Tuple{6},type}(mu...,zeros(4)...),SArray{Tuple{6,6},type}(C1...))
end

## Calcula los valroes de loglikelyhood para un set de parámetros y cada trayectoria de un set de trayectorias. 

function loglik_dataset(dataset,filter)
    N=length(dataset)
    type = typeof(filter.p[1])
    ll_vec = Vector{type}(undef, N)
    u=fill([],length(dataset[1]))
    for i in 1:N
        mv = d0(filter.p,true,dataset[1][1])
        filter.x=mv.μ
        filter.R=mv.Σ
        try
            sol = forward_trajectory(filter, u, dataset[i])
            ll_vec[i] = sol.ll
        catch e
            if e isa ErrorException && e.msg[1:49] == "Cholesky factorization of α*R̃*α' failed at time "
                # Si no puede ajustar la trayectoria (lo que suele resultar en este error), voy a asignar una loss infinita, por lo que no vale la pena simular más.
                ll_vec[i:N] = -Inf
                break
            else
                rethrow(e)
            end 
        end
    end

    return ll_vec
end