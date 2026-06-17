"
Lugar donde hacer pruebas de los solvers numéricos para chequear que todo funciona correctamente.
"

include("optical_trap_SDE_methods.jl")

# parámetros

model = SDEObs
solver_params = (; nssteps=4)#, ode_xo = zeros(10), sde_xo=zeros(6))

to,tf,Δt = 0.0,50.0,0.1
tpoints = to:Δt:tf

ux = -0.1
uy = -0.1
vx = 0.1
vy = 0.1
errx = 1e-9
erry = 1e-9
σ = 0.01
feedback_params=[ux,uy,vx,vy,errx,erry,σ]

ode_abstol = 1e-10
ode_reltol = 1e-6
ode_xo = thermal_values(n₀)

## Chequear las magnitudes tipicas de los distintos parámetros que gobiernan la dinámica.


println("Para chequear la magnitud del ruido por la dinámica monitorieada del sistema, y los parametros de la solución")

println("La magnitud inicial es: $(round.(thermal_values(n₀)[[1,3]],sigdigits=2)) para la posición, y $(round.(thermal_values(n₀)[[8,10]],sigdigits=2)) para el momento")

prob_ode = ODEProblem(ode_drift!, ode_xo, (to,tf))
sol_ode = solve(prob_ode, abstol=ode_abstol, reltol=ode_reltol, saveat=to:Δt:tf)

maximum_values = round.(maximum.([abs.(sol_ode[i,:]) for i in eachindex(ode_xo)]),sigdigits=2)

println("Las magnitudes máximas alcanzadas para cada correlador fueron $(maximum_values)")

println("La magnitud del ruido será en este caso:")
w=zeros(6,3)
sde_diffusion!(w,feedback_params,maximum_values)
show(stdout,"text/plain",round.(w,sigdigits=2))
println("Con un stepsize $Δt, esto sería una influencia del orden:")
show(stdout,"text/plain",round.(sqrt(Δt).*w,sigdigits=2))
println()
println()

println("En cuanto a la fuerza, asumiendo que el valor medio de la posición y el momento es nulo (subestimando entonces), pero usando los máximos de la dispersión:")
x_maxes = sqrt.(maximum_values[[1,3,8,10]])
println("Los valores típicos, basados únicamente en la dispersión serán del orden $(round.(x_maxes,sigdigits=2))")
f = zeros(6)
t_estim = 0.0:2*π/Ω/100:2*π/Ω
for t in t_estim
    a=zeros(6)
    sde_drift!(a,[x_maxes...,0.0,0.0],feedback_params,t)
    f .= [maximum([f[i],abs(a[i])]) for i in eachindex(f)]    
end
println("La fuerza, a partir de esta estimación de la posición, y descartando el feedback será del orden $(round.(f[1:4],sigdigits=2))")
println("Con un stepsize $Δt, esto sería una influencia del orden $(round.(Δt*f[1:4],sigdigits=2))")
println()

println("Como proceso de ornstein-uhlenbeck, descartando la variación temporal del momento, la distancia tipica será de $(round.([errx,erry]./sqrt.([vx,vy].*2),sigdigits=2))")
println("Y el tiempo típico del decaimiento típica será de $(round.([1/vx,1/vy],sigdigits=2))")
println("Esto habría que compararlo con la magnitud típica del momento de $(round.(x_maxes[3:4],sigdigits=2))")
b=zeros(6)
sde_drift!(b,[0.0,0.0,0.0,0.0,x_maxes[3:4]...],feedback_params,0.0)
println("Asumiendo que le pegamos al momento, la fuerza de feedback sería de orden $(round.(b[3:4],sigdigits=2))")
println("Teniendo en cuenta un stepsize de $Δt, esto es influencias de orden $(round.(b[3:4].*Δt,sigdigits=2)) en un paso temporal")
sde_drift!(b,[0.0,0.0,0.0,0.0,errx*sqrt.(vx*2),erry*sqrt(vy*2)],feedback_params,0.0) 
println("Las fluctuaciones en la fuerza serán de orden $(round.(b[3:4],sigdigits=2))")
println("Teniendo en cuenta un stepsize de $Δt, esto es fluctuaciones de orden $(round.(b[3:4].*Δt,sigdigits=2)) en un paso temporal")
println()

println("Las partes de la dinámica que dependen fluctuaciones en los parámetros son la fuerza armónica y de feedback")
println("En este caso, su sensitividad en un $Δt está dada por la siguiente matriz:")
mats = [sde_drift_jacobian(zeros(6),[],feedback_params,i,to,Δt) for i in eachindex(t_estim)]
sens_max = [maximum([abs(mats[j][i,k]) for j in eachindex(t_estim)]) for i in 1:6, k in 1:6]
show(stdout,"text/plain",round.(sens_max.*Δt,sigdigits=2))
println()
println("----------------------------------------------------------------------------------")

## Para analizar las trayectorias y el comportamiento del sistema en función de los parámetros
# y para comprobar que el modelo puede ajustarlo relativamente bien.

obs_trace = simulate(model, (tpoints, feedback_params, solver_params))
state_vec, obs_positions = get_retval(obs_trace)
obs_vec = [obs_positions[k,:] for k in 1:size(obs_positions)[1]]

s = thermal_values(n₀)
covmat = diagm(s[[1,3,8,10,8,10]])
covmat[3,5] = covmat[5,3] = covmat[3,3]
covmat[4,6] = covmat[6,4] = covmat[4,4]
covmat[6,6] += errx^2/2/vx
covmat[5,5] += erry^2/2/vy
d0 = MvNormal(zeros(6),covmat)
kf = trap_kalman_filter(d0,0.0,150.0,0.1,(feedback_params...,σ);ode_abstol=1e-10,ode_reltol=1e-6,supersample=1)
sol = forward_trajectory(kf,fill([],length(obs_vec)),obs_vec,kf.p)
inf_vec=[sol.x[i][j] for i in eachindex(sol.x), j in 1:6]

print("Done.")

fig1=Figure(title="Posición")
ax11 = Axis(fig1[1,1], xlabel = "t (us)", ylabel = "x (nm)")
ax12 = Axis(fig1[1,2], xlabel = "t (us)", ylabel = "y (nm)")
lines!(ax11, tpoints, obs_positions[:,1], label = "Measured")
lines!(ax11, tpoints, inf_vec[:,1], label = "Infered")
lines!(ax11, tpoints, state_vec[:,1], label = "True")
lines!(ax12, tpoints, obs_positions[:,2], label = "Measured")
lines!(ax12, tpoints, inf_vec[:,2], label = "Infered")
lines!(ax12, tpoints, state_vec[:,2], label = "True")
fig1[0,1:2]=Legend(fig1, [ax11,ax12], merge=true, orientation = :horizontal, tellheight = true)
fig1

fig2=Figure(title="Momento")
ax21 = Axis(fig2[1,1], xlabel = "t (us)", ylabel = "px (ev us / nm)")
ax22 = Axis(fig2[1,2], xlabel = "t (us)", ylabel = "py (ev us / nm)")
lines!(ax21, tpoints, inf_vec[:,3], label = "Infered")
lines!(ax21, tpoints, state_vec[:,3], label = "True")
lines!(ax22, tpoints, inf_vec[:,4], label = "Infered")
lines!(ax22, tpoints, state_vec[:,4], label = "True")
fig2[0,1:2]=Legend(fig2, [ax21,ax22], merge=true, orientation = :horizontal, tellheight = true)
fig2

fig3=Figure(title="Momento Estimado")
ax31 = Axis(fig3[1,1], xlabel = "t (us)", ylabel = "qx (ev us / nm)")
ax32 = Axis(fig3[1,2], xlabel = "t (us)", ylabel = "qy (ev us / nm)")
lines!(ax31, tpoints, inf_vec[:,5], label = "Infered")
lines!(ax31, tpoints, state_vec[:,5], label = "True")
lines!(ax32, tpoints, inf_vec[:,6], label = "Infered")
lines!(ax32, tpoints, state_vec[:,6], label = "True")
fig3[0,1:2]=Legend(fig3, [ax31,ax32], merge=true, orientation = :horizontal, tellheight = true)
fig3

## Genero un dataset para lo siguiente

N=100
dataset = []
for _ in 1:N
    obs_trace = simulate(model, (tpoints, (ux,uy,vx,vy,errx,erry,σ), solver_params))
    state_vec, obs_positions = get_retval(obs_trace)
    obs_vec = [obs_positions[k,:] for k in 1:size(obs_positions)[1]]
    push!(dataset,obs_vec)
end

## Puedo inferir los parámetros que rigen sobre el ruido del momento estimado?

function define_problem(dataset, loss_function, to, tf, Δt, noopt_params; ode_xo=thermal_values(n₀), ode_abstol=1e-10, ode_reltol=1e-6)
    ux, uy, σ = noopt_params
    loss(opt_params::Vector{T},_) where T = begin
        s = thermal_values(n₀)
        covmat = diagm(s[[1,3,8,10,8,10]])
        covmat[3,5] = covmat[5,3] = covmat[3,3]
        covmat[4,6] = covmat[6,4] = covmat[4,4]
        covmat[6,6] += errx^2/2/vx
        covmat[5,5] += erry^2/2/vy
        d0 = MvNormal(T.(zeros(6)),T.(covmat))
        p=T[T.(ux),T.(uy),opt_params...,T.(σ)]
        filter = trap_kalman_filter(d0,to,tf,Δt,p;ode_xo=s,ode_abstol=1e-10,ode_reltol=1e-6)
        loss_function(dataset,filter)
    end
    return loss
end

loss = define_problem(dataset,loglik_dataset,to,tf,Δt,(ux,uy,σ))

p_opt=[vx,vy,errx,erry]
p_alt=p_opt.*(1 .+ 0.1*randn(size(p_opt)))

loss_opt=loss(p_opt,0)
loss_alt=loss(p_alt,0)

m_opt=-mean(loss_opt)
m_alt=-mean(loss_alt)
std_opt=std(loss_opt)/sqrt(N)
std_alt=std(loss_alt)/sqrt(N)

print("Para N=$N. La loss media con parámetros óptimos: $m_opt ± $std_opt, con parámetros alternativos: $m_alt ± $std_alt")

p_alt1(exponent)=[vx*10^exponent,vy,errx,erry]
p_alt2(exponent)=[vx,vy,errx*10^exponent,erry]

p_test(params) = begin l=loss(params,0)
    return -mean(l),std(l)
end
exprange=-2:0.1:2

l1=[p_test(p_alt1(e)) for e in exprange]
l2=[p_test(p_alt2(e)) for e in exprange]

fig1=Figure()
ax11=Axis(fig1[1,1], xlabel = "Exponente para v (log(param/param_opt))", ylabel = "loss")
ax12=Axis(fig1[1,2], xlabel = "Exponente para err (log(param/param_opt))", ylabel = "loss")
lines!(ax11, exprange, [l[1] for l in l1])
errorbars!(ax11, exprange, [l[1] for l in l1], [l[2] for l in l1])
lines!(ax12, exprange, [l[1] for l in l2])
errorbars!(ax12, exprange, [l[1] for l in l2], [l[2] for l in l2])
fig1

#adtype = AutoForwardDiff()
#optf = OptimizationFunction(define_problem(dataset,loss_function_sum,to,tf,Δt,(ux,uy,σ)),adtype)
#prob = OptimizationProblem(optf, p_start)#, lb = zeros(4), ub=10 .*p)
#sol = solve(prob, Optim.GradientDescent())

## Puedo aunque sea optimizar ux y uy tal que se minimize la varianza de la posición al final?

tpoints = to:Δt:tf
feedback_params = (ux,uy,vx,vy,errx,erry)
solver_params = (; nssteps=4)#, ode_xo = zeros(10), sde_xo=zeros(6))

function final_variance_problem(to, tf, Δt, noopt_params)
    loss(opt_params::Vector{T},_) where T = begin
        s = thermal_values(n₀)
        d0 = MvNormal(T.(zeros(6)),T.(diagm(s[[1,3,8,10,8,10]])))
        p=T[opt_params..., T.(noopt_params)...]

        filter = trap_kalman_filter(d0,to,tf,Δt,p)
        
        for i in eachindex(to:Δt:tf)
            predict!(filter,[],filter.p)
        end

        c=covariance(filter)
        return(tr(c[1:2,1:2]))
    end
    return loss
end

p_start = [ux,uy]
adtype = AutoForwardDiff()

tf_vec = [2.0,5.0,10.0,20.0,50.0]
Δt_vec = [0.001,0.01,0.1,1.0]
sol = zeros(length(tf_vec),length(Δt_vec),2)

for i in eachindex(tf_vec), j in eachindex(Δt_vec)
    optf = OptimizationFunction(final_variance_problem(to,tf_vec[i],Δt_vec[j],(feedback_params[3:end]...,σ)),adtype)
    prob = OptimizationProblem(optf, p_start)#, lb = zeros(4), ub=10 .*p)
    u_opt = solve(prob, Optim.GradientDescent()).u
    sol[i,j,:] .= u_opt
end

fig1=Figure()
ax11=Axis(fig1[1,1], xlabel="tf", ylabel="ux_opt",xscale=log10)
for k in eachindex(Δt_vec)
    lines!(ax11,tf_vec,sol[:,k,1],label="$(Δt_vec[k])")
end
ax12=Axis(fig1[1,2], xlabel="tf", ylabel="ux_opt",xscale=log10)
for k in eachindex(Δt_vec)
    lines!(ax12,tf_vec,sol[:,k,2],label="$(Δt_vec[k])")
end
fig1[0,1:2]=Legend(fig1, [ax11,ax12], merge=true, orientation = :horizontal, tellheight = true)
fig1

##
#=

n_particles = 100
ess_thresh=0.5

sim = simulate(model, (tpoints, (ux,uy,vx,vy,errx,erry), σ, solver_params))
obs_vec = get_retval(sim)


state = loglike_particle_filter(model, n_particles, obs_vec, tpoints, feedback_params, σ, ess_thresh, solver_params)
print("Done.")
println("El estimativo para el log-likelyhood: $(log_ml_estimate(state))")
println("El estimativo para el sample-size: $(effective_sample_size(state))")

traces = sample_unweighted_traces(state, 10)

fig2 = Figure()
ax21 = Axis(fig2[1,1], xlabel = "t (us)", ylabel = "y (nm)")
ax22 = Axis(fig2[1,2], xlabel = "t (us)", ylabel = "y (nm)")
for t in traces
    positions = get_retval(trace)
    scatter!(ax21, tpoints, positions[:,1], label = "x",alpha=0.1,color=:red)
    scatter!(ax21, tpoints, positions[:,2], label = "y",alpha=0.1,color=:black)
    scatter!(ax22, positions[:,1], positions[:,2],alpha=0.1)
end
positions = get_retval(sim)
lines!(ax21, tpoints, positions[:,1], label = "x",color=:red)
lines!(ax21, tpoints, positions[:,2], label = "y",color=:black)
lines!(ax22, positions[:,1], positions[:,2])
fig2

=#