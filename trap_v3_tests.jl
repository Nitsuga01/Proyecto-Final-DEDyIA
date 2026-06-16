"
Lugar donde hacer pruebas de los solvers numéricos para chequear que todo funciona correctamente.
"

include("optical_trap_SDE_methods.jl")

## Para analizar las trayectorias y el comportamiento del sistema en función de los parámetros
# y para comprobar que el modelo puede ajustarlo relativamente bien.

model = SDEObs
tpoints = 0.0:0.1:150.0

ux = -0.001
uy = -0.001
vx = 10
vy = 10
errx = 1e-6
erry = 1e-6
σ = 0.01

# cosas que tienen que ver con Gen
feedback_params = (ux,uy,vx,vy,errx,erry)
solver_params = (; nssteps=4)#, ode_xo = zeros(10), sde_xo=zeros(6))
obs_trace = simulate(model, (tpoints, feedback_params, σ, solver_params))
state_vec, obs_positions = get_retval(obs_trace)
obs_vec = [obs_positions[k,:] for k in 1:size(obs_positions)[1]]

s = thermal_values(n₀)
d0 = MvNormal(zeros(6),diagm(s[[1,3,8,10,8,10]]))
kf = trap_kalman_filter(d0,0.0,150.0,0.1,(feedback_params...,σ);ode_abstol=1e-10,ode_reltol=1e-6)
sol=LowLevelParticleFilters.forward_trajectory(kf,fill([],length(obs_vec)),obs_vec,kf.p)
inf_vec=[sol.x[i][j] for i in eachindex(sol.x), j in 1:6]

print("Done.")

fig1=Figure(title="Posición")
ax11 = Axis(fig1[1,1], xlabel = "t (us)", ylabel = "x (nm)")
ax12 = Axis(fig1[1,2], xlabel = "t (us)", ylabel = "y (nm)")
lines!(ax11, tpoints, obs_positions[:,1], label = "measured")
lines!(ax11, tpoints, inf_vec[:,1], label = "infered")
lines!(ax11, tpoints, state_vec[:,1], label = "true")
lines!(ax12, tpoints, obs_positions[:,2], label = "measured")
lines!(ax12, tpoints, inf_vec[:,2], label = "infered")
lines!(ax12, tpoints, state_vec[:,2], label = "true")
fig1[0,1:2]=Legend(fig1, [ax11,ax12], merge=true, orientation = :horizontal, tellheight = true)
fig1

fig2=Figure(title="Momento")
ax21 = Axis(fig2[1,1], xlabel = "t (us)", ylabel = "px (ev us / nm)")
ax22 = Axis(fig2[1,2], xlabel = "t (us)", ylabel = "py (ev us / nm)")
lines!(ax21, tpoints, inf_vec[:,3], label = "infered")
lines!(ax21, tpoints, state_vec[:,3], label = "true")
lines!(ax22, tpoints, inf_vec[:,4], label = "infered")
lines!(ax22, tpoints, state_vec[:,4], label = "true")
fig2[0,1:2]=Legend(fig2, [ax21,ax22], merge=true, orientation = :horizontal, tellheight = true)
fig2

fig3=Figure(title="Momento Estimado")
ax31 = Axis(fig3[1,1], xlabel = "t (us)", ylabel = "px (ev us / nm)")
ax32 = Axis(fig3[1,2], xlabel = "t (us)", ylabel = "py (ev us / nm)")
lines!(ax31, tpoints, inf_vec[:,5], label = "infered")
lines!(ax31, tpoints, state_vec[:,5], label = "true")
lines!(ax32, tpoints, inf_vec[:,6], label = "infered")
lines!(ax32, tpoints, state_vec[:,6], label = "true")
fig3[0,1:2]=Legend(fig3, [ax31,ax32], merge=true, orientation = :horizontal, tellheight = true)
fig3

## Genero un dataset para lo siguiente

model = SDEObs

to,tf,Δt=0.0,150.0,0.1
ux = -0.001
uy = -0.001
vx = 10
vy = 10
errx = 1e-6
erry = 1e-6
σ = 0.01

tpoints = to:Δt:tf
feedback_params = (ux,uy,vx,vy,errx,erry)
solver_params = (; nssteps=4)#, ode_xo = zeros(10), sde_xo=zeros(6))

N=100
dataset = []
for _ in 1:N
    obs_trace = simulate(model, (tpoints, (ux,uy,vx,vy,errx,erry), σ, solver_params))
    state_vec, obs_positions = get_retval(obs_trace)
    obs_vec = [obs_positions[k,:] for k in 1:size(obs_positions)[1]]
    push!(dataset,obs_vec)
end

## Puedo inferir los parámetros que rigen sobre el ruido del momento estimado?

function define_problem(dataset, loss_function, to, tf, Δt, noopt_params; ode_xo=thermal_values(n₀), ode_abstol=1e-10, ode_reltol=1e-6)
    ux, uy, σ = noopt_params
    loss(opt_params::Vector{T},_) where T = begin
        s = thermal_values(n₀)
        d0 = MvNormal(T.(zeros(6)),T.(diagm(s[[1,3,8,10,8,10]])))
        p=T[T.(ux),T.(uy),opt_params...,T.(σ)]
        filter = trap_kalman_filter(d0,to,tf,Δt,p;ode_xo=s,ode_abstol=1e-10,ode_reltol=1e-6)
        loss_function(dataset,filter)
    end
    return loss
end

s = thermal_values(n₀)
d0 = MvNormal(zeros(6),diagm(s[[1,3,8,10,8,10]]))
kf = trap_kalman_filter(d0,to,tf,Δt,(feedback_params...,σ);ode_abstol=1e-10,ode_reltol=1e-6)
loss=define_problem(dataset,loglik_dataset,to,tf,Δt,(ux,uy,σ))

p_opt=[vx,vy,errx,erry]
p_alt=p_opt.*(1 .+ 0.1*randn(size(p_opt)))

loss_opt=loss(p_opt,0)
loss_alt=loss(p_alt,0)

m_opt=mean(loss_opt)
m_alt=mean(loss_alt)
std_opt=std(loss_opt)/sqrt(N)
std_alt=std(loss_alt)/sqrt(N)

print("Para N=$N. La loss media con parámetros óptimos: $m_opt ± $std_opt, con parámetros alternativos: $m_alt ± $std_alt")

p_alt1(exponent)=[vx*10^exponent,vy,errx,erry]
p_alt2(exponent)=[vx,vy,errx*10^exponent,erry]

p_test(params) = begin l=loss(params,0)
    return mean(l),std(l)
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

to,tf,Δt=0.0,100.0,0.1
ux = -0.00312
uy = -0.0160
vx = 10
vy = 10
errx = 1e-6
erry = 1e-6
σ = 0.01

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
optf = OptimizationFunction(final_variance_problem(to,tf,Δt,(feedback_params[3:end]...,σ)),adtype)
prob = OptimizationProblem(optf, p_start)#, lb = zeros(4), ub=10 .*p)
sol = solve(prob, Optim.GradientDescent())



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