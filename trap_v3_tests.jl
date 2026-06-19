"
Lugar donde hacer pruebas de los solvers numéricos para chequear que todo funciona correctamente.
"

include("optical_trap_SDE_methods.jl")

# parámetros

model = SDEObs
solver_params = (; nssteps=4)#, ode_xo = zeros(10), sde_xo=zeros(6))

to,tf,Δt = 0.0,50.0,0.05
tpoints = to:Δt:tf

ux = -0.1
uy = -0.1
vx = 0.1
vy = 0.1
errx = 1e-9
erry = 1e-9
σ = 0.05
feedback_params=SArray{Tuple{7},Float64}(ux,uy,vx,vy,errx,erry,σ)

ode_abstol = 1e-10
ode_reltol = 1e-6
ode_xo = thermal_values(n₀)

## Chequear las magnitudes tipicas de los distintos parámetros que gobiernan la dinámica.


println("Para chequear la magnitud del ruido por la dinámica monitorieada del sistema, y los parametros de la solución")

println("La magnitud inicial es: $(round.(thermal_values(n₀)[[1,3]],sigdigits=2)) para la posición, y $(round.(thermal_values(n₀)[[8,10]],sigdigits=2)) para el momento")

prob_ode = ODEProblem(ode_drift!, ode_xo, (to,tf))
sol_ode = solve(prob_ode, abstol=ode_abstol, reltol=ode_reltol, saveat=to:Δt:tf)
t_estim = 0.0:π/Ω/100:π/Ω

maximum_values = round.(maximum.([abs.(sol_ode[i,:]) for i in eachindex(ode_xo)]),sigdigits=2)

println("Las magnitudes máximas alcanzadas para cada correlador fueron $(maximum_values)")

println("La magnitud del ruido será en este caso:")
w=zeros(6,3)
sde_diffusion!(w,feedback_params,maximum_values)
show(stdout,"text/plain",round.(w,sigdigits=2))
println()
println("Con un stepsize $Δt, esto sería una influencia del orden:")
show(stdout,"text/plain",round.(sqrt(Δt).*w,sigdigits=2))
println()
println()

println("Para chequear, pude encontrar los siguientes ordenes en el ruido de la aproximación de la parte estocastica de la ecuación en nuestro caso")
println("Son dos, ambos de orden Δt^{3/2}. Voy a estimar su magnitud. El primero es de orden")
a=zeros(6,3)
b=zeros(6,3)
dotw=zeros(6,3)
for i in eachindex(to+Δt:Δt:tf)
    sde_diffusion!(a,feedback_params,sol_ode[:,i])
    sde_diffusion!(b,feedback_params,sol_ode[:,i+1])
    diff = (b-a)/Δt
    dotw .= [maximum([dotw[i,j],abs(diff[i,j])]) for i in 1:6, j in 1:3]
end
show(stdout,"text/plain",round.(sqrt(Δt^3/3).*dotw,sigdigits=2))
println()
println("Y el segundo término de orden:")
a=zeros(6,3)
b=zeros(6,3)
wgrada=zeros(6,3)
for i in eachindex(to:Δt:tf)
    sde_diffusion!(a,feedback_params,sol_ode[:,i])
    b=sde_drift_jacobian(zeros(6),[],feedback_params,i,to,Δt)
    d = b * a
    wgrada = [maximum([dotw[i,j],abs(d[i,j])]) for i in 1:6, j in 1:3]
end
show(stdout,"text/plain",round.(sqrt(Δt^3/3).*wgrada,sigdigits=2))
println()

println()

println("En cuanto a la fuerza, asumiendo que el valor medio de la posición y el momento es nulo (subestimando entonces), pero usando los máximos de la dispersión:")
x_maxes = sqrt.(maximum_values[[1,3,8,10]])
println("Los valores típicos, basados únicamente en la dispersión serán del orden $(round.(x_maxes,sigdigits=2))")
f = zeros(6)
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

# preparo la distribución inicial

kf = trap_kalman_filter(to,tf,Δt,feedback_params,d0(feedback_params,true,obs_vec[1]);ode_abstol=1e-10,ode_reltol=1e-6,supersample=1)
sol = forward_trajectory(kf,fill([],length(obs_vec)),obs_vec,kf.p)
inf_vec=[sol.x[i][j] for i in eachindex(sol.x), j in 1:6]

print("Done.")
## Genero una figura para esto

fig1=Figure(size=(1000,700))
g1l = fig1[1:2,1]
g1ur = fig1[1,2]
g1br = fig1[2,2]


ax11 = Axis(g1l[2,1], xlabel = "Time [us]", ylabel = L"$\left<x\right>$ [nm]")
ax12 = Axis(g1l[3,1], xlabel = "Time [us]", ylabel = L"$\left<y\right>$ [nm]")
lines!(ax11, tpoints, state_vec[:,1], label = "True")
lines!(ax12, tpoints, state_vec[:,2], label = "True")
lines!(ax11, tpoints, inf_vec[:,1], label = "Infered")
lines!(ax12, tpoints, inf_vec[:,2], label = "Infered")
lines!(ax11, tpoints, obs_positions[:,1], label = "Measured", alpha=0.5)
lines!(ax12, tpoints, obs_positions[:,2], label = "Measured", alpha=0.5)
leg=Legend(g1l[0,1], [ax11,ax12], merge=true, orientation = :horizontal, tellheight = true)
Label(g1l[1, 1], "Posición", font = :bold, tellwidth=false)
hidexdecorations!(ax11, grid = false, ticks=false)

ax21 = Axis(g1ur[1,1], xlabel = "Time [us]", ylabel = L"$\left<p_x\right>$ [ev us/nm]")
ax22 = Axis(g1ur[1,2], xlabel = "Time [us]", ylabel = L"$\left<p_y\right>$ [ev us/nm]")
lines!(ax21, tpoints, state_vec[:,3]*1e8, label = "True")
lines!(ax22, tpoints, state_vec[:,4]*1e8, label = "True")
lines!(ax21, tpoints, inf_vec[:,3]*1e8, label = "Infered")
lines!(ax22, tpoints, inf_vec[:,4]*1e8, label = "Infered")
Label(g1ur[0, 1:2], "Momento", font = :bold, tellwidth=false)
Label(g1ur[1, 1, Top()], halign = :left, L"\times 10^{-8}")
Label(g1ur[1, 2, Top()], halign = :left, L"\times 10^{-8}")
#hidexdecorations!(ax11, grid = false, ticks=false)


ax31 = Axis(g1br[1,1], xlabel = "Time [us]", ylabel = L"$\left<q_x\right>$ [ev us/nm]")
ax32 = Axis(g1br[1,2], xlabel = "Time [us]", ylabel = L"$\left<q_y\right>$ [ev us/nm]")
lines!(ax31, tpoints, state_vec[:,5]*1e8, label = "True")
lines!(ax32, tpoints, state_vec[:,6]*1e8, label = "True")
lines!(ax31, tpoints, inf_vec[:,5]*1e8, label = "Infered")
lines!(ax32, tpoints, inf_vec[:,6]*1e8, label = "Infered")
Label(g1br[0, 1:2], "Momento Estimado", font = :bold, tellwidth=false)
Label(g1br[1, 1, Top()], halign = :left, L"\times 10^{-8}")
Label(g1br[1, 2, Top()], halign = :left, L"\times 10^{-8}")
fig1

save("Graphics/state_inference.png", fig1)
## Genero un dataset para lo siguiente

N=1000
dataset = []
for _ in 1:N
    obs_trace = simulate(model, (tpoints, (ux,uy,vx,vy,errx,erry,σ), solver_params))
    tr_state_vec, tr_obs_positions = get_retval(obs_trace)
    tr_obs_vec = [tr_obs_positions[k,:] for k in 1:size(tr_obs_positions)[1]]
    push!(dataset,tr_obs_vec)
end

## Puedo inferir los parámetros que rigen sobre el ruido del momento estimado?

function define_problem(dataset, loss_function, to, tf, Δt, noopt_params; ode_xo=thermal_values(n₀), ode_abstol=1e-10, ode_reltol=1e-6)
    ux, uy, σ = noopt_params
    loss(opt_params::Vector{T},_) where T = begin
        p=SArray{Tuple{7},T}(ux,uy,opt_params...,σ)
        filter = trap_kalman_filter(to,tf,Δt,p,d0(p);ode_xo=thermal_values(n₀),ode_abstol=1e-10,ode_reltol=1e-6)
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
exprange=-3:0.1:3

l1=[p_test(p_alt1(e)) for e in exprange]
l2=[p_test(p_alt2(e)) for e in exprange]

## Gráfico de esta cosa.

fig1=Figure(figsize=(600,600))
Label(fig1[0, 0:2], L"Modelado inverso basado en $N=1000$ trayectorias", tellheight=true, tellwidth=false)
Label(fig1[1, 0], L"-\left<\log(\mathcal{L}\left(\mathbf{\theta}|\{\mathbf{x}^{OBS}_{i,t_o:t_f}\}_{i=1}^N)\right)\right>", tellheight=false, rotation = pi/2)
ax11=Axis(fig1[1,1], xlabel = "e", ylabel = "Mean Loss", title=L"v\sim v_0\cdot 10^e")
ax12=Axis(fig1[1,2], xlabel = "e", ylabel = "Mean Loss", title=L"\bar{\bar{G}}_{\mathbf q}\sim \bar{\bar{G}}_{\mathbf q 0}\cdot 10^e")
lines!(ax11, exprange, [l[1] for l in l1])
errorbars!(ax11, exprange, [l[1] for l in l1], [l[2]/sqrt(N) for l in l1])
lines!(ax12, exprange, [l[1] for l in l2])
errorbars!(ax12, exprange, [l[1] for l in l2], [l[2]/sqrt(N) for l in l2])
#hidexdecorations!(ax11, ticks = false, grid = false)
hideydecorations!(ax11)
hideydecorations!(ax12)
fig1
save("Graphics/param_inference.png", fig1)

#adtype = AutoForwardDiff()
#optf = OptimizationFunction(define_problem(dataset,loss_function_sum,to,tf,Δt,(ux,uy,σ)),adtype)
#prob = OptimizationProblem(optf, p_start)#, lb = zeros(4), ub=10 .*p)
#sol = solve(prob, Optim.GradientDescent())


## Puedo aunque sea optimizar ux y uy tal que se minimize la varianza de la posición al final?
#=

tf_vec = [10.0,20.0,50.0,100.0,200.0,500.0]
Δt_vec = [0.05,0.1,1.0]
sol = zeros(length(tf_vec),length(Δt_vec),2)

for i in eachindex(tf_vec), j in eachindex(Δt_vec)
    optf = OptimizationFunction(final_variance_problem(to,tf_vec[i],Δt_vec[j],feedback_params[3:end]),adtype)
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

=#
## Ahora me fijo como depende la varianza a tiempo 500.0 con u

function final_variance_problem(to, tf, Δt, noopt_params)
    loss(opt_params::Vector{T},_) where T = begin
        p = SArray{Tuple{7},T}(opt_params...,noopt_params...)
        filter = trap_kalman_filter(to,tf,Δt,p,d0(p,false))
        
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

tfo=500.0

loss = final_variance_problem(to,tfo,Δt,feedback_params[3:end])
optf = OptimizationFunction(loss,adtype)
prob = OptimizationProblem(optf, p_start)#, lb = zeros(4), ub=10 .*p)
sol = solve(prob, Optim.GradientDescent())
u_opt=sol.u

exprange=-1:0.05:1
vals_pred = [loss([u_opt[1]*10^e1, u_opt[2]*10^e2],[]) for e1 in exprange, e2 in exprange]

##

p_u=SA[u_opt...,feedback_params[3:end]...]
kf = trap_kalman_filter(to,tfo,Δt,p_uopt,d0(p_u);ode_abstol=1e-10,ode_reltol=1e-6,supersample=1)

covariances_opt=[]

for _ in 1:Int(tfo/Δt)
    predict!(kf, [], SA[u_opt...,feedback_params[3:end]...])
    push!(covariances_opt,tr(covariance(kf)[1:2,1:2]))
end

p_u=SA[(u_opt.*10)...,feedback_params[3:end]...]
kf = trap_kalman_filter(to,tfo,Δt,p_u,d0(p_u);ode_abstol=1e-10,ode_reltol=1e-6,supersample=1)

covariances_high=[]

for _ in 1:Int(tfo/Δt)
    predict!(kf, [], SA[(u_opt.*10)...,feedback_params[3:end]...])
    push!(covariances_high,tr(covariance(kf)[1:2,1:2]))
end

p_u=SA[(u_opt./10)...,feedback_params[3:end]...]
kf = trap_kalman_filter(to,tfo,Δt,p_u,d0(p_u);ode_abstol=1e-10,ode_reltol=1e-6,supersample=1)

covariances_low=[]

for _ in 1:Int(tfo/Δt)
    predict!(kf, [], SA[(u_opt./10)...,feedback_params[3:end]...])
    push!(covariances_low,tr(covariance(kf)[1:2,1:2]))
end
##

fig2=Figure(size=(450,450))
g1 = fig2[1,1]
g2 = fig2[2,1]
ax21 = Axis(g1[1,1], xlabel="Time [us]", ylabel=L"\text{tr}\left(\bar{\bar{C}}_{\mathbf{x}}(t)\right)", yscale=log10)
lines!(ax21, Δt:Δt:tfo, covariances_opt, label = L"\mathbf{u}_{opt}")
lines!(ax21, Δt:Δt:tfo, covariances_high, label = L"10\cdot\mathbf{u}_{opt}")
lines!(ax21, Δt:Δt:tfo, covariances_low, label = L"1/10\cdot\mathbf{u}_{opt}")
Legend(g1[0,1],[ax21],orientation = :horizontal,tellwidth=false, tellheight=true, padding = (0, 0, 0, 0))
ax22 = Axis(g2[1,1], xlabel=L"\log(u_x/u_{x0})", ylabel=L"\log(u_y/u_{y0})")
hm=heatmap!(ax22, exprange, exprange, log10.(vals_pred))
cb=Colorbar(g2[1,2],hm,label=L"\text{tr}\left(\bar{\bar{C}}_{\mathbf{x}}(t_f)\right) \; [\log_{10}]", tellwidth=true, alignmode=Mixed(right=0))
#colsize!(fig2.layout, 2, Auto(0.2))
fig2
save("Graphics/covar_optim.png", fig2)

##

fig=Figure(size=(800,400))
gl=fig[1,1]
gr=fig[1,2]

#Label(gl[0, 0:2], L"Modelado inverso basado en $N=1000$ trayectorias", tellheight=true, tellwidth=false)
Label(gl[1, 0], L"-\left<\log(\mathcal{L}\left(\mathbf{\theta}|\{\mathbf{x}^{OBS}_{i,t_o:t_f}\}_{i=1}^{%$N})\right)\right>", tellheight=false, rotation = pi/2)
Label(gl[0, 1], L"v\sim v_0\cdot 10^e", tellheight=true, tellwidth=false)
Label(gl[0, 2], L"\bar{\bar{G}}_{\mathbf q}\sim \bar{\bar{G}}_{\mathbf q 0}\cdot 10^e", tellheight=true, tellwidth=false)
ax11=Axis(gl[1,1], xlabel = "e", ylabel = "Mean Loss")
#Label(gl[0, 0:2], L"Modelado inverso basado en $N=1000$ trayectorias", tellheight=true, tellwidth=false)
ax12=Axis(gl[1,2], xlabel = "e", ylabel = "Mean Loss")
lines!(ax11, exprange, [l[1] for l in l1])
errorbars!(ax11, exprange, [l[1] for l in l1], [l[2]/sqrt(N) for l in l1])
lines!(ax12, exprange, [l[1] for l in l2])
errorbars!(ax12, exprange, [l[1] for l in l2], [l[2]/sqrt(N) for l in l2])
#hidexdecorations!(ax11, ticks = false, grid = false)
hideydecorations!(ax11)
hideydecorations!(ax12)

g21 = gr[1,1]
g22 = gr[2,1]
ax21 = Axis(g21[1,1], xlabel="Time [us]", ylabel=L"\text{tr}\left(\bar{\bar{C}}_{\mathbf{x}}(t)\right)", yscale=log10)
lines!(ax21, Δt:Δt:tfo, covariances_opt, label = L"\mathbf{u}_{opt}")
lines!(ax21, Δt:Δt:tfo, covariances_high, label = L"10\cdot\mathbf{u}_{opt}")
lines!(ax21, Δt:Δt:tfo, covariances_low, label = L"1/10\cdot\mathbf{u}_{opt}")
Legend(g21[0,1],[ax21],orientation = :horizontal,tellwidth=false, tellheight=true, padding = (0, 0, 0, 0))
ax22 = Axis(g22[1,1], xlabel=L"\log(u_x/u_{x0})", ylabel=L"\log(u_y/u_{y0})")
hm=heatmap!(ax22, exprange, exprange, log10.(vals_pred))
cb=Colorbar(g22[1,2],hm,label=L"\text{tr}\left(\bar{\bar{C}}_{\mathbf{x}}(t_f)\right) \; [\log_{10}]", tellwidth=true, alignmode=Mixed(right=0))
#colsize!(fig2.layout, 2, Auto(0.2))
save("Graphics/mega_fig.png", fig)
fig




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