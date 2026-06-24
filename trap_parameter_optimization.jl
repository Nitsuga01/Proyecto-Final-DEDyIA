"
Lugar para optimizar parámetros del sistema sin involucrar fitteos de trayectorias.
"

include("optical_trap_SDE_methods.jl")

# función que devuleve una función que fija los parámetros a no ajustar y, dado un set de parámetros, devuelve la cantidad a minimizar
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

# parámetros

model = SDEObs
solver_params = (; nssteps=4)#, ode_xo = zeros(10), sde_xo=zeros(6))

to,tf,Δt = 0.0,500.0,0.05
tpoints = to:Δt:tf

ux = -0.1
uy = -0.1
vx = 0.1
vy = 0.1
errx = 1e-9
erry = 1e-9
σ = 0.005
feedback_params=SArray{Tuple{7},Float64}(ux,uy,vx,vy,errx,erry,σ)

ode_abstol = 1e-10
ode_reltol = 1e-6
ode_xo = thermal_values(n₀)

p_start = [ux,uy]
adtype = AutoForwardDiff()

## Calcula el valor óptimo para minimizar la covarianza y luego el aspecto de la función de costo alrededor suyo.

loss = final_variance_problem(to,tf,Δt,feedback_params[3:end])
optf = OptimizationFunction(loss,adtype)
prob = OptimizationProblem(optf, p_start)#, lb = zeros(4), ub=10 .*p)
sol = solve(prob, Optim.GradientDescent())
u_opt=sol.u

# barre valores similares de los parámetros y estima la función de costo.
exprange=-1:0.05:1
vals_pred = [loss([u_opt[1]*10^e1, u_opt[2]*10^e2],[]) for e1 in exprange, e2 in exprange]

## Calcula la evolución en el tiempo de las fluctuaciones para tres parámetros distintos mediante filtros de Kalman.

p_uopt=SA[u_opt...,feedback_params[3:end]...]
kf = trap_kalman_filter(to,tf,Δt,p_uopt,d0(p_uopt);ode_abstol=1e-10,ode_reltol=1e-6,supersample=1)

covariances_opt=[]

for _ in 1:Int(tf/Δt)
    predict!(kf, [], p_uopt)
    push!(covariances_opt,tr(covariance(kf)[1:2,1:2]))
end

p_uhigh=SA[(u_opt.*10)...,feedback_params[3:end]...]
kf = trap_kalman_filter(to,tf,Δt,p_uhigh,d0(p_uhigh);ode_abstol=1e-10,ode_reltol=1e-6,supersample=1)

covariances_high=[]

for _ in 1:Int(tf/Δt)
    predict!(kf, [], p_uhigh)
    push!(covariances_high,tr(covariance(kf)[1:2,1:2]))
end

p_ulow=SA[(u_opt./10)...,feedback_params[3:end]...]
kf = trap_kalman_filter(to,tf,Δt,p_ulow,d0(p_ulow);ode_abstol=1e-10,ode_reltol=1e-6,supersample=1)

covariances_low=[]

for _ in 1:Int(tf/Δt)
    predict!(kf, [], p_ulow)
    push!(covariances_low,tr(covariance(kf)[1:2,1:2]))
end

## Genero dataset de la evolución del estado en el tiempo con distintos parámetros mediante integración directa

function gen_dataset(N,model,params)
    dataset=Array{Float64,3}(undef,N,length(params[1]),2)
    for i in 1:N
        obs_trace = simulate(model, params)
        state_vec, _ = get_retval(obs_trace)
        dataset[i,:,:].=state_vec[:,1:2]
    end
    dataset
end

dataset = gen_dataset(1000, SDEObs, (tpoints, p_uopt, solver_params))
dataset_high = gen_dataset(1000, SDEObs, (tpoints, p_uhigh, solver_params))
dataset_low = gen_dataset(1000, SDEObs, (tpoints, p_ulow, solver_params))

## Analiza el dataset para obtener los valores de fluctuación para cada tiempo obtenidos por integración numérica de las ecuaciones.

posmed = mean(dataset, dims=1)
posvar = [cov(dataset[:,k,:], dims=1) for k in eachindex(tpoints)]
losstrue = tr.(posvar)

posmed_high = mean(dataset_high, dims=1)
posvar_high = [cov(dataset_high[:,k,:], dims=1) for k in eachindex(tpoints)]
losstrue_high = tr.(posvar_high)

posmed_low = mean(dataset_low, dims=1)
posvar_low = [cov(dataset_low[:,k,:], dims=1) for k in eachindex(tpoints)]
losstrue_low = tr.(posvar_low)

lossopt = loss(u_opt,0)
losshigh = loss(10.0*u_opt,0)
losslow = loss(u_opt/10,0)

println("La suma de varianzas predicha por el optimizador: $lossopt. La suma de varianzas real: $(losstrue[end])")
println("La suma de varianzas predicha por el optimizador: $losshigh. La suma de varianzas real: $(losstrue_high[end])")
println("La suma de varianzas predicha por el optimizador: $losslow. La suma de varianzas real: $(losstrue_low[end])")


## Grafica las cosas

fig=Figure(size=(1000,500))

g21 = fig[1,1]
g22 = fig[1,2]
ax21 = Axis(g21[1,1], xlabel="Time [us]", ylabel=L"\text{tr}\left(C_{\left\langle\vec{x}\right\rangle(t)}\right)", yscale=log10)
l1=lines!(ax21, Δt:Δt:tf, covariances_opt,label=L"\vec{u}_{0}",color=Cycled(1))
l2=lines!(ax21, Δt:Δt:tf, covariances_high,label=L"10\cdot\vec{u}_{0}",color=Cycled(2))
l3=lines!(ax21, Δt:Δt:tf, covariances_low,label=L"1/10\cdot\vec{u}_{0}",color=Cycled(3))
l4=lines!(ax21, tpoints, losstrue,label=L"\vec{u}_{0}",color=Cycled(1),linestyle=:dash)
l5=lines!(ax21, tpoints, losstrue_high,label=L"10\cdot\vec{u}_{0}",color=Cycled(2),linestyle=:dash)
l6=lines!(ax21, tpoints, losstrue_low,label=L"1/10\cdot\vec{u}_{0}",color=Cycled(3),linestyle=:dash)

labels = [L"\vec{u}_{0}",L"\vec{u}_{0}",L"10\cdot\vec{u}_{0}",L"10\cdot\vec{u}_{0}",L"1/10\cdot\vec{u}_{0}",L"1/10\cdot\vec{u}_{0}"]
Legend(g21[0,1],[ax21], orientation = :horizontal,tellwidth=false, tellheight=true, padding = (0, 0, 0, 0), nbanks=1, merge=true)
#Legend(g21[0,1],[l1,l4,l2,l5,l3,l6], labels,orientation = :horizontal,tellwidth=false, tellheight=true, padding = (0, 0, 0, 0), nbanks=2, merge=true)
ax22 = Axis(g22[1,1], xlabel=L"\log(u_x/u_{x0})", ylabel=L"\log(u_y/u_{y0})")
hm=heatmap!(ax22, exprange, exprange, log10.(vals_pred))
cb=Colorbar(g22[1,2],hm,label=L"\text{tr}\left(C_{\left\langle\vec{x}(t_f)\right\rangle}\right) \; [\log_{10}]", tellwidth=true, alignmode=Mixed(right=0))
Label(fig[1, 1, TopLeft()], "a)",
        fontsize = 20,
        font = :bold,
        padding = (0, 5, 5, 0),
        halign = :right)
Label(fig[1, 2, TopLeft()], "b)",
        fontsize = 20,
        font = :bold,
        padding = (0, 5, 5, 0),
        halign = :right)
#colsize!(fig2.layout, 2, Auto(0.2))
save("Graphics/param_optimization.pdf", fig)
fig