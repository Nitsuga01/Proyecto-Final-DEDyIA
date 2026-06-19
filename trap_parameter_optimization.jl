"
Lugar donde hacer pruebas de los solvers numéricos para chequear que todo funciona correctamente.
"

include("optical_trap_SDE_methods.jl")

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
σ = 0.05
feedback_params=SArray{Tuple{7},Float64}(ux,uy,vx,vy,errx,erry,σ)

ode_abstol = 1e-10
ode_reltol = 1e-6
ode_xo = thermal_values(n₀)

p_start = [ux,uy]
adtype = AutoForwardDiff()

##

loss = final_variance_problem(to,tf,Δt,feedback_params[3:end])
optf = OptimizationFunction(loss,adtype)
prob = OptimizationProblem(optf, p_start)#, lb = zeros(4), ub=10 .*p)
sol = solve(prob, Optim.GradientDescent())
u_opt=sol.u

exprange=-1:0.05:1
vals_pred = [loss([u_opt[1]*10^e1, u_opt[2]*10^e2],[]) for e1 in exprange, e2 in exprange]

##

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

## Genero dataset con los parametros óptimos.

function gen_dataset(N,model,params)
    dataset=Array{Float64,2}(undef,N,2)
    for i in 1:N
        obs_trace = simulate(model, params)
        state_vec, _ = get_retval(obs_trace)
        dataset[i,:].=state_vec[end,1:2]
    end
    dataset
end

dataset = gen_dataset(1000, SDEObs, (tpoints, p_uopt, solver_params))
dataset_high = gen_dataset(1000, SDEObs, (tpoints, p_uhigh, solver_params))
dataset_low = gen_dataset(1000, SDEObs, (tpoints, p_ulow, solver_params))

posmed = mean(dataset, dims=1)
posvar = cov(dataset, dims=1)
losstrue = tr(posvar)

posmed_high = mean(dataset_high, dims=1)
posvar_high = cov(dataset_high, dims=1)
losstrue_high = tr(posvar_high)

posmed_low = mean(dataset_low, dims=1)
posvar_low = cov(dataset_low, dims=1)
losstrue_low = tr(posvar_low)

lossopt = loss(u_opt,0)
losshigh = loss(10.0*u_opt,0)
losslow = loss(u_opt/10,0)

println("La suma de varianzas predicha por el optimizador: $lossopt. La suma de varianzas real: $losstrue")
println("La suma de varianzas predicha por el optimizador: $losshigh. La suma de varianzas real: $losstrue_high")
println("La suma de varianzas predicha por el optimizador: $losslow. La suma de varianzas real: $losstrue_low")


#@gen function particle_loss(N,model,params)
    

#end


##

fig=Figure(size=(800,400))
#gl=fig[1,1]
gr=fig[1,2]

#=
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
=#

g21 = gr[1,1]
g22 = gr[2,1]
ax21 = Axis(g21[1,1], xlabel="Time [us]", ylabel=L"\text{tr}\left(\bar{\bar{C}}_{\mathbf{x}}(t)\right)", yscale=log10)
lines!(ax21, Δt:Δt:tf, covariances_opt, label = L"\mathbf{u}_{opt}")
lines!(ax21, Δt:Δt:tf, covariances_high, label = L"10\cdot\mathbf{u}_{opt}")
lines!(ax21, Δt:Δt:tf, covariances_low, label = L"1/10\cdot\mathbf{u}_{opt}")
Legend(g21[0,1],[ax21],orientation = :horizontal,tellwidth=false, tellheight=true, padding = (0, 0, 0, 0))
ax22 = Axis(g22[1,1], xlabel=L"\log(u_x/u_{x0})", ylabel=L"\log(u_y/u_{y0})")
hm=heatmap!(ax22, exprange, exprange, log10.(vals_pred))
cb=Colorbar(g22[1,2],hm,label=L"\text{tr}\left(\bar{\bar{C}}_{\mathbf{x}}(t_f)\right) \; [\log_{10}]", tellwidth=true, alignmode=Mixed(right=0))
#colsize!(fig2.layout, 2, Auto(0.2))
save("Graphics/mega_fig.png", fig)
fig