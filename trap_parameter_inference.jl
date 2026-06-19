"
Lugar donde hacer pruebas de los solvers numéricos para chequear que todo funciona correctamente.
"

include("optical_trap_SDE_methods.jl")

function gen_dataset(N,model,params)
    dataset=[]
    for _ in 1:N
        obs_trace = simulate(model, params)
        _, tr_obs_positions = get_retval(obs_trace)
        tr_obs_mat = SArray{Tuple{size(obs_positions)...},typeof(obs_positions[1])}(tr_obs_positions...)
        tr_obs_vec = eachrow(tr_obs_mat)
        push!(dataset,tr_obs_vec)
    end
    dataset
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
## Genero un dataset

N=1000
@time dataset = gen_dataset(N,SDEObs,(tpoints,feedback_params,solver_params))
print("Done.")

## Puedo inferir los parámetros que rigen sobre el ruido del momento estimado?

function define_problem(dataset, loss_function, to, tf, Δt, noopt_params; ode_xo=thermal_values(n₀), ode_abstol=1e-10, ode_reltol=1e-6)
    ux, uy, σ = noopt_params
    loss(opt_params::Vector{T},_) where T = begin
        p=SArray{Tuple{7},T}(ux,uy,opt_params...,σ)
        filter = trap_kalman_filter(to,tf,Δt,p,d0(p);ode_xo=ode_xo,ode_abstol=ode_abstol,ode_reltol=ode_reltol)
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

## Repito con otros parámetros para el ruido del momento estimado.

errx_alt = 1e-9*1e3
erry_alt = 1e-9*1e3

## Genero un dataset

N=1000
dataset = []
for _ in 1:N
    obs_trace = simulate(model, (tpoints, (ux,uy,vx,vy,errx_alt,erry_alt,σ), solver_params))
    tr_state_vec, tr_obs_positions = get_retval(obs_trace)
    tr_obs_vec = [tr_obs_positions[k,:] for k in 1:size(tr_obs_positions)[1]]
    push!(dataset,tr_obs_vec)
end

## Puedo inferir los parámetros que rigen sobre el ruido del momento estimado?

function define_problem(dataset, loss_function, to, tf, Δt, noopt_params; ode_xo=thermal_values(n₀), ode_abstol=1e-10, ode_reltol=1e-6)
    ux, uy, σ = noopt_params
    loss(opt_params::Vector{T},_) where T = begin
        p=SArray{Tuple{7},T}(ux,uy,opt_params...,σ)
        filter = trap_kalman_filter(to,tf,Δt,p,d0(p);ode_xo=ode_xo,ode_abstol=ode_abstol,ode_reltol=ode_reltol)
        loss_function(dataset,filter)
    end
    return loss
end

loss = define_problem(dataset,loglik_dataset,to,tf,Δt,(ux,uy,σ))

p_opt=[vx,vy,errx_alt,erry_alt]
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
exprange=-4:0.1:1

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
save("Graphics/param_inference_noisy.png", fig1)