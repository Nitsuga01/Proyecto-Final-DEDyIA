"
Lugar donde inferir parámetros del sistema a partir de trayectorias.
"

include("optical_trap_SDE_methods.jl")

## Preliminares

function gen_dataset(N,model,params)
    dataset=[]
    for _ in 1:N
        obs_trace = simulate(model, params)
        _, tr_obs_positions = get_retval(obs_trace)
        tr_obs_mat = SArray{Tuple{size(tr_obs_positions)...},typeof(tr_obs_positions[1])}(tr_obs_positions...)
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
σ = 0.005
feedback_params=SArray{Tuple{7},Float64}(ux,uy,vx,vy,errx,erry,σ)

ode_abstol = 1e-10
ode_reltol = 1e-6
ode_xo = thermal_values(n₀)
## Genero un dataset

N=1000
@time dataset = gen_dataset(N,SDEObs,(tpoints,feedback_params,solver_params))
print("Done.")

## Puedo inferir los parámetros que rigen sobre el ruido del momento estimado?

# Esta función genera dado un dataset, una función de error y algunos parametros extra una función de pérdida
# que se evalúa directamente sobre los parámetros a ajustar.
function define_problem(dataset, loss_function, to, tf, Δt, noopt_params; ode_xo=thermal_values(n₀), ode_abstol=1e-10, ode_reltol=1e-6)
    ux, uy, σ = noopt_params
    loss(opt_params::Vector{T},_) where T = begin
        p=SArray{Tuple{7},T}(ux,uy,opt_params...,σ)
        filter = trap_kalman_filter(to,tf,Δt,p,d0(p);ode_xo=ode_xo,ode_abstol=ode_abstol,ode_reltol=ode_reltol)
        loss_function(dataset,filter)
    end
    return loss
end

loss = define_problem(dataset,loglik_dataset,to,tf,Δt,(ux,uy,σ)) # función de pérdida de este problema
p_alt1(exponent)=[vx*10^exponent,vy,errx,erry] # genera parametros variando vx exponencialmente
p_alt2(exponent)=[vx,vy,errx*10^exponent,erry] # genera parametros variando errx exponencialmente

# Esta funcion sirve para, dado unos parámetros, integrar el valor de la loss function sobre cada trayectoria
# en un único valor y asignarle un error mediante un bootstrap no paramétrico de las trayectorias. 
function param_test_many_trajectories(params,M=10000)
    ll=loss(params,0)
    llmax = maximum(ll)
    pvec = ℯ.^(ll.-llmax)

    mll_shift = sum(pvec)

    mll_bootstrap = sum(rand(pvec,(length(pvec),M)),dims=1)

    # - log de la likelyhood media
    loss_val = -log(mll_shift) -llmax

    # - log de la likelyhood resampleada para bootstrapear el error.
    loss_bootstrap = -llmax .- log.(mll_bootstrap)

    loss_std=std(loss_bootstrap)
    return loss_val, loss_std
end

# barre en un conjunto de exponentes y calcula para cada uno el error.
exprange1=-2:0.1:2
l1=[param_test_many_trajectories(p_alt1(e)) for e in exprange1]
exprange2=-1:0.1:3
l2=[param_test_many_trajectories(p_alt2(e)) for e in exprange2]

## Gráfico de esta cosa.

fig1=Figure(size=(800,400))
#Label(fig1[0, 0:2], L"Modelado inverso basado en $N=1000$ trayectorias", tellheight=true, tellwidth=false)
Label(fig1[1, 0], L"-\log\left(\left\langle L\left(\mathbf{\theta}|\{\mathbf{x}^{OBS}_{i,t_o:t_f}\}_{i=1}^N\right)\right\rangle\right)", tellheight=false, rotation = pi/2)
ax11=Axis(fig1[1,1], xlabel = "e", ylabel = "Mean Loss", title=L"\tilde{v}_x\sim v_{x}\cdot 10^e")
ax12=Axis(fig1[1,2], xlabel = "e", ylabel = "Mean Loss", title=L"\tilde{\epsilon}_x\sim \epsilon_{x}\cdot 10^e")
lines!(ax11, exprange1, [l[1] for l in l1])
errorbars!(ax11, exprange1, [l[1] for l in l1], [l[2] for l in l1])
lines!(ax12, exprange2, [l[1] for l in l2])
errorbars!(ax12, exprange2, [l[1] for l in l2], [l[2] for l in l2])
#hidexdecorations!(ax11, ticks = false, grid = false)
hideydecorations!(ax11)
hideydecorations!(ax12)
Label(fig1[1, 1, TopLeft()], "a)",
        fontsize = 20,
        font = :bold,
        padding = (0, 5, 5, 0),
        halign = :right)
Label(fig1[1, 2, TopLeft()], "b)",
        fontsize = 20,
        font = :bold,
        padding = (0, 5, 5, 0),
        halign = :right)

save("Graphics/param_inference.pdf", fig1)
fig1