"""
Lugar para fittear trayectorias utilizando filtros de Kalman para ver que onda.
"""

include("optical_trap_SDE_methods.jl")

## parámetros

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

## Para analizar las trayectorias y el comportamiento del sistema en función de los parámetros
# y para comprobar que el modelo puede ajustarlo relativamente bien.

# Genero una trayectoria
obs_trace = simulate(model, (tpoints, feedback_params, solver_params))
state_vec, obs_positions = get_retval(obs_trace)
obs_mat = SArray{Tuple{size(obs_positions)...},typeof(obs_positions[1])}(obs_positions...)
obs_vec = eachrow(obs_mat)


# Ajusto la trayectoria con un filtro de kalman y extraigo el vector de estados inferido.
kf = trap_kalman_filter(to,tf,Δt,feedback_params,d0(feedback_params,true,obs_vec[1]);ode_abstol=1e-10,ode_reltol=1e-6,supersample=1)
sol = forward_trajectory(kf,fill([],length(obs_vec)),obs_vec,kf.p)
inf_vec=reinterpret(reshape, typeof(obs_positions[1]), sol.x)'
inf_vec_std=[[2*sqrt(sol.R[k][j,j]) for k in eachindex(sol.R)] for j in 1:6]
loglik_dataset([obs_vec],kf)

println("Done.")
## Genero una figura para esto

skip=1000

fig1=Figure(size=(1000,700))
g1l = fig1[1:2,1]
g1ur = fig1[1,2]
g1br = fig1[2,2]

ax11 = Axis(g1l[2,1], xlabel = "Tiempo [us]", ylabel = L"$\left<x\right>$ [nm]")
ax12 = Axis(g1l[3,1], xlabel = "Tiempo [us]", ylabel = L"$\left<y\right>$ [nm]")
lines!(ax11, tpoints, obs_positions[:,1], label = "Medido",color=Cycled(3),alpha=0.2)
lines!(ax12, tpoints, obs_positions[:,2], label = "Medido",color=Cycled(3),alpha=0.2)
lines!(ax11, tpoints, state_vec[:,1], label = "Verdadero",color=Cycled(1))
lines!(ax12, tpoints, state_vec[:,2], label = "Verdadero",color=Cycled(1))
lines!(ax11, tpoints, inf_vec[:,1], label = "Inferido",color=Cycled(2))
lines!(ax12, tpoints, inf_vec[:,2], label = "Inferido",color=Cycled(2))
band!(ax11, tpoints, inf_vec[:,1].-inf_vec_std[1], inf_vec[:,1].+inf_vec_std[1], label = "Inferido",alpha=0.7, color=Cycled(2))
band!(ax12, tpoints, inf_vec[:,2].-inf_vec_std[2], inf_vec[:,2].+inf_vec_std[2], label = "Inferido",alpha=0.7, color=Cycled(2))
leg=Legend(g1l[0,1], [ax11,ax12], merge=true, orientation = :horizontal, tellheight = true)
Label(g1l[1, 1], "Posición", font = :bold, tellwidth=false)
hidexdecorations!(ax11, grid = false, ticks=false)

ax21 = Axis(g1ur[1,1], xlabel = "Tiempo [us]", ylabel = L"$\left<p_x\right>$ [ev us/nm]")
ax22 = Axis(g1ur[1,2], xlabel = "Tiempo [us]", ylabel = L"$\left<p_y\right>$ [ev us/nm]")
lines!(ax21, tpoints, state_vec[:,3]*1e8, label = "Verdadero",color=Cycled(1))
lines!(ax22, tpoints, state_vec[:,4]*1e8, label = "Verdadero",color=Cycled(1))
lines!(ax21, tpoints, inf_vec[:,3]*1e8, label = "Inferido",color=Cycled(2))
lines!(ax22, tpoints, inf_vec[:,4]*1e8, label = "Inferido",color=Cycled(2))
band!(ax21, tpoints, inf_vec[:,3]*1e8.-inf_vec_std[3]*1e8, inf_vec[:,3]*1e8.+inf_vec_std[3]*1e8, label = "Inferido",alpha=0.7, color=Cycled(2))
band!(ax22, tpoints, inf_vec[:,4]*1e8.-inf_vec_std[4]*1e8, inf_vec[:,4]*1e8.+inf_vec_std[4]*1e8, label = "Inferido",alpha=0.7, color=Cycled(2))
Label(g1ur[0, 1:2], "Momento", font = :bold, tellwidth=false)
Label(g1ur[1, 1, Top()], halign = :left, L"\times 10^{-8}")
Label(g1ur[1, 2, Top()], halign = :left, L"\times 10^{-8}")
ylims!(ax21,1.1*1e8*minimum([minimum(inf_vec[skip:end,3].-inf_vec_std[3][skip:end]),minimum(state_vec[:,3])]),1.1*1e8*maximum([maximum(inf_vec[skip:end,3].+inf_vec_std[3][skip:end]),maximum(state_vec[:,3])]))
ylims!(ax22,1.1*1e8*minimum([minimum(inf_vec[skip:end,4].-inf_vec_std[4][skip:end]),minimum(state_vec[:,4])]),1.1*1e8*maximum([maximum(inf_vec[skip:end,4].+inf_vec_std[4][skip:end]),maximum(state_vec[:,4])]))
#hidexdecorations!(ax11, grid = false, ticks=false)


ax31 = Axis(g1br[1,1], xlabel = "Tiempo [us]", ylabel = L"$\left<q_x\right>$ [ev us/nm]")
ax32 = Axis(g1br[1,2], xlabel = "Tiempo [us]", ylabel = L"$\left<q_y\right>$ [ev us/nm]")
lines!(ax31, tpoints, state_vec[:,5]*1e8, label = "Verdadero",color=Cycled(1))
lines!(ax32, tpoints, state_vec[:,6]*1e8, label = "Verdadero",color=Cycled(1))
lines!(ax31, tpoints, inf_vec[:,5]*1e8, label = "Inferido",color=Cycled(2))
lines!(ax32, tpoints, inf_vec[:,6]*1e8, label = "Inferido",color=Cycled(2))
band!(ax31, tpoints, inf_vec[:,5]*1e8.-inf_vec_std[5]*1e8, inf_vec[:,5]*1e8.+inf_vec_std[5]*1e8, label = "Inferido",alpha=0.7, color=Cycled(2))
band!(ax32, tpoints, inf_vec[:,6]*1e8.-inf_vec_std[6]*1e8, inf_vec[:,6]*1e8.+inf_vec_std[6]*1e8, label = "Inferido",alpha=0.7, color=Cycled(2))
Label(g1br[0, 1:2], "Momento Estimado", font = :bold, tellwidth=false)
Label(g1br[1, 1, Top()], halign = :left, L"\times 10^{-8}")
Label(g1br[1, 2, Top()], halign = :left, L"\times 10^{-8}")
ylims!(ax31,1.1*1e8*minimum([minimum(inf_vec[skip:end,5].-inf_vec_std[5][skip:end]),minimum(state_vec[:,5])]),1.1*1e8*maximum([maximum(inf_vec[skip:end,5].+inf_vec_std[5][skip:end]),maximum(state_vec[:,5])]))
ylims!(ax32,1.1*1e8*minimum([minimum(inf_vec[skip:end,6].-inf_vec_std[6][skip:end]),minimum(state_vec[:,6])]),1.1*1e8*maximum([maximum(inf_vec[skip:end,6].+inf_vec_std[6][skip:end]),maximum(state_vec[:,6])]))
save("Graphics/state_inference.pdf", fig1)
fig1