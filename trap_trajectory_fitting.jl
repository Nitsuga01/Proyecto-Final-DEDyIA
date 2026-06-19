include("optical_trap_SDE_methods.jl")

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

## Para analizar las trayectorias y el comportamiento del sistema en función de los parámetros
# y para comprobar que el modelo puede ajustarlo relativamente bien.

obs_trace = simulate(model, (tpoints, feedback_params, solver_params))
state_vec, obs_positions = get_retval(obs_trace)
obs_mat = SArray{Tuple{size(obs_positions)...},typeof(obs_positions[1])}(obs_positions...)
obs_vec = eachrow(obs_mat)

# preparo la distribución inicial

kf = trap_kalman_filter(to,tf,Δt,feedback_params,d0(feedback_params,true,obs_vec[1]);ode_abstol=1e-10,ode_reltol=1e-6,supersample=1)
sol = forward_trajectory(kf,fill([],length(obs_vec)),obs_vec,kf.p)
inf_vec=reinterpret(reshape, typeof(obs_positions[1]), sol.x)'
loglik_dataset([obs_vec],kf)

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