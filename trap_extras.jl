"
Lugar donde hacer pruebas de los solvers numéricos para chequear que todo funciona correctamente.
"

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
σ = 0.005
feedback_params=SArray{Tuple{7},Float64}(ux,uy,vx,vy,errx,erry,σ)

ode_abstol = 1e-10
ode_reltol = 1e-6
ode_xo = thermal_values(n₀)

##

prob_ode = ODEProblem(ode_drift!, ode_xo, (to,tf))
sol_ode = solve(prob_ode, abstol=ode_abstol, reltol=ode_reltol, saveat=to:Δt:tf)

##

fig = Figure(size=(1000,2000))
Label(fig[0,0:1],"Segundos momentos de la distribución")
labels = [L"C_{xx}",L"C_{xy}",L"C_{yy}",L"C_{xp_x}",L"C_{xp_y}",L"C_{yp_x}",L"C_{yp_y}",L"C_{p_xp_x}",L"C_{p_xp_y}",L"C_{p_yp_y}"]
for k in eachindex(ode_xo)
    ax = Axis(fig[(k-1)%5+1,(k-1)÷5])
    lines!(ax,to:Δt:tf,[sol_ode[j][k] for j in eachindex(to:Δt:tf)], label=labels[k])
    axislegend(ax)
    if (k-1)%5+1 != 5
        hidexdecorations!(ax,grid=false,ticks=false)
    end
end
fig
save("Graphics/second_moments.pdf", fig)
