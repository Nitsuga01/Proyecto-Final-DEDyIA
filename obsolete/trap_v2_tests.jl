"
Lugar donde hacer pruebas de los solvers numéricos para chequear que todo funciona correctamente.
"

using CairoMakie
include("trap_v2.jl")

## Cosas solo con la trampa

ode_xo = thermal_values(n₀)
sde_xo = sqrt.(ode_xo[[1, 3, 8, 10]])
tspan = (0.0, 100.0)
ux=-0.05
uy=-0.05
err=0.0

sol_sde, sol_ode = trap_sim(sde_xo, ode_xo, tspan, (ux,uy,err), 1e-14, 1e-6, 1e-7, 1e-5) 

fig1=Figure(figsize=(2000,800),fontsize=6)
T=tspan[1]:0.1:tspan[2]
ax11 = Axis(fig1[1,1], xlabel = "t (us)", ylabel = "C_xx (nm^2)")
ax12 = Axis(fig1[1,2], xlabel = "t (us)", ylabel = "C_xy (nm^2)")
ax13 = Axis(fig1[1,3], xlabel = "t (us)", ylabel = "C_yy (nm^2)")
ax14 = Axis(fig1[2,1], xlabel = "t (us)", ylabel = "C_pxpx ((ev us / nm)^2)")
ax15 = Axis(fig1[2,2], xlabel = "t (us)", ylabel = "C_pxpy ((ev us / nm)^2)")
ax16 = Axis(fig1[2,3], xlabel = "t (us)", ylabel = "C_pypy ((ev us / nm)^2)")
ax17 = Axis(fig1[1,4], xlabel = "t (us)", ylabel = "C_xpx (ev us)")
ax18 = Axis(fig1[1,5], xlabel = "t (us)", ylabel = "C_xpy (ev us)")
ax19 = Axis(fig1[2,4], xlabel = "t (us)", ylabel = "C_ypx (ev us)")
ax110 = Axis(fig1[2,5], xlabel = "t (us)", ylabel = "C_ypy (ev us)")
lines!(ax11, T, sol_ode(T)[1,:], label = "C_xx")
lines!(ax12, T, sol_ode(T)[2,:], label = "C_xy")
lines!(ax13, T, sol_ode(T)[3,:], label = "C_yy")
lines!(ax17, T, sol_ode(T)[4,:], label = "C_xpx")
lines!(ax18, T, sol_ode(T)[5,:], label = "C_xpy")
lines!(ax19, T, sol_ode(T)[6,:], label = "C_ypx")
lines!(ax110, T, sol_ode(T)[7,:], label = "C_ypy")
lines!(ax14, T, sol_ode(T)[8,:], label = "C_pxpx")
lines!(ax15, T, sol_ode(T)[9,:], label = "C_pxpy")
lines!(ax16, T, sol_ode(T)[10,:], label = "C_pypy")
fig1

fig3=Figure()
ax31 = Axis(fig3[1,1], xlabel = "t (us)", ylabel = "C_xy (nm^2)")
lines!(ax31, T, sol_ode(T)[2,:])
fig3

fig4=Figure()
ax41 = Axis(fig4[1,1], xlabel = "C_yy (nm^2)", ylabel = "C_xx (nm^2)")
lines!(ax41, sol_ode(T)[1,:], sol_ode(T)[3,:])
fig4

fig5=Figure()
ax51 = Axis(fig5[1,1], xlabel = "C_pxpx ((ev us / nm)^2)", ylabel = "C_pypy ((ev us / nm)^2)")
lines!(ax51, sol_ode(T)[8,:], sol_ode(T)[10,:])
fig5

fig2=Figure()
ax21 = Axis(fig2[1,1], xlabel = "t (us)", ylabel = "Pos (nm)")
lines!(ax21, sol_sde.t, sol_sde[1,:], label = "x")
lines!(ax21, sol_sde.t, sol_sde[2,:], label = "y")
ax22 = Axis(fig2[1,2], xlabel = "x (nm)", ylabel = "y (nm)")
lines!(ax22, sol_sde[1,:], sol_sde[2,:])
ax23 = Axis(fig2[2,1], xlabel = "t (us)", ylabel = "Momento (ev us / nm)")
lines!(ax23, sol_sde.t, sol_sde[3,:], label = "x")
lines!(ax23, sol_sde.t, sol_sde[4,:], label = "y")
ax24 = Axis(fig2[2,2], xlabel = "px (ev us / nm)", ylabel = "py (ev us / nm)")
lines!(ax24, sol_sde[3,:], sol_sde[4,:])
fig2
