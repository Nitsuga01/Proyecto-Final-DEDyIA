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
