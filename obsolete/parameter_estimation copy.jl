"""
Acá irán la manera de estimar la función de costo, y sus derivadas basada en diferentes métodos.

Turing provee del method of moments.
Gen provee de opciones más avanzadas.
"""



@model function fitSDE(data)
    # Parameters with prior distributions.
    ux ~ truncated(Normal(-0.1, 0.01); lower=-0.5, upper=0.0)
    uy ~ truncated(Normal(-0.1, 0.01); lower=-0.5, upper=0.0)
    vx ~ truncated(Normal(1, 0.25); lower=0.0, upper=3.0)
    vy ~ truncated(Normal(1, 0.25); lower=0.0,upper=3.0)
    errx ~ Normal(0, 0.01)
    erry ~ Normal(0, 0.01)
    errm ~ InverseGamma(2, 0.001)

    # Simulate the SDE.
    tvals=0.0:0.1:10.0
    predicted = trap_sim((tvals[1],tvals[end]),nothing,nothing,[ux,uy,vx,vy,errx,erry],saveat=tvals)[1].u

    # Early exit if simulation could not be computed successfully.
    if predicted.retcode !== :Success
        Turing.@addlogprob! -Inf
        return nothing
    end

    # Observations.
    for i in eachindex(tvals)
        data[i] ~ MvNormal(predicted[i], errm^2 * I)
    end

    return nothing
end;

## Prueba de esto.

ux=-0.05
uy=-0.05
vx=1
vy=1
errx=0.0
erry=0.0

#sdedata = ensemble_trap_sim(10,(0.0,10.0),nothing,nothing,(ux,uy,vx,vy,errx,erry),saveat=0.0:0.1:10.0)[1]
sdedata = trap_sim((0.0,10.0),nothing,nothing,(ux,uy,vx,vy,errx,erry),saveat=0.0:0.1:10.0)[1]

print("")
##

model_sde = fitSDE(sdedata.u)

print("")

##

#setadbackend(:forwarddiff)
chain_sde = sample(
    model_sde,
    NUTS(0.25),
    5000;
    init_params=[0.001, -0.05, -0.05, 1, 1, 0.0, 0.0],
    progress=false,
)

print("")
##
plot(chain_sde)