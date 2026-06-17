"""
En este archivo se van a encontrar las funciones necesarias para simular una partícula atrapada en una trampa óptica
bajo medicion continua.
"""

## Funciones necesarias para las ecuaciones dinámicas del problema

# OBS, estoy usando la fórmula que dan en el paper de la trampa óptica en detalle. 
# La que dan en el principal está mal pq falta dividir por w_o ^ 2 por un tema de unidades.
# También parecen olvidarse de un 2 multiplicando.
function potential(t,x,y)
    km = I_G - 2 * sqrt(I_L * I_G) * cos(2 * Ω * t)
    kp = I_G + 2 * sqrt(I_L * I_G) * cos(2 * Ω * t)
    kxy = 4 * sqrt(I_L * I_G) * sin(2 * Ω * t)
    c = 2 * V_0 / w₀^2
    return c * (km * x^2 + kp * y^2 - kxy * x * y)
end

# Esta es la fuerza correspondiente al potencial de la trampa
function force(t,x,y)
    km = I_G - 2 * sqrt(I_L*I_G) * cos(2 * Ω * t)
    kp = I_G + 2 * sqrt(I_L*I_G) * cos(2 * Ω * t)
    kxy = 4 * sqrt(I_L*I_G)*sin(2 * Ω * t)
    c = 2 * V_0 / w₀^2
    return c * ( -2 * km * x + kxy * y), c * ( -2 * kp * y + kxy * x)
end

# Estas dan las derivadas de los valores medios de la posición y momento en la trampa 
function sde_drift!(du,u,p,t)
    # valor medio de posición y momento en la trampa
    x, y, px, py, qx, qy = u

    # fuerza de feedback y segundos momentos de posición y momento en la trampa´.
    ux, uy, vx, vy, errx, erry, σ = p

    # fuerza generada por el potencial de la trampa
    fx, fy = force(t,x,y) # obs que como la fuerza es lineal, la puedo evaluar directo en los vals medios.

    # derivadas de los valores medios de posición y momento en la trampa
    du[1] = px/m
    du[2] = py/m
    du[3] = fx+ux*qx
    du[4] = fy+uy*qy
    du[5] = vx*(px-qx)
    du[6] = vy*(py-qy)
end

# Estas dan los kicks estocásticos para los valores medios de la posición y momento en la trampa. Resulta ser ruido aditivo
# que depende de los valores de los segundos momentos a cada tiempo.
function sde_diffusion!(du, feedback_params, second_moments)
    # fuerza de backaction, medición y segundos momentos de posición y momento en la trampa
    ux, uy, vx, vy, errx, erry, σ = feedback_params
    C_xx, C_xy, C_yy, C_xpx, C_xpy, C_ypx, C_ypy = second_moments

    # constantes que gobiernan la magnitud del ruido
    ax = sqrt(2 * Λx * ηx)
    ay = sqrt(2 * Λy * ηy)

    # coeficientes que gobiernan la magnitud del ruido sobre los valores medios de posición y momento en la trampa
    du[1,1] = ax * C_xx
    du[2,1] = ax * C_xy
    du[3,1] = ax * C_xpx
    du[4,1] = ax * C_xpy
    du[1,2] = ay * C_xy
    du[2,2] = ay * C_yy
    du[3,2] = ay * C_ypx
    du[4,2] = ay * C_ypy
    du[5,3] = errx
    du[6,3] = erry
end

# Estas dan las derivadas de los segundos momentos de la posición y momento en la trampa.
function ode_drift!(du,u,p,t)
    # segundos momentos de posición y momento en la trampa
    C_xx, C_xy, C_yy, C_xpx, C_xpy, C_ypx, C_ypy, C_pxpx, C_pxpy, C_pypy = u

    # cantidades que gobiernan la magnitud del ruido, backaction, decorrelación, la fuerza en la trampa
    ax = 2 * Λx * ηx
    ay = 2 * Λy * ηy
    bx = 2 * Λx * ħ^2
    by = 2 * Λy * ħ^2
    c = 2 * V_0 / w₀^2

    # fuerza
    km = I_G - 2 * sqrt(I_L * I_G) * cos(2 * Ω * t)
    kp = I_G + 2 * sqrt(I_L * I_G) * cos(2 * Ω * t)
    kxy = 4 * sqrt(I_L * I_G) * sin(2 * Ω * t)
    C_dxVx = c * (2 * km * C_xx - kxy * C_xy)
    C_dxVy = c * (2 * km * C_xy - kxy * C_yy)
    C_dyVx = c * (2 * kp * C_xy - kxy * C_xx)
    C_dyVy = c * (2 * kp * C_yy - kxy * C_xy)
    C_dxVpx = c * (2 * km * C_xpx - kxy * C_ypx)
    C_dxVpy = c * (2 * km * C_xpy - kxy * C_ypy)
    C_dyVpx = c * (2 * kp * C_ypx - kxy * C_xpx)
    C_dyVpy = c * (2 * kp * C_ypy - kxy * C_xpy)

    # Deruvadas de los segundos momentos en la trampa, separadas según su contribución.
    # du :  L_Liouville          + L_Recoil +  L_Diffusion
    du[1] = 2/m * C_xpx                      - ax * C_xx^2 - ay * C_xy^2                # d/dt C_xx
    du[2] = 1/m * (C_xpy + C_ypx)            - ax * C_xx * C_xy - ay * C_yy * C_xy      # d/dt C_xy
    du[3] = 2/m * C_ypy                      - ax * C_xy^2 - ay * C_yy^2                # d/dt C_yy
    du[4] = 1/m * C_pxpx - C_dxVx            - ax * C_xx * C_xpx - ay * C_xy * C_ypx    # d/dt C_xpx
    du[5] = 1/m * C_pxpy - C_dyVx            - ax * C_xx * C_xpy - ay * C_xy * C_ypy    # d/dt C_xpy
    du[6] = 1/m * C_pxpy - C_dxVy            - ax * C_xy * C_xpx - ay * C_yy * C_ypx    # d/dt C_ypx
    du[7] = 1/m * C_pypy - C_dyVy            - ax * C_xy * C_xpy - ay * C_yy * C_ypy    # d/dt C_ypy
    du[8] = - 2 * C_dxVpx        + bx        - ax * C_xpx^2 - ay * C_ypx^2              # d/dt C_pxpx
    du[9] = - (C_dxVpy + C_dyVpx)            - ax * C_xpx * C_xpy - ay * C_ypx * C_ypy  # d/dt C_pxpy
    du[10] = - 2 * C_dyVpy       + by        - ax * C_xpy^2 - ay * C_ypy^2              # d/dt C_pypy
    
end

# parámetros iniciales térmicos para el sistema
function thermal_values(n)
    C_xx = x_zpf^2 * (2 * n + 1) # variance of position in x for thermal state with mean occupation n
    C_yy = y_zpf^2 * (2 * n + 1) # variance of position in y for thermal state with mean occupation n
    C_xy = 0
    C_xpx = 0
    C_xpy = 0
    C_ypx = 0
    C_ypy = 0
    C_pxpx = ħ^2 / (4 * x_zpf^2) * (2 * n + 1) # variance of position in px for thermal state with mean occupation n
    C_pxpy = 0
    C_pypy = ħ^2 / (4 * y_zpf^2) * (2 * n + 1) # variance of position in py for thermal state with mean occupation n
    return [C_xx, C_xy, C_yy, C_xpx, C_xpy, C_ypx, C_ypy, C_pxpx, C_pxpy, C_pypy]
end