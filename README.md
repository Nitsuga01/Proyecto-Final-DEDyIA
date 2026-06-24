# Proyecto-Final-DEDyIA

En este repositorio se guarda todo el código utilizado para simular el comportamiento de la trampa armónica considerada con y sin monitoreo.

## Trampa armónica con ruido estocástico en la frecuencia

En archivos .py se encuentran los scripts utilizados para la simulación de las ecuaciones de Lindblad y de la red neuronal utilizada para inferir parámetros de la misma. La estructura de los archivos es la siguiente:

- Lindblad.py contiene las funciones utilizadas para simular la dinámica del sistema generando datasets.
- NN Gammas y Omega.py contiene el script utilizado para resolver el problema inverso y obtener los parametros Gammas y Omega pero no los del ruido aleatorio utilizando una red neuronal.
- NN extendida.py contiene el script utilizado para resolver el problema inverso y obtener los parametros Gammas, Omega y del ruido aleatorio utilizando una red neuronal.

## Trampa armónica con monitoreo y feedback

En archivos .jl se encuentran los scripts utilizados para simular las ecuaciones diferenciales estocásticas y realizar problemas de optimización sobre la misma mediante integración numérica y filtros de Kalman. La estructura de los archivos es la siguiente:

- trap_v3.jl incluye las funciones que rigen la evolución (parte determinista determinista y estocástica) de la partícula en la trampa.
- gen_solvers.jl incluye una funcion que utiliza la librería Gen para simular la evolución del sistema según un método de Euler Mauryama modificado.
- Kalman_solvers.jl contine funciones útiles para emplear filtros de Kalman sobre el sistema y para calcular la verosimilitud de sets de parámetros dados sets de trayectorias.
- optical_trap_SDE_methods.jl contiene las constantes de la trampa e incluye a los archivos previos dentro de sí.
- trap_orders_magnitude printea estimaciones de la influencia de distintos términos deterministas y estocásticos para los solvers numéricos.
- trap_trajectory_fitting.jl se utiliza para ajustar trayectorias individuales del sistema con filtros de Kalman.
- trap_parameter_inference.jl contiene el script utilizado para evaluar la verosimilitud de parámetros que rigen la estimación del momento dado un conjunto de trayectorias.
- trap_parameter_optimization.jl contiene el script utilziado para obtener parámetros que minimizen la covarianza del sistema a un tiempo avanzado.
- trap_extras.jl contiene el código utilizado para realizar el gráfico ubicado en el apéndice de la evolución de las distintas componentes de la matriz de covarianza.