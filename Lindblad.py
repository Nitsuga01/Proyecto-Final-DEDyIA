#%%
import os
os.environ["OMP_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
os.environ["OPENBLAS_NUM_THREADS"] = "12"

#%% Librerías
import numpy as np
from qutip import *
import matplotlib.pyplot as plt

#%% Unidades de ev, us y nm
# ==========================================================
# Constantes y parámetros
# ==========================================================

hbar = 6.582119569e-10          # h barra
m = 7.11532e-6                  # masa de la partícula
IG = 0.3                        # Intensidades
IL = 1.0 - IG
Omega_x = 150e-3 * 2*np.pi      # Frecuencias del haz Gaussiano
Omega_y = 150e-3 * 2*np.pi
Omega_0 = 0.54120755            # Frecuencia crítica de la trampa
Omega = 5 * Omega_0             # Frecuencia de giro de la trampa
lamb = 1.55e3                   # Longitud de onda del haz Gaussiano
c = 2.99792458e11               # Velocidad de la luz
P = 4.369055e11                 # Potencia haz Gaussiano
NA = 0.6                        # Apertura numérica del haz Gaussiano
epsilon_0 = 0.05526349406       # Permitividad dieléctrica
n = 1.45                        # Índice de refracción
R = 50                          # Radio de la nanopartícula

# ==========================================================
# Parámetros derivados
# ==========================================================

k = 2*np.pi/lamb                                            # Número de onda del haz Gaussiano
w0 = lamb/(np.pi*NA)                                        # Cintura del haz Gaussiano
alpha = 4*np.pi*epsilon_0*R**3*(n**2 - 1)/(n**2 + 2)        # Polarizabilidad del haz Gaussiano
V0 = 2*np.real(alpha)*P/(c*np.pi*w0**4*epsilon_0)           # Constante del potencial V(x,y,t)
C_vec = np.array([1/5, 2/5])
Gamma_x = (2*IG*P*k**5*np.abs(alpha)**2)/(15*m*c*Omega_x*w0**2*np.pi**2*epsilon_0**2) * C_vec[0]    #Coeficientes de disipación
Gamma_y = (2*IG*P*k**5*np.abs(alpha)**2)/(15*m*c*Omega_y*w0**2*np.pi**2*epsilon_0**2) * C_vec[1]

x_zpf = np.sqrt(hbar/(2*m*Omega_x))
px_zpf = np.sqrt(hbar*m*Omega_x/2)
y_zpf = np.sqrt(hbar/(2*m*Omega_y))
py_zpf = np.sqrt(hbar*m*Omega_y/2)

# ==========================================================
# Espacio de Hilbert
# ==========================================================

Nx = 30
Ny = 30

Ix = qeye(Nx)
Iy = qeye(Ny)

x = x_zpf * tensor(position(Nx), Iy)
px = px_zpf * tensor(momentum(Nx), Iy)
y = y_zpf * tensor(Ix, position(Ny))
py = py_zpf * tensor(Ix, momentum(Ny))

# ==========================================================
# Estado inicial
# ==========================================================

nmedio_x = 0.8
nmedio_y = 0.8

rho_x = thermal_dm(Nx, nmedio_x)
rho_y = thermal_dm(Ny, nmedio_y)
rho0 = tensor(rho_x, rho_y)

# ==========================================================
# Tiempo
# ==========================================================

t0 = 0.0
tf = 100.0
Nt = 2500

t_list = np.linspace(t0, tf, Nt)
dt = t_list[1] - t_list[0]

# ==========================================================
# Ruido Ornstein-Uhlenbeck
# ==========================================================


tau_c = 1.0
sigma_OU = 0.3

''' Si quiero sin ruido
tau_c = 1000000000
sigma_OU = 0
'''

np.random.seed(1234)

xi = np.zeros(Nt)

for i in range(Nt - 1):
    xi[i+1] = xi[i] - (dt/tau_c)*xi[i] + sigma_OU*np.sqrt(2*dt/tau_c) * np.random.randn()

# Frecuencia efectiva

Omega_eff = Omega + xi
phi = np.zeros(Nt)

for i in range(Nt-1):
    phi[i+1] = phi[i] + Omega_eff[i]*dt

def phi_t(t):
    return np.interp(t, t_list, phi)

# ==========================================================
# Operadores Hamiltonianos
# ==========================================================

K = (px**2 + py**2)/(2*m)

Axx = V0*x**2/hbar
Ayy = V0*y**2/hbar

Axy = V0*x*y/hbar

# ==========================================================
# Coeficientes temporales
# ==========================================================

def k_x(t, args):
    return IG + 2*np.sqrt(IG*IL)*np.cos(2*phi_t(t))

def k_y(t, args):
    return IG - 2*np.sqrt(IG*IL)*np.cos(2*phi_t(t))

def k_xy(t, args):
    return -4*np.sqrt(IG*IL)*np.sin(2*phi_t(t))

# ==========================================================
# Hamiltoniano, observables y evolución
# ==========================================================

H = [K/hbar, [Axx, k_x], [Ayy, k_y], [Axy, k_xy]]                           # Hamiltoniano
c_ops = [np.sqrt(Gamma_x) * x/x_zpf, np.sqrt(Gamma_y) * y/y_zpf]
obs = [x, y, px, py, x*x, y*y, px*px, py*py]

print("Comenzando simulación...")

evolucion = mesolve(H, rho0, t_list, c_ops, obs)

print("Simulación finalizada")

# ==========================================================
# Resultados
# ==========================================================

x_mean = evolucion.expect[0]
y_mean = evolucion.expect[1]

px_mean = evolucion.expect[2]
py_mean = evolucion.expect[3]

x2_mean = evolucion.expect[4]
y2_mean = evolucion.expect[5]

px2_mean = evolucion.expect[6]
py2_mean = evolucion.expect[7]

sigma_x = np.sqrt(np.real(x2_mean - x_mean**2))
sigma_y = np.sqrt(np.real(y2_mean - y_mean**2))
sigma_px = np.sqrt(np.real(px2_mean - px_mean**2))
sigma_py = np.sqrt(np.real(py2_mean - py_mean**2))

# ==========================================================
# Guardado en TXT
# ==========================================================

datos = np.column_stack(
    (t_list, x_mean, y_mean, px_mean, py_mean, x2_mean, y2_mean, px2_mean, py2_mean, sigma_x, sigma_y, sigma_px, sigma_py, xi, Omega_eff, phi)
)

directorio = r"/media/maxi_murgia/Datos/uwu/Materias UBA/Datos, Eq dif e IA"
os.makedirs(directorio, exist_ok=True)

#archivo = os.path.join(directorio, "datos_sin_OU.txt")
archivo = os.path.join(directorio, "datos_con_OU.txt")

np.savetxt(
    archivo,
    datos,
    header=
    "t x_mean y_mean px_mean py_mean "
    "x2_mean y2_mean px2_mean py2_mean "
    "sigma_x sigma_y sigma_px sigma_py xi Omega_eff phi",
    fmt="%.12e"
)

print("Archivo datos_OU.txt guardado")

# ==========================================================
# Gráfico de <x²> y <y²>
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(t_list, x2_mean, lw=2, label=r'$\langle x^2 \rangle$')
plt.plot(t_list, y2_mean, lw=2, label=r'$\langle y^2 \rangle$')
plt.xlabel(r'Tiempo [$\mu s$]')
plt.ylabel(r'nm$^2$')

plt.title("Evolución de los segundos momentos")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))

plt.plot(x2_mean, y2_mean, lw=2)
plt.xlabel(r'$\langle x^2 \rangle (nm^2)$')
plt.ylabel(r'$\langle y^2 \rangle (nm^2)$')

plt.title("Espacio de fases de segundos momentos en posición")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ==========================================================
# Gráfico de <px²> y <py²>
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(t_list, px2_mean, lw=2, label=r'$\langle px^2 \rangle$')
plt.plot(t_list, py2_mean, lw=2, label=r'$\langle py^2 \rangle$')
plt.xlabel(r'Tiempo [$\mu s$]')
plt.ylabel(r'eV.us/nm')

plt.title("Evolución de los segundos momentos")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))

plt.plot(px2_mean, py2_mean, lw=2)
plt.xlabel(r'$\langle px^2 \rangle (eV.us/nm)$')
plt.ylabel(r'$\langle py^2 \rangle (eV.us/nm)$')

plt.title("Espacio de fases de segundos momentos en impulso lineal")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ==========================================================
# Gráfico del ruido OU
# ==========================================================

plt.figure(figsize=(8,4))

plt.plot(t_list, xi, lw=1.5)

plt.xlabel(r'Tiempo [$\mu s$]')
plt.ylabel(r'$\xi(t)$')

plt.title('Proceso Ornstein-Uhlenbeck')
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
