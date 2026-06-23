#%%
import os
os.environ["OMP_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
os.environ["OPENBLAS_NUM_THREADS"] = "12"

#%%

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from qutip import *

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
# TIEMPO
# ==========================================================

t0 = 0.0
tf = 100.0
Nt = 300

t_list = np.linspace(t0, tf, Nt)
dt = t_list[1] - t_list[0]

# ==========================================================
# SIMULADOR
# ==========================================================

def generar_simulacion(Gamma_x, Gamma_y, Omega):

    # ------------------------------------------------------
    # Ruido OU distinto en cada simulación
    # ------------------------------------------------------

    tau_c = (0.1*np.random.randn() + 1) * 1.0

    sigma_OU = (0.1*np.random.randn() + 1) * 0.3

    xi = np.zeros(Nt)

    for i in range(Nt - 1):
        xi[i+1] = xi[i] - (dt/tau_c)*xi[i] + sigma_OU*np.sqrt(2*dt/tau_c) * np.random.randn()

    Omega_eff = Omega + xi

    phi = np.zeros(Nt)

    for i in range(Nt-1):
        phi[i+1] = phi[i] + Omega_eff[i]*dt

    # ------------------------------------------------------
    # Interpolación
    # ------------------------------------------------------

    def phi_t(t):
        return np.interp(t, t_list, phi)

    # ------------------------------------------------------
    # Hamiltoniano
    # ------------------------------------------------------

    K = (px**2 + py**2)/(2*m)

    Axx = V0*x**2/hbar
    Ayy = V0*y**2/hbar

    Axy = V0*x*y/hbar

    def k_x(t, args):
        return IG + 2*np.sqrt(IG*IL)*np.cos(2*phi_t(t))

    def k_y(t, args):
        return IG - 2*np.sqrt(IG*IL)*np.cos(2*phi_t(t))

    def k_xy(t, args):
        return -4*np.sqrt(IG*IL)*np.sin(2*phi_t(t))

    H = [K/hbar, [Axx, k_x], [Ayy, k_y], [Axy, k_xy]]                           # Hamiltoniano
    c_ops = [np.sqrt(Gamma_x) * x/x_zpf, np.sqrt(Gamma_y) * y/y_zpf]
    obs = [x*x, y*y, px*px, py*py]

    sol = mesolve(H, rho0, t_list, c_ops, obs)

    x2 = np.real(sol.expect[0])
    y2 = np.real(sol.expect[1])

    px2 = np.real(sol.expect[2])
    py2 = np.real(sol.expect[3])

    curvas = np.concatenate([x2, y2, px2, py2])

    return curvas, x2, y2, px2, py2

# ==========================================================
# GENERAR DATASET
# ==========================================================

Nsim = 10

X = []
Y = []
X2_EXP = []
Y2_EXP = []
PX2_EXP = []
PY2_EXP = []

for n in range(Nsim):

    Gamma_x_sim = (0.1*np.random.randn() + 1) * Gamma_x

    Gamma_y_sim = (0.1*np.random.randn() + 1) * Gamma_y

    Omega_sim = (0.1*np.random.randn() + 1) * Omega

    curvas, x2_exp, y2_exp, px2_exp, py2_exp = generar_simulacion(Gamma_x_sim, Gamma_y_sim, Omega_sim)

    X.append(curvas)
    Y.append([Gamma_x_sim,Gamma_y_sim,Omega_sim])
    X2_EXP.append(x2_exp)
    Y2_EXP.append(y2_exp)
    PX2_EXP.append(px2_exp)
    PY2_EXP.append(py2_exp)

    print(n+1,"/",Nsim)

X = np.array(X,dtype=np.float32)
Y = np.array(Y,dtype=np.float32)
X2_EXP = np.array(X2_EXP,dtype=np.float32)
Y2_EXP = np.array(Y2_EXP,dtype=np.float32)
PX2_EXP = np.array(PX2_EXP,dtype=np.float32)
PY2_EXP = np.array(PY2_EXP,dtype=np.float32)

X_mean = X.mean(axis=0)
X_std  = X.std(axis=0)

X_norm = (X - X_mean)/(X_std + 1e-12)

Y_mean = Y.mean(axis=0)
Y_std  = Y.std(axis=0)

Y_norm = (Y - Y_mean)/Y_std

print(np.min(X_std))

# ==========================================================
# DATASET PYTORCH
# ==========================================================

class QuantumDataset(Dataset):

    def __init__(self,X,Y):
        self.X = torch.tensor(X)
        self.Y = torch.tensor(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self,idx):
        return (self.X[idx], self.Y[idx])

dataset = QuantumDataset(X_norm,Y_norm)

loader = DataLoader(dataset, batch_size=16, shuffle=True)

# ==========================================================
# RED NEURONAL
# ==========================================================

input_dim = X.shape[1]

model = nn.Sequential(

    nn.Linear(input_dim, 512),

    nn.ReLU(),

    nn.Linear(512, 256),

    nn.ReLU(),

    nn.Linear(256, 128),

    nn.ReLU(),

    nn.Linear(128, 3)
)

# ==========================================================
# ENTRENAMIENTO
# ==========================================================

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

loss_fn = nn.MSELoss()

epochs = 200

for epoch in range(epochs):

    loss_epoch = 0

    for xb,yb in loader:

        pred = model(xb)

        loss = loss_fn(pred, yb)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        loss_epoch += loss.item()

    print(epoch, loss_epoch)

# ==========================================================
# TEST
# ==========================================================

curva, x2_exp, y2_exp, px2_exp, py2_exp = generar_simulacion(Gamma_x, Gamma_y, Omega)

entrada0 = ((curva - X_mean)/(X_std + 1e-12))

entrada = torch.tensor(entrada0, dtype=torch.float32).unsqueeze(0)

pred_norm = model(entrada).detach().numpy()[0]

pred = pred_norm*Y_std + Y_mean

print("\nREAL")

print(Gamma_x, Gamma_y,Omega)

print("\nPREDICHO")

print(pred)


#%% Acá calculamos los errores asociados a los parámetros obtenidos

# ==========================================================
# BOOTSTRAP DE DATOS
# ==========================================================

print("\nComenzando bootstrap...")

# ----------------------------------------------------------
# Error experimental asumido
# ----------------------------------------------------------

Nboot = 10000
ruido = 1e-4
predicciones = []

# ----------------------------------------------------------
# Bootstrap
# ----------------------------------------------------------
for i in range(Nsim):

    for k in range(Nboot):

        x2_boot = X2_EXP[i] * (ruido*np.random.randn() + 1)
        y2_boot = Y2_EXP[i] * (ruido*np.random.randn() + 1)
        px2_boot = PX2_EXP[i] * (ruido*np.random.randn() + 1)
        py2_boot = PY2_EXP[i] * (ruido*np.random.randn() + 1)

        curva_boot = np.concatenate([x2_boot, y2_boot, px2_boot, py2_boot])

        entrada0 = (curva_boot - X_mean)/(X_std + 1e-12)

        entrada = torch.tensor(entrada0, dtype=torch.float32).unsqueeze(0)

        pred_norm = model(entrada).detach().numpy()[0]

        pred = pred_norm*Y_std + Y_mean

        predicciones.append(pred)

        if (k+1)%100 == 0:
            print(k+1,"/",Nboot)

predicciones = np.array(predicciones)

print("Bootstrap finalizado")

Gamma_x_boot = predicciones[:,0]

Gamma_y_boot = predicciones[:,1]

Omega_boot = predicciones[:,2]

Gamma_x_mean = np.mean(Gamma_x_boot)
Gamma_y_mean = np.mean(Gamma_y_boot)
Omega_mean   = np.mean(Omega_boot)

Gamma_x_std = np.std(Gamma_x_boot)
Gamma_y_std = np.std(Gamma_y_boot)
Omega_std   = np.std(Omega_boot)

print("\n===================================")
print("RESULTADOS BOOTSTRAP")
print("===================================\n")

print(f"Gamma_x = " f"{Gamma_x_mean:.6e}" f" ± " f"{Gamma_x_std:.6e}")

print(f"Gamma_y = " f"{Gamma_y_mean:.6e}" f" ± " f"{Gamma_y_std:.6e}")

print(f"Omega = " f"{Omega_mean:.6f}" f" ± " f"{Omega_std:.6f}")

# ==========================================================
# INTERVALOS DE CONFIANZA
# ==========================================================

Gamma_x_IC = np.percentile(Gamma_x_boot, [2.5,97.5])

Gamma_y_IC = np.percentile(Gamma_y_boot, [2.5,97.5])

Omega_IC = np.percentile(Omega_boot, [2.5,97.5])

print("\nIC 95%")

print("Gamma_x =",Gamma_x_IC)

print("Gamma_y =",Gamma_y_IC)

print("Omega   =",Omega_IC)


# %%
