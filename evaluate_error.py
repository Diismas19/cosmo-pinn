import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Importando nossa classe do solver e as constantes
from solver import PINNSolver
from model import Z0, Z_MIN, Z_MAX, B_MIN, B_MAX

def fR_odes(z, u, b, Omega_m0_L):
    """
    Define o sistema de EDOs para o modelo f(R).
    z: variável independente (redshift)
    u: vetor de estado [x, y, v, Omega, r]
    b, Omega_m0_L: parâmetros do modelo
    """
    x, y, v, Omega, r = u
    epsilon = 1e-9 # Evita divisão por zero

    gamma_term = ((r + b) * ((r + b)**2 - 2 * b)) / (4 * b * r + epsilon)

    # Derivadas (du/dz)
    dxdz = (1 / (1 + z)) * (-Omega - 2*v + x + 4*y + x*v + x**2)
    dydz = (-1 / (1 + z)) * (v*x*gamma_term - x*y + 4*y - 2*y*v)
    dvdz = (-v / (1 + z)) * (x*gamma_term + 4 - 2*v)
    dOmegadz = (Omega / (1 + z)) * (-1 + 2*v + x)
    drdz = (-r * gamma_term * x) / (1 + z)

    return [dxdz, dydz, dvdz, dOmegadz, drdz]


device = torch.device("cpu") # Para avaliação, CPU é suficiente e mais simples
print(f"Usando o dispositivo: {device}")

pinn_solver = PINNSolver()
try:
    # Carregamos os modelos, garantindo que eles operem no dispositivo correto
    pinn_solver.model_x.load_state_dict(torch.load("model_x.pth", map_location=device))
    pinn_solver.model_y.load_state_dict(torch.load("model_y.pth", map_location=device))
    pinn_solver.model_v.load_state_dict(torch.load("model_v.pth", map_location=device))
    pinn_solver.model_Omega.load_state_dict(torch.load("model_Omega.pth", map_location=device))
    pinn_solver.model_r_prime.load_state_dict(torch.load("model_r_prime.pth", map_location=device))
except FileNotFoundError:
    print("Erro: Arquivos de modelo treinado (.pth) não encontrados.")
    exit()

for model in pinn_solver.models:
    model.to(device)
    model.eval()
print("Modelos PINN treinados carregados com sucesso!")

# Parâmetros para o nosso gráfico de erro
Omega_fixed = 0.3
z_grid = np.linspace(0.1, 3.0, 50)  # Eixo X do gráfico
b_grid = np.linspace(0.01, 2.0, 50) # Eixo Y do gráfico
Z_MESH, B_MESH = np.meshgrid(z_grid, b_grid)
ERROR_MESH = np.zeros_like(Z_MESH)

print("Iniciando cálculo do erro. Isso pode levar alguns minutos...")

# Itera sobre cada ponto (b, z) da nossa grade
for i in range(len(b_grid)):
    for j in range(len(z_grid)):
        b_val = B_MESH[i, j]
        z_val = Z_MESH[i, j]

        # a) Solução da PINN
        z_tensor = torch.tensor([[z_val]], dtype=torch.float32, device=device)
        b_tensor = torch.tensor([[b_val]], dtype=torch.float32, device=device)
        Omega_tensor = torch.tensor([[Omega_fixed]], dtype=torch.float32, device=device)
        
        with torch.no_grad():
            pinn_sol = pinn_solver.get_solution(z_tensor, b_tensor, Omega_tensor)
        v_pinn = pinn_sol['v'].item()
        r_pinn = torch.exp(pinn_sol['r_prime']).item()
        H_div_H0_pinn = np.sqrt((r_pinn / (2 * v_pinn)) * (1 - Omega_fixed))

        # b) Solução Numérica (Ground Truth)
        # Condições iniciais em z=10
        term_num_comum = Omega_fixed * (1 + 10)**3
        term_den_comum = 1 / (term_num_comum + 1 - Omega_fixed)
        y0 = 0.5 * (term_num_comum + 2 * (1 - Omega_fixed)) * term_den_comum
        v0 = 0.5 * (term_num_comum + 4 * (1 - Omega_fixed)) * term_den_comum
        Omega0 = term_num_comum * term_den_comum
        r0 = (term_num_comum + 4 * (1 - Omega_fixed)) / (1 - Omega_fixed)
        u0 = [0, y0, v0, Omega0, r0]

        # Resolve a EDO de z=10 até z_val
        sol_numeric = solve_ivp(fR_odes, [10, z_val], u0, args=(b_val, Omega_fixed), dense_output=True, method='RK45')
        x_num, y_num, v_num, Omega_num, r_num = sol_numeric.sol(z_val)
        H_div_H0_numeric = np.sqrt((r_num / (2 * v_num)) * (1 - Omega_fixed))

        # c) Cálculo do Erro Percentual
        percent_error = 100 * np.abs((H_div_H0_pinn - H_div_H0_numeric) / H_div_H0_numeric)
        ERROR_MESH[i, j] = percent_error
    
    # Imprime o progresso
    print(f"Progresso: {((i+1)/len(b_grid)*100):.1f}% concluído.")

print("Cálculo do erro finalizado.")

plt.figure(figsize=(8, 6))
# Usamos pcolormesh para criar um mapa de calor
c = plt.pcolormesh(Z_MESH, B_MESH, ERROR_MESH, cmap='viridis', shading='gouraud', vmax=1.2)
plt.colorbar(c, label='Erro Percentual (%)')
plt.xlabel('Redshift (z)', fontsize=14)
plt.ylabel('Parâmetro b', fontsize=14)
plt.title(f'Erro Percentual da Solução PINN vs. Numérica ($\Omega_{{m,0}}^{{\Lambda}} = {Omega_fixed}$)', fontsize=15)
plt.savefig('error.pdf')
