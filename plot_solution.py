import torch
import numpy as np
import matplotlib.pyplot as plt

# Importando a classe do nosso solver e as constantes
from solver import PINNSolver
from model import Z_MIN, Z_MAX

# Verificando se uma GPU está disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando o dispositivo: {device}")

# Instancia o solver. Ele irá criar 5 novos modelos com pesos aleatórios.
pinn_solver = PINNSolver()

# Agora, carregamos os pesos que salvamos durante o treinamento para cada modelo.
# O PyTorch faz o "match" dos pesos salvos com a arquitetura do modelo.
try:
    pinn_solver.model_x.load_state_dict(torch.load("model_x.pth"))
    pinn_solver.model_y.load_state_dict(torch.load("model_y.pth"))
    pinn_solver.model_v.load_state_dict(torch.load("model_v.pth"))
    pinn_solver.model_Omega.load_state_dict(torch.load("model_Omega.pth"))
    pinn_solver.model_r_prime.load_state_dict(torch.load("model_r_prime.pth"))
except FileNotFoundError:
    print("Erro: Arquivos de modelo treinado (.pth) não encontrados.")
    print("Certifique-se de que o treinamento foi concluído e os arquivos estão no mesmo diretório.")
    exit()

# Move os modelos para o dispositivo e os coloca em modo de avaliação.
# .eval() informa ao PyTorch que não estamos mais treinando, otimizando a performance.
for model in pinn_solver.models:
    model.to(device)
    model.eval()

# Fixamos um valor para Omega e variamos o parâmetro b.
Omega_fixed = 0.3
b_values_to_plot = [0,0.1, 0.5, 1.0, 2.0]

# Criamos um conjunto de 200 pontos no eixo z para ter uma curva suave.
z_plot = torch.linspace(Z_MIN, Z_MAX, 200).view(-1, 1).to(device)


plt.figure(figsize=(10, 6))
# Itera sobre cada valor de 'b' que queremos plotar
for b_val in b_values_to_plot:
    print(f"Gerando solução para b = {b_val}...")
    
    # Prepara os tensores de b e Omega para terem o mesmo tamanho do tensor z
    b_plot = torch.full_like(z_plot, fill_value=b_val)
    Omega_plot = torch.full_like(z_plot, fill_value=Omega_fixed)

    # Usa o solver para obter a solução.
    # torch.no_grad() desliga o cálculo de gradientes, tornando a inferência mais rápida.
    with torch.no_grad():
        solution = pinn_solver.get_solution(z_plot, b_plot, Omega_plot)

    # Extrai as variáveis necessárias da solução
    v = solution['v']
    r_prime = solution['r_prime']
    r = torch.exp(r_prime) # Converte r' de volta para r
    H_div_H0 = torch.sqrt((r / (2 * v)) * (1 - Omega_fixed))

    # Converte os tensores para numpy para poder plotar com matplotlib
    z_numpy = z_plot.cpu().numpy()
    H_div_H0_numpy = H_div_H0.cpu().numpy()

    # Adiciona a curva ao gráfico
    if b_val == 0:
      plt.plot(z_numpy, H_div_H0_numpy, label=r'b=0 ($\Lambda CDM$)')
    else:
      plt.plot(z_numpy, H_div_H0_numpy, label=f'b = {b_val}')

plt.xlabel('Redshift (z)', fontsize=14)
plt.ylabel('$H(z) / H_0^{\Lambda}$', fontsize=14)
plt.title(f'Solução da PINN para o Modelo $f(R)$ com $\Omega_{{m,0}}^{{\Lambda}} = {Omega_fixed}$', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim(0, 5)
plt.xlim(0, 3) 
plt.savefig('sol_pinn.pdf')
