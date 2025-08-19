import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import time

from solver import PINNSolver
from model import Z_MIN, Z_MAX, B_MIN, B_MAX, OMEGA_M0_L_MIN, OMEGA_M0_L_MAX

NUM_ITERATIONS = 200000 
BATCH_SIZE_Z = 128
BATCH_SIZE_PARAMS = 64
LEARNING_RATE = 1e-4 # Taxa de aprendizado inicial

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando o dispositivo: {device}")

pinn_solver = PINNSolver()
for model in pinn_solver.models:
    model.to(device)

params_to_optimize = []
for model in pinn_solver.models:
    params_to_optimize.extend(model.parameters())

optimizer = optim.Adam(params_to_optimize, lr=LEARNING_RATE)

# Inicializa o agendador de taxa de aprendizado
# 'gamma=0.9' significa que a cada 'step_size' iterações, a taxa de aprendizado será multiplicada por 0.9
scheduler = ExponentialLR(optimizer, gamma=0.98)

print("Iniciando treinamento refinado...")
start_time = time.time()

for iteration in range(1, NUM_ITERATIONS + 1):
    optimizer.zero_grad()
    
    z_samples = (Z_MAX - Z_MIN) * torch.rand(BATCH_SIZE_Z, 1, device=device) + Z_MIN
    b_samples = (B_MAX - B_MIN) * torch.rand(BATCH_SIZE_PARAMS, 1, device=device) + B_MIN
    Omega_m0_L_samples = (OMEGA_M0_L_MAX - OMEGA_M0_L_MIN) * torch.rand(BATCH_SIZE_PARAMS, 1, device=device) + OMEGA_M0_L_MIN
    
    z_batch = z_samples.repeat_interleave(BATCH_SIZE_PARAMS, dim=0)
    b_batch = b_samples.repeat(BATCH_SIZE_Z, 1)
    Omega_m0_L_batch = Omega_m0_L_samples.repeat(BATCH_SIZE_Z, 1)
    
    # Obtém o dicionário de perdas
    loss_dict = pinn_solver.loss_fn(z_batch, b_batch, Omega_m0_L_batch)
    total_loss = loss_dict['total']
    
    total_loss.backward()
    optimizer.step()
    
    # A cada 5000 iterações, diminui a taxa de aprendizado
    if iteration % 5000 == 0:
        scheduler.step()

    # Imprime o progresso a cada 1000 iterações
    if iteration % 1000 == 0:
        elapsed_time = time.time() - start_time
        current_lr = scheduler.get_last_lr()[0]
        
        # Log mais detalhado
        print(f"Iter: {iteration}/{NUM_ITERATIONS} | LR: {current_lr:.2e} | Tempo: {elapsed_time:.2f}s")
        print(f"    Loss Total: {loss_dict['total']:.3e} | EDOs: {loss_dict['edos']:.3e} | Constr: {loss_dict['constraints']:.3e}")
        print(f"    Perdas EDOs (x,y,v,Ω,r): {loss_dict['x']:.2e}, {loss_dict['y']:.2e}, {loss_dict['v']:.2e}, {loss_dict['Omega']:.2e}, {loss_dict['r']:.2e}")
        print("-" * 50)
        
        start_time = time.time()

torch.save(pinn_solver.model_x.state_dict(), "model_x.pth")
torch.save(pinn_solver.model_y.state_dict(), "model_y.pth")
torch.save(pinn_solver.model_v.state_dict(), "model_v.pth")
torch.save(pinn_solver.model_Omega.state_dict(), "model_Omega.pth")
torch.save(pinn_solver.model_r_prime.state_dict(), "model_r_prime.pth")
print("Modelos salvos com sucesso!")
