import torch
import torch.nn as nn
import numpy as np
import time

# Ponto inicial para as condições iniciais
Z0 = torch.tensor(10.0, dtype=torch.float32)

# Intervalos dos parâmetros para o treinamento das redes (bundle)
Z_MIN, Z_MAX = 0.0, 10.0
B_MIN, B_MAX = 0.0, 5.0
OMEGA_M0_L_MIN, OMEGA_M0_L_MAX = 0.1, 0.4

# Classe para definir a arquitetura da nossa rede neural usando PyTorch
class PINN_Architecture(nn.Module):
    """
    Define a arquitetura da rede neural herdando de nn.Module.
    Arquitetura: (3 entradas) -> (32 neurônios, tanh) -> (32 neurônios, tanh) -> (1 saída)
    """
    def __init__(self):
        super(PINN_Architecture, self).__init__()
        
        # Definindo as camadas da rede
        self.layer1 = nn.Linear(3, 32) # Camada de entrada/oculta 1
        self.layer2 = nn.Linear(32, 32) # Camada oculta 2
        self.output_layer = nn.Linear(32, 1) # Camada de saída
        
        # Definindo a função de ativação
        self.tanh = nn.Tanh()

    def forward(self, x):
        """Define o fluxo de dados através da rede (a passagem para frente)."""
        # Passa pela camada 1 e aplica a função de ativação
        x = self.tanh(self.layer1(x))
        # Passa pela camada 2 e aplica a função de ativação
        x = self.tanh(self.layer2(x))
        # Passa pela camada de saída (geralmente sem ativação para problemas de regressão)
        x = self.output_layer(x)
        return x

# Criando as 5 redes neurais, uma para cada variável do sistema de EDOs.
model_x = PINN_Architecture()
model_y = PINN_Architecture()
model_v = PINN_Architecture()
model_Omega = PINN_Architecture()
model_r_prime = PINN_Architecture() # Lembre-se, esta rede vai prever r' = ln(r)

# Vamos visualizar a arquitetura de um dos modelos para confirmar.
print("\nArquitetura de um dos modelos (ex: model_x):")
print(model_x)
