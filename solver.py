import torch
import torch.nn as nn

# Importando a arquitetura da nossa rede do arquivo model.py
from model import PINN_Architecture

# Importando as constantes físicas que definimos anteriormente
from model import Z0, B_MAX 
# (Vamos usar Z0 para a reparametrizacao e B_MAX para a normalização de b')

def calculate_base_solution(z, Omega_m0_L):
    """
    Calcula a solução base do modelo (solução para b=0, ou seja, LambdaCDM).
    
    Args:
        z (torch.Tensor): Tensor de valores de redshift.
        Omega_m0_L (torch.Tensor): Tensor de valores do parâmetro de densidade de matéria.

    Returns:
        dict: Um dicionário contendo os tensores para cada variável da solução base.
    """
    # A solução base é definida em z0=10, mas a forma funcional é a mesma para qualquer z
    # Vamos renomear z0 para z na equação para maior clareza
    z0 = z
    
    # Termos comuns para simplificar os cálculos
    term_num_comum = Omega_m0_L * (1 + z0)**3
    term_den_comum = 1 / (term_num_comum + 1 - Omega_m0_L)
    
    # Solução para cada variável quando b=0
    x_hat = torch.zeros_like(z) # x(z) é sempre 0 para b=0
    
    y_hat_num = term_num_comum + 2 * (1 - Omega_m0_L)
    y_hat = 0.5 * y_hat_num * term_den_comum
    
    v_hat_num = term_num_comum + 4 * (1 - Omega_m0_L)
    v_hat = 0.5 * v_hat_num * term_den_comum
    
    Omega_hat = term_num_comum * term_den_comum
    
    r_hat_num = term_num_comum + 4 * (1 - Omega_m0_L)
    r_hat = r_hat_num / (1 - Omega_m0_L)
    
    return {
        'x': x_hat, 'y': y_hat, 'v': v_hat, 
        'Omega': Omega_hat, 'r': r_hat
    }

class PINNSolver:
    """
    Classe que encapsula os modelos e a lógica de reparametrização da PINN.
    """
    def __init__(self):
        # Inicializa as 5 redes neurais, uma para cada variável
        self.model_x = PINN_Architecture()
        self.model_y = PINN_Architecture()
        self.model_v = PINN_Architecture()
        self.model_Omega = PINN_Architecture()
        self.model_r_prime = PINN_Architecture()
        # Coloca todos os modelos em uma lista para facilitar o acesso aos parâmetros
        self.models = [self.model_x, self.model_y, self.model_v, self.model_Omega, self.model_r_prime]

    def get_solution(self, z, b, Omega_m0_L):
        """
        Calcula a solução final reparametrizada.

        Args:
            z (torch.Tensor): Tensor de valores de redshift.
            b (torch.Tensor): Tensor de valores do parâmetro b.
            Omega_m0_L (torch.Tensor): Tensor de valores de densidade de matéria.

        Returns:
            dict: Dicionário com as soluções finais (reparametrizadas) para cada variável.
        """
        # Normalização das variáveis de entrada para a rede
        z_prime = 1 - z / Z0
        b_prime = b / B_MAX

        # Concatena as entradas para formar o input da rede
        # O formato esperado é (batch_size, num_features), onde num_features=3
        nn_input = torch.cat([z_prime, b_prime, Omega_m0_L], dim=1)
        
        # Obtém as saídas brutas (a "correção") de cada rede neural
        x_N = self.model_x(nn_input)
        y_N = self.model_y(nn_input)
        v_N = self.model_v(nn_input)
        Omega_N = self.model_Omega(nn_input)
        r_prime_N = self.model_r_prime(nn_input)

        # Calcula a solução base (LambdaCDM, para b=0)
        base_sols = calculate_base_solution(z, Omega_m0_L)
        
        # Fator de reparametrização que zera a contribuição da rede em z=z0 e b=0
        reparam_factor = (1 - torch.exp(-z_prime)) * (1 - torch.exp(-b))

        # Solução final para as 4 primeiras variáveis
        tilde_x = base_sols['x'] + reparam_factor * x_N
        tilde_y = base_sols['y'] + reparam_factor * y_N
        tilde_v = base_sols['v'] + reparam_factor * v_N
        tilde_Omega = base_sols['Omega'] + reparam_factor * Omega_N
        
        # Para r', a lógica é a mesma, mas a base é o logaritmo de r_hat
        # Usamos torch.log para manter a operação dentro do grafo computacional do PyTorch
        r_prime_hat = torch.log(base_sols['r'])
        tilde_r_prime = r_prime_hat + reparam_factor * r_prime_N
        
        return {
            'x': tilde_x, 'y': tilde_y, 'v': tilde_v, 
            'Omega': tilde_Omega, 'r_prime': tilde_r_prime
        }

    def loss_fn(self, z, b, Omega_m0_L):
        """
        Calcula a perda total e também retorna um dicionário com as perdas individuais.
        """
        z.requires_grad_()
        sols = self.get_solution(z, b, Omega_m0_L)
        x, y, v, Omega, r_prime = sols['x'], sols['y'], sols['v'], sols['Omega'], sols['r_prime']
        r = torch.exp(r_prime)

        # Cálculo dos resíduos das EDOs (L_R)
        grad_outputs = torch.ones_like(x)
        dxdz = torch.autograd.grad(x, z, grad_outputs=grad_outputs, create_graph=True)[0]
        dydz = torch.autograd.grad(y, z, grad_outputs=grad_outputs, create_graph=True)[0]
        dvdz = torch.autograd.grad(v, z, grad_outputs=grad_outputs, create_graph=True)[0]
        dOmegadz = torch.autograd.grad(Omega, z, grad_outputs=grad_outputs, create_graph=True)[0]
        dr_primedz = torch.autograd.grad(r_prime, z, grad_outputs=grad_outputs, create_graph=True)[0]
        drdz = r * dr_primedz
        
        epsilon = 1e-9 
        gamma_term = ((r + b) * ((r + b)**2 - 2 * b)) / (4 * b * r + epsilon)

        res_x = dxdz - (1 / (1 + z)) * (-Omega - 2*v + x + 4*y + x*v + x**2)
        res_y = dydz - (-1 / (1 + z)) * (v*x*gamma_term - x*y + 4*y - 2*y*v)
        res_v = dvdz - (-v / (1 + z)) * (x*gamma_term + 4 - 2*v)
        res_Omega = dOmegadz - (Omega / (1 + z)) * (-1 + 2*v + x)
        res_r = drdz - (-r * gamma_term * x) / (1 + z)
        
        loss_x = torch.mean(res_x**2)
        loss_y = torch.mean(res_y**2)
        loss_v = torch.mean(res_v**2)
        loss_Omega = torch.mean(res_Omega**2)
        loss_r = torch.mean(res_r**2)
        
        # Definimos pesos para balancear as perdas. A loss_r é nossa base (peso 1).
        # Aumentamos o peso das outras para que tenham magnitude similar à de loss_r.
        w_x, w_y, w_v, w_Omega, w_r = 1e5, 1e4, 1e4, 1e5, 1.0
        loss_edos_total = (w_x * loss_x + 
                           w_y * loss_y + 
                           w_v * loss_v + 
                           w_Omega * loss_Omega + 
                           w_r * loss_r)

        # Cálculo dos resíduos das restrições (L_C)
        res_c1 = 1 - (Omega + v - x - y)
        res_c2_lhs = Omega * r * (1 - Omega_m0_L.detach()) * (r + b - 2)
        res_c2_rhs = 2 * y * Omega_m0_L.detach() * (1 + z)**3 * (r + b)
        res_c2 = res_c2_lhs - res_c2_rhs
        res_c3_lhs = Omega * r * (1 - Omega_m0_L.detach()) * ((r + b)**2 - 2 * b)
        res_c3_rhs = 2 * v * Omega_m0_L.detach() * (1 + z)**3 * (r + b)**2
        res_c3 = res_c3_lhs - res_c3_rhs

        loss_c1 = torch.mean(res_c1**2)
        loss_c2 = torch.mean((res_c2*1e-5)**2)
        loss_c3 = torch.mean((res_c3*1e-5)**2)

        loss_constraints_total = loss_c1 + loss_c2 + loss_c3
        
        # Perda total
        total_loss = loss_edos_total + loss_constraints_total
        
        # Dicionário com todas as perdas para análise
        loss_dict = {
            'total': total_loss,
            'edos': loss_edos_total,
            'constraints': loss_constraints_total,
            'x': loss_x, 'y': loss_y, 'v': loss_v, 
            'Omega': loss_Omega, 'r': loss_r,
            'c1': loss_c1, 'c2': loss_c2, 'c3': loss_c3
        }
        
        return loss_dict
