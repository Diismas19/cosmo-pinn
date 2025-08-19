# Solução da Dinâmica Cosmológica do Modelo f(R) de Hu-Sawicki com Redes Neurais Informadas pela Física

Este repositório contém a implementação em PyTorch de uma Rede Neural Informada pela Física (PINN) para resolver o sistema de equações diferenciais que descreve a dinâmica de fundo do universo no modelo de gravidade modificada $f(R)$ de Hu-Sawicki.

O projeto foi desenvolvido como um estudo de caso sobre a aplicação de técnicas de aprendizado de máquina em problemas complexos da física teórica, com foco na superação de desafios comuns de treinamento, como o desbalanceamento da função de perda.

## Contexto

Modelos cosmológicos modernos são descritos por sistemas de equações diferenciais complexas que, muitas vezes, não possuem solução analítica. A abordagem tradicional via solvers numéricos pode gerar gargalos computacionais em análises estatísticas. As PINNs surgem como uma alternativa promissora, aprendendo a solução ao satisfazer diretamente as leis físicas do problema, de forma não-supervisionada.

## Principais Características

-   **Implementação em PyTorch:** Código modular e claro para a definição, treinamento e validação da PINN.
-   **Técnicas Avançadas:** Utilização de reparametrização perturbativa para impor condições de contorno e ponderação manual da perda para garantir a convergência.
-   **Validação Rigorosa:** Comparação da acurácia do modelo treinado contra um solver numérico de alta precisão da biblioteca SciPy.
-   **Reproducibilidade:** O código está estruturado para permitir a fácil reprodução dos resultados apresentados.

## Configuração do Ambiente

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/Diismas19/cosmo-pinn
    cd cosmo-pinn
    ```

2.  **Crie e ative um ambiente virtual (recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows: venv\Scripts\activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

## Como Usar

O pipeline é executado em duas etapas principais:

1.  **Treinamento do Modelo:**
    Execute o script de treinamento. Este processo é computacionalmente intensivo e pode levar um tempo considerável, dependendo do hardware. Os modelos treinados (arquivos `.pth`) serão salvos automaticamente.
    ```bash
    python train.py
    ```

2.  **Avaliação e Visualização do Erro:**
    Após o treinamento, execute o script de avaliação para gerar o mapa de erro percentual, comparando a solução da PINN com a solução numérica. O gráfico será exibido na tela e salvo.
    ```bash
    python evaluate_error.py
    ```

## Resultados

O treinamento do modelo com ponderação de perdas foi bem-sucedido. A figura abaixo mostra o erro percentual da solução da PINN, validando a alta precisão do método em uma vasta região do espaço de parâmetros.

![Mapa de Erro da PINN](error.pdf)
*Mapa do erro percentual da solução da PINN para $H(z)/H_0^{\Lambda}$ em comparação com a solução numérica.*

## Agradecimentos

Este trabalho foi fortemente baseado na metodologia apresentada no artigo "Cosmology-informed neural networks to solve the background dynamics of the Universe" de A. T. Chantada et al. (2023).

## Licença

Este projeto é distribuído sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.
