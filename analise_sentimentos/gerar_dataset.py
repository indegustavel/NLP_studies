import numpy as np

def gerar_dataset_gigante(n_amostras=100000):
    posi = ["excelente", "maravilhoso", "amei", "recomendo", "melhor", "espetacular"]
    nega = ["horrível", "porcaria", "odiei", "ruim", "péssimo", "desperdício"]
    fillers = ["o filme", "a cena", "o roteiro", "a atuação", "no cinema"]

    textos_treino = []
    labels = []

    for _ in range(n_amostras // 2):
        # Lógica de geração (simplificada aqui para o exemplo)
        palavra = np.random.choice(posi)
        textos_treino.append(f"{palavra} " + " ".join(np.random.choice(fillers, 50)))
        labels.append(1)
        
        palavra = np.random.choice(nega)
        textos_treino.append(f"{palavra} " + " ".join(np.random.choice(fillers, 50)))
        labels.append(0)

    return np.array(textos_treino), np.array(labels)

# Isso garante que o código só rode se você executar o arquivo diretamente
if __name__ == "__main__":
    t, l = gerar_dataset_gigante(10)
    print("Teste de geração concluído!")