import numpy as np
import random

def gerar_dataset_gigante(n_amostras=5000, seq_len=900):
    """
    Gera frases longas onde a primeira palavra define o sentimento,
    mas a última palavra é sempre o oposto (para enganar a RNN).
    O meio é preenchido com palavras aleatórias de 'ruído'.
    """
    X = []
    y = []

    # Criamos um vocabulário de "palavras de ruído"
    # Ex: "palavra_1", "palavra_55", etc. 
    # Isso simula um texto real com vocabulário variado, o que é pior para a memória que um token repetido.
    vocab_ruido = [f"contexto_{i}" for i in range(1000)] 

    for _ in range(n_amostras):
        # 1. Gera o "recheio" da frase (400 palavras aleatórias)
        # O uso de random.choice garante que o estado oculto da RNN mude a cada passo
        meio_ruidoso = [random.choice(vocab_ruido) for _ in range(seq_len)]
        
        if np.random.rand() > 0.5:
            # --- CASO POSITIVO (Label 1) ---
            # A verdade está no início: "excelente"
            # A armadilha está no fim: "horrivel"
            frase_lista = ["excelente"] + meio_ruidoso + ["horrivel"]
            label = 1
        else:
            # --- CASO NEGATIVO (Label 0) ---
            # A verdade está no início: "horrivel"
            # A armadilha está no fim: "excelente"
            frase_lista = ["horrivel"] + meio_ruidoso + ["excelente"]
            label = 0
            
        # Junta a lista em uma string única separada por espaços
        X.append(" ".join(frase_lista))
        y.append(label)
    
    return np.array(X), np.array(y)

# --- Testando a geração ---
X_texto, y_texto = gerar_dataset_gigante()

print(f"Tipo do dado: {type(X_texto[0])}") # Agora é <class 'numpy.str_'>
print(f"Tamanho da frase: {len(X_texto[0].split())} palavras")
print("-" * 30)
print(f"Exemplo (Início...Fim):\n{X_texto[0][:50]} ... {X_texto[0][-50:]}")
print(f"Label Real: {y_texto[0]}")