import numpy as np
import requests
import re
from collections import Counter, defaultdict

# =============================================================================
# 1. PREPARAÇÃO DOS DADOS (O CORPUS)
# =============================================================================
print("1/6 - Baixando e limpando o texto de Dom Casmurro...")
url = "https://www.gutenberg.org/cache/epub/55752/pg55752.txt"
texto_bruto = requests.get(url).text.lower()

# Usamos Regex para pegar apenas palavras, removendo números e pontuação
# O GloVe precisa de uma sequência linear de tokens
palavras = re.findall(r'\b[a-zà-ÿ]+\b', texto_bruto)

# Criamos um vocabulário com as 2000 palavras mais frequentes.
# Por que limitar? Porque a matriz de coocorrência cresce ao quadrado (V x V).
vocab_size = 2000
contagem = Counter(palavras)
vocab = [p for p, _ in contagem.most_common(vocab_size)]

# Mapeamentos para transformar string em índice e vice-versa
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for i, w in enumerate(vocab)}

# =============================================================================
# 2. MATRIZ DE COOCORRÊNCIA (A "MEMÓRIA GLOBAL")
# =============================================================================
# O diferencial do GloVe: ele não esquece o texto. Ele resume tudo numa matriz.
print("2/6 - Construindo matriz de coocorrência (isso pode demorar um pouco)...")
window_size = 5
X = defaultdict(float)

for i, p_central in enumerate(palavras):
    if p_central not in word2idx: continue
    idx_i = word2idx[p_central]
    
    # Definimos a janela de contexto ao redor da palavra atual
    start = max(0, i - window_size)
    end = min(len(palavras), i + window_size + 1)
    
    for j in range(start, end):
        if i == j or palavras[j] not in word2idx: continue
        idx_j = word2idx[palavras[j]]
        
        # O paper original do GloVe sugere que palavras mais distantes 
        # na janela valem menos: peso = 1 / distância
        distancia = abs(i - j)
        X[(idx_i, idx_j)] += 1.0 / distancia

# =============================================================================
# 3. HIPERPARÂMETROS E INICIALIZAÇÃO
# =============================================================================
embedding_dim = 35  # Tamanho do vetor (ex: 100 números para descrever 'capitú')
eta = 0.05           # Taxa de aprendizado (Learning Rate)
epochs = 15          # Quantas vezes vamos ajustar os vetores
x_max = 100          # Teto para a função de peso (evita que 'o', 'a', 'e' dominem)
alpha = 0.75         # Expoente da função de peso (ajuste fino da importância)

# Inicializamos dois conjuntos de vetores e biases (vieses)
# No GloVe, cada palavra i tem um vetor W_i e um vetor de contexto W_tilde_i
W = (np.random.rand(vocab_size, embedding_dim) - 0.5) / embedding_dim
W_tilde = (np.random.rand(vocab_size, embedding_dim) - 0.5) / embedding_dim
b = (np.random.rand(vocab_size) - 0.5) / embedding_dim
b_tilde = (np.random.rand(vocab_size) - 0.5) / embedding_dim

# =============================================================================
# 4. O LOOP DE TREINAMENTO (GRADIENTE DESCENDENTE)
# =============================================================================
# O objetivo: minimizar J = f(X_ij) * (w_i.w_j + b_i + b_j - log(X_ij))^2
print(f"3/6 - Iniciando treinamento por {epochs} épocas...")

for epoch in range(epochs):
    total_cost = 0
    # Iteramos apenas sobre os pares que realmente apareceram juntos (X_ij > 0)
    for (i, j), x_ij in X.items():
        
        # A) Função de Peso: Amortece o impacto de palavras excessivamente comuns
        # f(x) = (x/x_max)^alpha se x < x_max, caso contrário 1
        weight = (x_ij / x_max)**alpha if x_ij < x_max else 1.0
        
        # B) O Erro: A diferença entre o que o modelo "acha" (dot product + bias)
        # e a realidade estatística (log da coocorrência)
        # Usamos log porque relações semânticas tendem a ser lineares no log-space
        dot_product = np.dot(W[i], W_tilde[j])
        log_x_ij = np.log(x_ij)
        diff = dot_product + b[i] + b_tilde[j] - log_x_ij
        
        # C) Custo: Erro quadrático ponderado
        total_cost += weight * (diff**2)
        
        # D) Atualização (Gradiente Descendente):
        # Ajustamos os vetores na direção oposta ao erro
        grad_common = weight * diff
        
        # Atualiza vetores
        W[i] -= eta * grad_common * W_tilde[j]
        W_tilde[j] -= eta * grad_common * W[i]
        
        # Atualiza os biases (ajustam a "frequência base" da palavra)
        b[i] -= eta * grad_common
        b_tilde[j] -= eta * grad_common
        
    print(f"   Época {epoch+1}/{epochs} | Custo Médio: {total_cost/len(X):.4f}")

# No GloVe, o vetor final de uma palavra é a soma do seu vetor principal 
# com o seu vetor de contexto. Isso melhora a estabilidade.
vetores_finais = W + W_tilde

# =============================================================================
# 5. FUNÇÕES DE ANÁLISE (SIMILARIDADE)
# =============================================================================
def get_vector(word):
    if word in word2idx:
        return vetores_finais[word2idx[word]]
    return None

def similaridade(w1, w2):
    v1, v2 = get_vector(w1), get_vector(w2)
    if v1 is None or v2 is None: return 0.0
    # Similaridade de Cosseno: medida padrão para vetores de NLP
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def mais_proximas(word, n=5):
    if word not in word2idx: return []
    v_base = get_vector(word)
    distancias = []
    for w in vocab:
        if w == word: continue
        score = similaridade(word, w)
        distancias.append((w, score))
    # Ordena do maior score para o menor
    return sorted(distancias, key=lambda x: x[1], reverse=True)[:n]

# =============================================================================
# 6. RESULTADOS
# =============================================================================
print("\n" + "="*40)
print("RESULTADOS DO MODELO GLOVE (DOM CASMURRO)")
print("="*40)

duplas = [("capitú", "bentinho"), ("capitú", "escobar"), ("bentinho", "ciúmes"), ("olhos", "ressaca")]

for p1, p2 in duplas:
    print(f"Similaridade [{p1} - {p2}]: {similaridade(p1, p2):.4f}")

print("\nPalavras mais próximas de 'capitú':")
for pal, score in mais_proximas("capitú"):
    print(f" -> {pal}: {score:.4f}")