#### Word2Vec com o máximo de abstração possível (Gensim)

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import spacy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import requests

#carregar o modelo do spacy em PTBR, desabilitamos parser e ner para otimizar performance, já que não vamos utiliza-los.
nlp = spacy.load("pt_core_news_sm", disable=["parser","ner"])

print("Baixando 'Dom Casmurro'...")
url = "https://www.gutenberg.org/cache/epub/55752/pg55752.txt"
response = requests.get(url)
texto_completo = response.text

#Texto sem ser tokenizado
corpus = [linha for linha in texto_completo.splitlines() if len(linha) > 10]

print(f"Livro baixado! Total de linhas para processar: {len(corpus)}")

#para cada token na frase processada, pega o lema minúsculo se não for stop word e nem letra.
texto_tokenizado = []
# nlp.pipe retorna um gerador, iteramos sobre os docs processados
for doc in nlp.pipe(corpus, batch_size=50, n_process=1): # n_process > 1 se quiser usar multi-core no spacy
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]
    if tokens: # garante que não salvamos listas vazias
        texto_tokenizado.append(tokens)

print("Processando e tokenizando (aguarde um pouco)...")

def plotar_grafico(model, top_n = 20):
    # 1. Extrair todas as palavras e seus vetores
    palavras = list(model.wv.index_to_key)[:top_n]
    vetores = model.wv[palavras]
    
    # 2. Reduzir de 8 dimensões para 2 usando PCA
    pca = PCA(n_components=2)
    vetores_2d = pca.fit_transform(vetores)
    
    # 3. Criar o gráfico
    plt.figure(figsize=(14, 10))
    plt.scatter(vetores_2d[:, 0], vetores_2d[:, 1], c='blue', edgecolors='k', alpha=0.7)
    
    # 4. Adicionar os nomes das palavras nos pontos
    for i, palavra in enumerate(palavras):
        plt.annotate(palavra, xy=(vetores_2d[i, 0], vetores_2d[i, 1]), xytext=(5, 2), 
                     textcoords='offset points', ha='right', va='bottom', fontsize=9)
    
    plt.title("Mapa Semântico do Word2Vec (PCA)")
    plt.grid(True)
    plt.show()

# print(texto_tokenizado)

#Usando modelo word2vec da biblioteca gensim (nível altísimo de abstração)
#Vector_size maiores capturam mais nuances, mas precisam de muito texto para treino
#window maiores focam em semântica, menores focam em sintaxe.
#sg = 0 é a técnica CBOW (prever palavra no meio do contexto), sg = 1 é Skip-gram (prever contexto a partir de uma palavra)
#epoch é quantas vezes o modelo vai ser seu dataset inteiro. Como temos poucas palavras, é recomendado que seja bastante.
#compute loss para vermos a taxa de erro do modelo
#workers é a quantidade de núcleos que seu pc vai usar para treinar o modelo
model = Word2Vec(sentences=texto_tokenizado, vector_size=30, window=6, min_count = 4, sg=1, epochs = 100, compute_loss = True, workers = 4)



#Executando modelo para verificar vetores da palavra "bentinho"
vector = model.wv['bentinho']

#print('Vetores da palavra "bentinho": ', vector)

#função de ver a loss

# --- FORMATAÇÃO DE SAÍDA ---

def imprimir_cabecalho(titulo):
    print("\n" + "="*50)
    print(f" {titulo.upper()} ".center(50, " "))
    print("="*50)

# 1. Erro de Treinamento
imprimir_cabecalho("Status do Treinamento")
loss_final = model.get_latest_training_loss()
print(f" Loss Acumulada:  {loss_final:,.2f}".replace(",", "."))
print(f" Épocas concluídas: {model.epochs}")

# 2. Perfil Semântico: Bentinho
imprimir_cabecalho("Perfil de 'Bentinho'")
print(f"{'PALAVRA':<15} | {'SIMILARIDADE':>12}")
print("-" * 30)
for palavra, score in model.wv.most_similar('bentinho'):
    print(f"{palavra:<15} | {score:>12.4f}")

# 3. O Grande Embate (Similaridade)
imprimir_cabecalho("O Teste da Traição")
sim_marido = model.wv.similarity('bentinho', 'capitú')
sim_amigo = model.wv.similarity('escobar', 'capitú')

print(f" Bentinho + Capitú: {sim_marido:.4f}")
print(f" Escobar  + Capitú: {sim_amigo:.4f}")
print("-" * 50)
veredito = " ALERTA: Proximidade Suspeita!" if sim_amigo > sim_marido else " Calma: Laços Conjugais Fortes."
print(veredito.center(50))

# 4. Álgebra Vetorial
imprimir_cabecalho("Álgebra Proibida")
print("> Equação: (Capitú + Escobar) - Bentinho\n")

resultado = model.wv.most_similar(
    positive=['capitú', 'escobar'], 
    negative=['bentinho'], 
    topn=10
)

print(f"{'RANK':<5} | {'CONCEITO':<15} | {'SCORE':>10}")
for i, (palavra, score) in enumerate(resultado, 1):
    print(f"{i:<5} | {palavra:<15} | {score:>10.4f}")
print("="*50)