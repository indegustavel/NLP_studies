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
model = Word2Vec(sentences=texto_tokenizado, vector_size=103, window=7, min_count = 5, sg=0, epochs = 7, compute_loss = True, workers = 4)

#Executando modelo para verificar vetores da palavra "bentinho"
vector = model.wv['bentinho']

#função de ver a loss
loss_final = model.get_latest_training_loss()
print(f"Erro final acumulado: {loss_final}")
#print('Vetores da palavra "bentinho": ', vector)
print('\n\nPalavras mais parecidas com "bentinho"', model.wv.most_similar('bentinho'))

plotar_grafico(model)

