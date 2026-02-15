#### Word2Vec com o máximo de abstração possível (Gensim)

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import spacy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#carregar o modelo do spacy em PTBR, desabilitamos parser e ner para otimizar performance, já que não vamos utiliza-los.
nlp = spacy.load("pt_core_news_sm", disable=["parser","ner"])


#Texto sem ser tokenizado
corpus = [
    "Eu gosto de NLP!!!", 
    "Python é muito legal; ele resolve tudo.",
    "O Word2Vec cria vetores? Sim, cria.",
    "Estou estudando NLP",
    "Será que alguém algum dia vai ver esse código? rss",
    "Olá, espero que esteja bem",
    "Se você for um contratante e está avaliando meu currículo, espero que esteja gostando do que está vendo",
    "Se você for apenas um curioso, também espero que esteja gostando, rsrs",
    "Estou estudando NLP há 7 horas e 30 minutos, talvez eu fique louco jaja"
    "NLP é muito legal",
    "Eu programo NLP em python",
    "Você vai avaliar minhas habilidades em NLP?",
    "A melhor linguagem para programar NLP é Python"
]

#List comprehension onde tokenizamos, normalizamos, lematizamos e tiramos stop words.
#para cada token na frase processada, pega o lema minúsculo se não for stop word e nem letra.
texto_tokenizado = [
    [token.lemma_.lower() for token in nlp(frase) if not token.is_stop and token.is_alpha]
    for frase in corpus
]

def plotar_grafico(model):
    # 1. Extrair todas as palavras e seus vetores
    palavras = list(model.wv.index_to_key)
    vetores = model.wv[palavras]
    
    # 2. Reduzir de 8 dimensões para 2 usando PCA
    pca = PCA(n_components=2)
    vetores_2d = pca.fit_transform(vetores)
    
    # 3. Criar o gráfico
    plt.figure(figsize=(12, 8))
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
model = Word2Vec(sentences=texto_tokenizado, vector_size=13, window=10, min_count = 1, sg=0, epochs = 21, compute_loss = True)

#Executando modelo para verificar vetores da palavra "nlp"
vector = model.wv['python']

#função de ver a loss
loss_final = model.get_latest_training_loss()
print(f"Erro final acumulado: {loss_final}")
#print('Vetores da palavra "python": ', vector)
print('\n\nPalavras mais parecidas com "python"', model.wv.most_similar('nlp'))

plotar_grafico(model)

