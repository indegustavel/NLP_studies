#### Word2Vec com o máximo de abstração possível (Gensim)

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import spacy

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
    "Se você for um contratante e está me avaliando/avaliando meu currículo, espero que esteja gostando do que está vendo",
    "Se você for apenas um curioso, também espero que esteja gostando, rsrs",
    "Estou estudando NLP há 7 horas e 30 minutos, talvez eu fique louco jajá"
    "NLP é muito legal",
    "Eu programo NLP em python",
    "Você vai avaliar minhas habilidades em NLP?"
]

#List comprehension onde tokenizamos, normalizamos, lematizamos e tiramos stop words.
#para cada token na frase processada, pega o lema minúsculo se não for stop word e nem letra.
texto_tokenizado = [
    [token.lemma_.lower() for token in nlp(frase) if not token.is_stop and token.is_alpha]
    for frase in corpus
]

# print(texto_tokenizado)

#Usando modelo word2vec da biblioteca gensim (nível altísimo de abstração)
#Vector_size maiores capturam mais nuances, mas precisam de muito texto para treino
#window maiores focam em semântica, menores focam em sintaxe.
#sg = 0 é a técnica CBOW (prever palavra no meio do contexto), sg = 1 é Skip-gram (prever contexto a partir de uma palavra)
#epoch é quantas vezes o modelo vai ser seu dataset inteiro. Como temos poucas palavras, é recomendado que seja bastante.
model = Word2Vec(sentences=texto_tokenizado, vector_size=15, window=5, min_count = 1, sg=0, epochs = 1000)

#Executando modelo para verificar vetores da palavra "nlp"
vector = model.wv['nlp']

print('Vetores da palavra "NLP": ', vector)
print('\n\nPalavras mais parecidas com "NLP"', model.wv.most_similar('nlp'))
