#### TF-IDF NA MÃO ####

#Calculando TF

#Definição dos documentos
documentos = ["mãe aném", "O rato roeu a roupa do rei de roma",  "Os cachorros caçam os gatos", "a velha", "a mãe", "minha mãe é legal"]

#função de normalizar os documentos
def tratamento(frase):
    return frase.lower()

#list comprehesion para normalizacao
documentos1 = [tratamento(frase) for frase in documentos]

print("Frases normalizadas:", documentos1)

#transformação em string para usar split() e separar as palavras
super_string = " ".join(documentos1)
palavras = super_string.split()

print("\n\nEssas são as palavras separadas:", palavras)

#set para remover duplicatas, list para transformar novamente em lista e sorted para organizar
vocabulario = sorted(list(set(palavras)))

#transforma cada frase em uma lista de palavras
documentos_tokenizados = [frase.split() for frase in documentos1]

def calcular_tf(frase_tokenizada, vocabulario):
    tf_da_frase = []
    total_de_palavras_na_frase = len(frase_tokenizada) #denominador da formula

    for palavras in vocabulario:
        qtd = frase_tokenizada.count(palavras)
        
        #frequencia do termo / total de termos no documento
        tf = qtd / total_de_palavras_na_frase
        tf_da_frase.append(tf)
    
    return tf_da_frase

#cria a matriz de frequências para todos os documentos 
matriz_tf = [calcular_tf(doc, vocabulario) for doc  in documentos_tokenizados]

#Calculando IDF

from math import log

idf_pesos = []
total_documentos = len(documentos_tokenizados)

for palavra in vocabulario:
    docs_com_a_palavra = 0

    #verifica quantos documentos a palavra aparece pelo menos uma vez
    for doc in documentos_tokenizados:
        if palavra in doc:
            docs_com_a_palavra += 1
    

    # IDF = log (total de documentos / documentos que contêm a palavra)
    resultado_idf = log(total_documentos / docs_com_a_palavra)
    idf_pesos.append(resultado_idf)


# --- MULTIPLICAÇÃO TF * IDF ---

matriz_tfidf = []

for linha_tf in matriz_tf:
    tfidf_da_frase = []
    

    #multiplicando o peso local pelo peso global (TF x IDF)
    for i in range(len(linha_tf)):
        valor_final = linha_tf[i] * idf_pesos[i]
        tfidf_da_frase.append(valor_final)
    
    matriz_tfidf.append(tfidf_da_frase)

print("\n\nPalavra | Peso TF-IDF")

for p, peso in zip(vocabulario, matriz_tfidf[0]):

    if peso > 0: # Imprimir apenas as que tem peso

        print(f"{p}: {peso:.4f}")

print(f"Tamanho do vocabulário: {len(vocabulario)}")
print(f"Tamanho da lista de pesos IDF: {len(idf_pesos)}")

#### TF-IDF COM BIBLIOTECAS (SCIKIT LEARN) ####

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

#criando vetorizador
vetorizador = TfidfVectorizer()

#Aprende o o vocabulário e calcula TFIDF de uma vez
tfidf_matriz = vetorizador.fit_transform(documentos)

#visualizando com pandas
df_tfidf = pd.DataFrame(
    tfidf_matriz.toarray(),
    columns = vetorizador.get_feature_names_out(),
    index = [f"Frase {i+1}" for i in range(len(documentos))]
)

print("\n\n Usando scikit-learn: \n", df_tfidf.round(4))