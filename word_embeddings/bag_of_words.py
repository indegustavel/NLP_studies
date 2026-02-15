##### BAG OF WORDS NA MÃO ####

textos = ["O gato subiu no telhado", "O cachorro late", "Meu nome é Gustavo"] #Definindo palavras que servirão de dataset.

#função para colocar todas as letras em minúsculo
def tratamento(frase):
    return frase.lower()
    
#list comprehesion para iterar sobre todas as palavras.    
textos_tratados = [tratamento(frase) for frase in textos]

print("Esses são os textos em minúsculo: \n", textos_tratados)

#Criando uma string para conseguir usar .split() e separar as palavras.
super_string = " ".join(textos_tratados)
todas_palavras = super_string.split()

print("\n Essas são as palavras: \n", todas_palavras)

#Organizando as palavras
palavras_unicas = set(todas_palavras) #Deixando cada palavra única
palavras_unicas = list(palavras_unicas) #Voltando o string para lista para aplicar p sort
palavras_unicas.sort() #aplicando sort para organizar as palavras

print("\n Palavras únicas: \n", palavras_unicas)


bag_of_words = [] 

for frase in textos_tratados: 
    palavras_da_frase = frase.split() #Transformando a frase numa lista de palavras e guardando.
    vetor_frase = [] 
    for palavras_do_vocab in palavras_unicas:
        contagem = palavras_da_frase.count(palavras_do_vocab)
        vetor_frase.append(contagem)
    bag_of_words.append(vetor_frase)    

print("\nBag of Words na mão: \n", bag_of_words)


#### BAG OF WORD USANDO BIBLIOTECAS (SCIKIT LEARN) ####

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(textos)

matriz_bow = X.toarray()

print("-"*200)
print("Vocabulário do Scikit Learn: ")
print(vectorizer.get_feature_names_out())

print("\nMatriz Bag-of-Words:")
print(matriz_bow)