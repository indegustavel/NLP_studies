# RNN básica utilizando Keras para aplicar análise de sentimentos

import keras
from keras import layers

import numpy as np

#nosso dataset (keras 3 não aceita lista, temos que usar o np.array por isso)
textos_treino = np.array([
    "este filme é excelente e muito bom",
    "que porcaria de roteiro horrível",
    "amei cada segundo da atuação",
    "perdi meu tempo assistindo isso",
    "favorito da vida recomendo muito",
    "odiei a história e os atores",
    "é o melhor filme que já vi na minha vida",
    "que filme ridículo",
    "achei mediano",
    "filme muito ruim",
    "gostei bastante"
], dtype=object)

#
labels = np.array([1, 0, 1, 0, 1, 0, 1, 0, 0.5, 0 , 1])

max_palavras = 1000 #tamanho do vocabulário
tamanho_sequencia = 10 #quantos termos ler por frase

# Aqui, abstraimos a normalização, padronização, tokenização e indexação do dataset. 
#é interessante pontuar que, como as frases tem tamanhos diferentes, essa camada preenche com zeros ( 0 ) as frases para que todos inputs tenham o mesmo tamanho (tamanho_sequencia)

vectorizer = layers.TextVectorization(
    max_tokens = max_palavras,
    output_mode = 'int',
    output_sequence_length = tamanho_sequencia
)

vectorizer.adapt(textos_treino)

#Definindo arquitetura
model = keras.Sequential([

    #expondo ao keras o formato do input (string)
    layers.Input(shape=(1,), dtype = "string"), 

    #transformando string em números
    vectorizer,

    # Embedding: Transforma inteiros em vetores de 32 posições
    layers.Embedding(input_dim=1000, output_dim=32),

    # A RNN recebe a saída do Embedding
    layers.SimpleRNN(units=64),

    # Sigmoid é para classificação binária. 
    layers.Dense(1, activation='sigmoid')

])

#compilando o modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#treinando rede neural com nosso dataset
model.fit(textos_treino, labels, epochs=10)

#aplicando no novo texto
frase_nova = np.array(["gostei"], dtype = "object")

previsao = model.predict(frase_nova)

print(f"\nResultado da previsão (0 a 1): {previsao[0][0]:.4f}")
print("Sentimento: Positivo" if previsao > 0.5 else "Sentimento: Negativo")
