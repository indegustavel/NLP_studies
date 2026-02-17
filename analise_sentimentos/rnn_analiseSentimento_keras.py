# RNN básica utilizando Keras para aplicar análise de sentimentos

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Limpa avisos chatos do TensorFlow

import keras
from keras import layers

import numpy as np

from gerar_dataset import gerar_dataset_gigante


max_palavras = 20000 #tamanho do vocabulário
tamanho_sequencia = 200 #quantos termos ler por frase

print("Gerando dataset massivo...")

textos_treino, labels = gerar_dataset_gigante(n_amostras=50000)

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

    # transformando números em vetores densos
    layers.Embedding(input_dim = max_palavras, output_dim = 64),

    # A RNN recebe a saída do Embedding
    layers.SimpleRNN(units=64),

    # Sigmoid é para classificação binária. 
    layers.Dense(1, activation='sigmoid')

])

#compilando o modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


print("\nIniciando o treino")

#treinando rede neural com nosso dataset.
# Convertendo para numpy array com dtype='object' para evitar problemas com strings Unicode longas
# O Keras precisa de numpy array para usar validation_split
textos_array = np.array([str(texto) for texto in textos_treino], dtype='object')
model.fit(textos_array, labels, epochs=1, batch_size=64, validation_split=0.2, verbose=1)

#aplicando no novo texto
frase_nova = np.array(["esse filme é ruim"], dtype = "object")

previsao = model.predict(frase_nova)

print(previsao[0][0])
