#### Modelo GloVe com nível alto de abstração. Hiperparâmetros já definidos pelo SpaCy

import spacy
import requests

#Carregar o modelo (o 'md' contém ~20 mil vetores únicos para 685k palavras)
nlp = spacy.load("pt_core_news_sm")

print("Baixando Dom Casmurro")
url = "https://www.gutenberg.org/cache/epub/55752/pg55752.txt"
resposta = requests.get(url)
texto = resposta.text

# 3. Vamos analisar os personagens principais
bentinho = nlp("Bentinho")
capitu = nlp("Capitu")
escobar = nlp("Escobar")
casmurro = nlp("casmurro")

print("-" * 30)
print(f"Similaridade Bentinho - Capitu: {bentinho.similarity(capitu):.4f}")
print(f"Similaridade Bentinho - Escobar: {bentinho.similarity(escobar):.4f}")
print(f"Similaridade Capitu - Escobar: {capitu.similarity(escobar):.4f}")
print(f"Similaridade Bentinho - Casmurro: {bentinho.similarity(casmurro):.4f}")

# 4. Buscando palavras similares a "ciúme" no contexto do livro
# (Isso mostra o poder dos vetores para captar semântica)
conceito = nlp("ciúme")
print("-" * 30)
print(f"O quão próximo 'Capitu' está de '{conceito.text}': {capitu.similarity(conceito):.4f}")
print(f"O quão próximo 'Bentinho' está de '{conceito.text}': {bentinho.similarity(conceito):.4f}")