import spacy

nlp = spacy.load("pt_core_news_sm")

doc = nlp("Viajei aos Estados Unidos e encontrei o Donald Trump")

tokens_filtrados = [token.text for token in doc if not token.is_stop]

print('Frase normal', doc)
print('Tokens filtrados:', tokens_filtrados)

# Viajei aos Estados Unidos e encontrei o Donald Trump