import spacy

nlp = spacy.load("pt_core_news_sm")

doc = nlp("olá, eu sou o Gustavo. Agora estou aprendendo sobre Stop Words.")

tokens_filtrados = [token.text for token in doc if not token.is_stop]

print('Frase normal', doc)
print('Tokens filtrados:', tokens_filtrados)