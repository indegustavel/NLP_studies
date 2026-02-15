import spacy

#carregar o modelo em PTBR
nlp = spacy.load("pt_core_news_sm")

#definir lista de palavras
palavras = ["programa", "programação", "programado", "programas", "programador", "fui", "foi", "ir", "irei", "iremos"]

#definir tudo em uma frase para dar "contexto"
texto = " ".join(palavras)

doc = nlp(texto)

#extrai o "lema" de cada token

palavras_lema = [token.lemma_ for token in doc]

print(palavras_lema)