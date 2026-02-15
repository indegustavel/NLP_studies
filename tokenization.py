import nltk

text = "olá, eu sou o Gustavo. Estou estudando NLP e tokenização é uma parte importante disso."

tokens = nltk.word_tokenize(text)
print(tokens)