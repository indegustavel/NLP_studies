import nltk
from nltk.tokenize import LegalitySyllableTokenizer
from nltk.corpus import words

silaba_tokenizer = LegalitySyllableTokenizer(words.words())

text = "olá, eu sou o Gustavo. Estou estudando NLP e tokenização é uma parte importante disso."

tokens_palavras = nltk.word_tokenize(text)
tokens_silabas = silaba_tokenizer.tokenize(text)

print("Tokenização por palavra: ", tokens_palavras)
print("Tokenização por sílaba: ", tokens_silabas)
