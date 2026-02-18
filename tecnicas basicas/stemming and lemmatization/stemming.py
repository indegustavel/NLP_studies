import nltk #Importando a biblioteca de NLP Toolkit
nltk.download('rslp')
from nltk.stem import RSLPStemmer #Importando a função de stemming 

st = RSLPStemmer()

palavras = ["programa", "programação", "programado", "programas", "programador", "ir", "fui", "ser", "iremos", "iríamos"]

#list comprehension: [funcao(item) para cada item na lista]
palavras_stem = [st.stem(palavra) for palavra in palavras]

print("Stemming das palavras: ", palavras_stem)
