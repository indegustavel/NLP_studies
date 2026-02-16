import spacy

#Carregar o modelo (o 'md' contém ~20 mil vetores únicos para 685k palavras)
nlp = spacy.load("en_core_web_md")

#processar uma frase
doc = nlp("I preffer coffe over tea.")

#Ver valores individuais
for token in doc:
    #mostra a palavra, seu vetor no modelo e a norma do vetor (tamanho)
    print(f"{token.text:10} | Vetor: {token.has_vector:5} | L2 Norm: {token.vector_norm:.2f}")

#definindo palavras para comparação
word1 = nlp("banana")
word2 = nlp("apple")
word3 = nlp("car")

#imprimindo comparações
print(f"\nSimilaridade banana-laranja: {word1.similarity(word2):.4f}")
print(f"Similaridade banana-carro: {word1.similarity(word3):.4f}")
