# Word Embeddings

Word embeddings são representações númericas de palavras em um espaço vetorial, que visa capturar significado e as realções entre elas.

Nesse repositório, vou tratar das 4 técnicas padrões de word embeddings: Bag of Words (BoW), TF-IDF, Word2Vec e GloVe.

## O que é cada uma delas?

### Bag Of Words

É a transformação do texto em número baseado na presença, contagem ou frequência de cada palavra no texto. 

Esse algoritmo é feito em 3 etapas: Coleta de dados, criação do vocabulário e criação dos vetores de documentos.

### TF-IDF

TF-IDF (Term Frequency-Inverse document frequency) é uma medida estatística que visa ser um indicativo de relevância de um termo no processo de mineração de palavras de um texto.

Seu algoritmo funciona tentando entender a relevância de uma palavra no contexto, sem considerar apenas a quantidade de vezes que ela aparece, mas também o quão rara ela é em um determinado conjunto de documentos. Ou seja, palavras como "e", "a", "ou" e "esta" tem uma relevância baixa mesmo que muito repetidos, porque aparecem bastante em quaisquer documentos. Isso valoriza palavras-chaves e descarta termos comuns na sua busca. TF -> Frequência dos Termos; IDF -> Frequência inversa dos documentos. Quanto mais seu termo aparece no seu documento e menos em outros documentos, mais relevante ele é. 

### Word2Vec

Word2Vec é um algoritmo que cria word embeddings a partir da aplicação de redes neurais, atribuindo valor semântico às palavras analisadas de acordo com o contexto.

A rede utilizada normalmente pode ter 2 estruturas diferentes: CBOW (Continuous Bag of Words) e o Skip-Gram. 

O CBOW funciona tentando prever a palavra central com base no contexto (palavras em volta).
Exemplo: ("O _____ está cheio hoje"). Aqui, a rede neural vai tentar prever se a palavra a ser completada é "ônibus", "anel", "copo", etc.

A Skip-gram por sua vez, funciona ao contrário, como se estivesse completando a frase ou desenvolvendo um contexto 
Exemplo: ("__ amo ____"). A rede neural tenta adivinhar o que colocar em volta para criar uma frase com sentido. Ela pode tentar palavras como "eu" e "você" e verificar que acertou ou pode tentar palavras como "céu" e "garrafa" e errar feio.

Com isso, a rede neural, normalmente treinada com quantidades massivas de dados, consegue estabelecer um valor semântico a cada palavra, de acordo com o contexto que ela normalmente vê essa palavra.

-> O *Rei* sentou no trono.
-> A *Rainha" sentou no trono.

O algoritmo word2vec detecta que há uma relação semântica forte entre rei e rainha, porque o contexto em que ambos são usados são iguais. (Diferente de computador, por exemplo, visto que é raro você ler que um *computador* sentou no trono, rsss)


Observação: Como a rede sabe se a palavra que ela previu é um acerto ou um erro? Por meio da Loss Function, Backpropagation e Gradiente Descendente. 

De forma simples, após a rede prever um resultado, o algoritmo compara com o resultado que seria o correto e faz um cálculo da diferença dos vetores da palavra correta e da palavra que a rede previu (Loss Function). Com isso, através do backpropagation, o erro volta ao começo da rede neural e avisa o quão errado aquele resultado está. Após esse processo, a rede sabe onde errou e utiliza o gradiente descendente para ajustar os pesos em uma direção onde o erro diminua e o vetor da palavra prevista seja mais próximo ao vetor da palavra correta.
