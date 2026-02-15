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

O algoritmo word2vec detecta que há uma relação semântica forte entre rei e rainha, porque o contexto em que ambos são usados são semelhantes. (Diferente de computador, por exemplo, visto que é raro você ler que um *computador* sentou no trono, rsss)


Observação: Como a rede sabe se a palavra que ela previu é um acerto ou um erro? Por meio da Loss Function, Backpropagation e Gradiente Descendente. 

De forma simples, após a rede prever um resultado, o algoritmo compara com o resultado que seria o correto e faz um cálculo da diferença dos vetores da palavra correta e da palavra que a rede previu (Loss Function). Com isso, através do backpropagation, o erro volta ao começo da rede neural e avisa o quão errado aquele resultado está. Após esse processo, a rede sabe onde errou e utiliza o gradiente descendente para ajustar os pesos em uma direção onde o erro diminua e o vetor da palavra prevista seja mais próximo ao vetor da palavra correta.

### GloVe

GloVe é uma forma de gerar word embeddings usando matriz de coocorrência entre as palavras. A ideia é que a relação entre duas palavras pode ser deduzida de acordo com a probabilidade de as duas serem semelhantes a outra palavra "alvo".

A primeira parte é a criação de uma matriz global de coocorrência, com base no nosso dataset. Nela é enumerado quantas vezes todas as palavras aparecem pertos das outras palavras do dataset.

Imagine que treinamos um algoritmo GloVe com um dataset sobre física. A matriz de coocorrência das palavras é esse abaixo.

| Palavra       | gelo | vapor | sólido | gás | moda | Total|
|               |      |       |        |     |      |      |
| **gelo**      | 0    | 10    | 80     | 5   | 1    |  96  |
| **vapor**     | 10   | 0     | 2      | 75  | 1    |  88  |
| **sólido**    | 80   | 2     | 0      | 10  | 0    |  92  |
| **gás**       | 5    | 75    | 10     | 0   | 0    |  90  |
| **moda**      | 1    | 1     | 0      | 0   | 0    |   2  |

Essa matriz indica quantas vezes tais palavras ocorreram em contextos semelhantes, com base em todo dataset. Vemos que a relação entre "gelo" e "sólido" é alta, enquanto a relação entre "moda" e "vapor" é baixa. Porém, o modelo não observe apenas esses valores, ele busca probabilidades condicionais.

A probabilidade condicional P(K/i) responde a pergunta: Dada a palavra i, qual é a chance da palavra k aparecer no contexto dela?

**A fórmula é simples: Xik/Xi.**

Xik é o número de vezes que a palavra K aparece perto da palavra i, e Xi é o total de vezes que qualquer palavra aparece perto de i (coluna "Total").

Vamos para um exemplo prático: Dado a palavra sólido, vamos compara-las com "gelo" e "vapor".

**Para "gelo": P(sólido/gelo) = 80/96 ≈ 0,8333**

80 é a quantidade de vezes que sólido aparece perto de gelo (pbserve na matriz, linha de "sólido" com a coluna de "gelo", "Xik é o número de vezes que a palavra K aparece perto da palavra i")
96 é a quantidade total de vezes que gelo aparece (observe na matriz, linha de "gelo" com coluna de "total")

**Para "vapor": P(sólido/vapor) = 2/88 ≈ 0,022**

2 é a quantidade de vezes que sólido aparece perto de vapor (observe na matriz, linha de "sólido" com a coluna de "vapor", "Xik é o número de vezes que a palavra K aparece perto da palavra i")
88 é a quantidade total de vezes que vapor aparece. (olhe na matriz, linha de "vapor" com coluna de "total")

Com isso, pegamos a probabilidade condicional, e agora?
o GloVe não olha esses números isolados, ele faz a razão entre essas probabilidades.

**P(sólido/gelo) / P(sólido/vapor) = 0,8333/0,022 = 37,87**

Como o resultado é muito grande, indica que a palavra sólido é muito mais ligada ao gelo do que ao vapor.

Caso inverso: Se fizéssemos P(gás/gelo) / P (gás/vapor), veríamos que seria uma razão de um valor pequeno e um valor grande, o que resultaria em um número próximo de 0. Isso implicaria que a palavra "vapor" está no contexto de "gás", mas não de gelo.

Caso neutro: Se fizéssemo P(moda/gelo) / P(moda/vapor), veríamos que seria uma razão de números pequenos e semelhantes, resultando em algo próximo de 1. Isso significa quem ambos os termos não tem relação com a palavra alvo. Se a palavra fosse "água", ambos as probabilidades condicionais seriam altas, fazendo com que a divisão também seja próximo de 1, o que não serviria para diferenciar os dois conceitos.

