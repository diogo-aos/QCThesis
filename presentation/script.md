# After comments

## Cover slide
time: 11s

Bom dia! Eu sou o Alferes Diogo Silva e vou apresentar o trabalho realizado no âmbito da minha dissertação de mestrado com o tema "Efficient Evidence Accumulation Clustering for large datasets or big data".

## Contents
time: 23s

Vou começar por dar algum contexto: o que é clustering e explicar o algoritmo que optimizei. Tendo isto em conta, os objectivos apresentados tornarse-ão mais claros.

Depois vou passar para a descrição da solução proposta para a optimização do algoritmo, dando êmfase a cada uma das suas partes.

Finalmente, passo então para os resultados da optimização, desde a validação até a um estudo mais extenso nos parâmetros do algoritmo.

## What is clustering (1)
time: 35s

Começemos então pelo conceito fundamental à dissertação: o que é o _clustering_? O objectivo do clustering é descobrir os agrupamento naturais de um conjunto de pontos ou objectos.

Clustering é uma ferramenta não supervisionada, ou seja, trabalha apenas com o conjunto de dados sem ajuda de informação extra. É um ferramenta bastante usada para análise exploratória e útil quando temos pouca ou nenhuma informação extra mas queremos descobrir qual é a estrutura inerente aos dados.

Existem uma grande panóplia de algoritmos de clustering. Típicamente, diferentes algoritmos dão bons resultados em conjuntos de dados com determinadas caracteristicas, dimensão, complexidade, etc.

## What is clustering (2)
time: 51s

Nesta figura, isso torna-se claro. Na figura (a) está um conjunto de dados. Para um humano é trivial identificar grupos nesta figura, mas para um computador isto é uma tarefa difícil. Contudo, um humano teria dificuldades em analisar um conjunto de dados grande ou conjuntos de dados com mais de 2 ou 3 dimensões.

As restantes figuras são agrupamentos resultantes de diferentes algoritmos (neste caso _K-Médias_ e _Single-Link_) ou o mesmo algoritmo com parâmetros de inicialização diferentes. Diferentes cores representam grupos diferentes.

Por simples observação destas figuras, torna-se óbvio que diferentes algoritmos ou parâmetros de inicialização dão resultados considerávelmente diferentes.

Idealmente, o que se queria era ter um algoritmo capaz de oferecer bons resultados em qualquer situação. Embora seja uma tarefa extremamente difícil, existem algoritmos que endereçam esse problema, como o _Evidence Accumulation Clustering_, ou EAC.

## Evidence Accumulation Clustering (1)
time: 28s

O EAC é um algoritmo moderno e robusto, tentando dar bons resultados num grande espectro de conjunto de dados e ser resistente a _outliers_. É, também, um método em _ensemble_, o que significa que usa vários agrupamentos (ou partições) diferentes e combina-os numa partição de melhor qualidade que qualquer uma das outras. Estas partições podem vir de algoritmos diferentes, do mesmo algoritmo com inicializações diferentes ou até de diferentes perspectivas do conjunto de dados.

## EAC: Production (2)
time: 17s

EAC é um método de 3 partes. A primeira parte, __produção__, começa por produzir um conjunto (ou _ensemble_) de várias partições distintas. Neste exemplo, as figuras (b) e (c) são duas partições resultantes da aplicação do método K-Médias no conjunto de dados da figura (a).

## EAC: Combination (3)
time: 45s

A fase seguinte é a de __combinação__, onde as partições da ensemble são combinadas numa única estrutura com informação nova - uma nova perspectiva do conjunto de dados. No caso do EAC, usa-se o conceito de co-associações entre os objectos do conjunto de dados. Essencialmente, o valor da co-associação entre dois objectos é o número de vezes que aparecem no mesmo grupo na ensemble. Por exemplo, se o objecto 1 e 20 são agrupados no mesmo grupo 50 vezes, então o valor da associação entre estes dois objectos é 50. 

As figuras (c) e (d) mostram matrizes de coassociação das partições nas figuras (a) e (b). A figura (e) mostra a matriz de co-associação final - a soma de todas as co-associações em toda a ensemble.

Esta etapa de combinação estão na essência do EAC.

## EAC: Recovery (4)
time: 22s

Finalmente, a partição final é extraida na fase de __recuperação__. É necessário extrair a partição final desta nova representação em co-associações. A figura (b) apresenta uma partição extraida da matriz de co-associações na figura (a).
aud


É de referir que a primeira e última etapas do EAC podem usar qualquer algoritmo.

## Goals
time: 40s

A robustez do EAC vem com um preço: maior complexidade computacional. O que significa que também é mais dificil a aplicação deste algoritmo a conjuntos de dados grandes.

É neste contexto que os objectivos da dissertação se inserem. O principal objectivo é possibilidade de aplicação do EAC a grandes conjuntos de dados. Para isso, é necessário arranjar estratégias para reduzir a complexidade computacional do EAC. Tendo uma versão optimizada do EAC, é necessário validá-la com a versão original. Sendo equivalentes, o EAC optimizado é aplicado a conjuntos de dados grandes para analisar como é que diferentes parâmetros se comportam à medida que se o conjunto de dados crece.
E, por fim, aplicação do EAC a conjuntos de dados grandes reais.

Devo referir que a optimização do EAC foi feita de modo a ser possível analisar conjuntos de dados grandes numa estação de trabalho (não requerendo servidores pesados ou clusters de máquinas).


## Proposed Solution: overview (1)
time: s

Passemos agora para a solução proposta para optimização do EAC. Cada uma das fases do EAC foi submetida a optimização. Nesta figura estão os vários candidatos a optimização nas diferentes fases. Na fase de produção consideraram-se quantum clustering, algoritmos de clustering inspirados em analogias de mecânica quântica, e uma versão paralelizada em GPU do K-Médias.

Na fase de combinação explorou-se um método de rápida construção de matrizes esparsas.

Na fase final, consideraram-se candidatos em GPU, CPU e baseados em disco.

## Proposed Solution: final (2)
time: s

Destes, o quantum clustering foi descartado por ser demasiado lento e o candidato em GPU da última fase demonstrou ser lento no contexto do EAC e díficil de escalar.

## Optimizing production of ensemble (1)
time: s

Passemos então para a optimização da produção. Usou-se o K-Médias, um algoritmo simples e um dos mais conhecidos algortmos de clustering. Além disso, este algoritmo já foi usado para a produção de ensembles em EAC com sucesso anteriormente.

A primeira fase do K-Médias, a de _labelling_, começa por encontrar qual o centroide (o representante de um cluster) mais perto de cada ponto. Na fase seguinte, a de _update_, as centroides são actualizadas com os novos valores - a média dos pontos correspondentes a cada centroide.

O parâmetro de inicialização do K-Médias é o número de clusters que a partição final terá e, tipicamente, os centroides iniciais são aleatórios. No caso do EAC, o K-Médias é executado várias vezes e o número de clusters varia entre um intervalo [Kmin, Kmax].

A fase de labelling do K-Médias é inerentemente paralela, uma vez que o cálculo do centroide mais perto de um ponto é independente do cálculo dos restantes pontos. Foi nesta óptica que se procedeu à optimização da fase de produção.

## GPU (2)
time: s

Antes de descrever a paralelização do K-Médias, vou dar um pequena introdução a computação em GPUs. Eu usei CUDA, a plataforma de computação em GPU da NVIDIA. Uma GPU tem dezenas, centenas ou até milhares de processadores simples, representando uma capacidade de paralelismo muito maior que um CPU tradicional. Cada processador corre uma _thread_ de execução com o mesmo programa e cada thread processa uma parte dos dados. As threads têm acesso a memória partilhada que permite alguma comunição entre elas.

Isto é uma explicação muito simplificada mas é suficiente para perceber porque é que é possível ter melhor desempenho que em CPU.

## GPU K-Médias (3)
time: s

Como é que o K-Médias foi paralelizado? O CPU começa por transferir os dados e os centroides para a GPU. Cada thread na GPU vai calcular a centroide mais perto de cada ponto e guardar o resultado num vector. Esse vector é depois transferido para a memória principal, o CPU é responsável pela fase de update e envia os novos centroides para a GPU. Os passos 2-4 são repetidos enquanto não se atingir o critério de paragem, que no nosso caso é o número de iterações.

## Optimizing combination (1)
time: s

Na fase de combinação, o grande desafio é a complexidade quadrática da matriz de coassociações. Contudo, esta matriz tem duas propriedades interessantes: é simétrica, o que quer dizer que pode-se guardar apenas metade da matriz sem perda de informação; e é esparsa    , como se pode ver na figura onde o espaço em branco são zeros da matriz.

A abordagem escolhida para endereçar este problema foi explorar a natureza esparsa da matriz, uma abordagem existente na literatura que foi extendida por esta dissertação.

## Optimizing combination (2)
time: s

Comecei por avaliar implementações de diferentes formatos de matrizes esparsas existentes nas bibliotecas de computação cientifica e númerica. No entanto, estas implementações ou consumiam demasiada memória ou eram extremamente lentas.

O grande problema é a construção de uma matriz esparsa: a memória tem de ser alocada à medida que se adicionam novos elementos e, além disso, indexar uma matriz esparsa é um processo pesado que contrasta com o que acontece com uma matriz tradicional.

Isto levou-me a procurar soluções eficientes para construir uma matriz esparsa na literatura. Não encontrando nada de relevante, desenvolvi a minha própria estratégia, o EAC CSR: um compromisso entre eficiência e memória utilizada. Na verdade, ela é mais rápida que qualquer uma das implementações avaliadas.

A minha estratégia foi baseada no format CSR, o mais modesto em memória utilizada. Essencialmente, começa-se por fazer uma estimativa de qual o número máximo de associações que qualquer ponto terá. Tipicamente, tenta-se que esta estimativa seja para _worst case scenario_. Depois, aloca-se memória suficiente que permita que todos os pontos possam ter esse número de associações. Com toda a memória pré-allocada, a construção da matriz pode ser feita de forma mais eficiente.

## Optimizing combination (3)
time: s

Na figura (a), a preto, está o número de associações existentes em cada ponto. A linha a vermelho é o número máximo de associações possível. Neste exemplo, isto significa que só foi reservado espaço para 1.4% das associações comparado com a complexidade quadrática.

Na figura (b), tomou-se proveito da propriedade de simetria, e apenas metade da matriz foi preenchida, razão pela qual o numero de associações decresce. Isto significa que pode-se reduzir a quantidade de memória pré-alocada. Como o número de associações decresce à medida que se percorre a matriz, fez-se um "corte" linear apresentado pela linha a verde. Isto permitiu uma redução ainda maior na memória pré-alocada.

## Optimizing final clustering
time: s

Na fase final, usou-se o Single-Link, que já foi utilizado com EAC anteriormente com sucesso.

O Single-Link é um algoritmo hierárquico, uma vez que contrói uma hierarquia com os pontos do conjunto de dados, e aglomerativo, uma vez que começa por assumir que todos os pontos são clusters diferentes e, em cada iteração, une os dois clusters mais próximos até que todos os clusters sejam unidos.

O Single-Link opera sobre uma matriz de proximidade entre os pontos do conjunto de dados, o que, no fundo, é o que a matriz de co-associações é.

A chave para optimizar a fase de recuperação é a equivalência entre o Single-Link e o problema de Minimum Spanning Tree em grafos.

(Num grafo em que os nós são os pontos do conjunto de dados e as arestas que os ligam são as co-associações, o Minimum Spanning Tree é o caminho mais curto que liga todos pontos do conjunto de dados.)

Tipicamente, o Single-Link processa todos os elementos na matriz de proximidade, mas, devido a esta equivalência, é possível processar apenas os elementos não nulos. Como já se viu, a matriz de co-associações é bastante esparsa o que se traduz em muito menos processamento.

Uma das soluções foi usar esta equivalência. A outra foi implementar uma variante capaz de trabalhar sobre matrizes de co-associações guardadas em disco, o que apesar de ser muito mais lento, permite a aplicação do EAC a problemas de muito maior dimensão.

## Results: validation
time: s

Passemos agora para os resultados. Vou começar por mostrar que a versão optimizada é equivalente à original. Esta tabela mostra a diferença de precisão entre a versão optimizada e a original para diversos datasets de pequena dimensão. Os valores são tão pequenos que podem ser desprezados. E a diferença existe, basicamente, porque a versão optimizada foi implementada em Python e a original em Matlab.

## Results: speed-up over original version
time: s

Apesar de não ser o principal objectivo da dissertação, a versão optimizada é bastante mais rápida que a original em datasets de pequena dimensão, como se pode ver nesta tabela. Existe speedup considerável em todas as fases e em todos os datasets.

## Results: GPU K-Médias (1)
time: s

Vou agora falar do speed-up apenas do K-Médias em GPU. Esta figura mostra o speed-up obtido para problemas bidimensionais e as diferentes linhas correspondem a diferentes número de clusters. Pode-se ver que o speedup aumenta com o número de clusters e com número de pontos do problema, uma vez que existe maior capacidade de paralelismo. 

## Results: GPU K-Médias (2)
time: s

Contudo, o speed-up diminui com o aumento do número de dimensões. Nesta caso, temos 200 dimensões e o speed-up não passa de 3. Os resultados aqui apresentados não espelham o que existe na literatura. Estou convencido que a razão para isto é a não utilização de todas as optimizações possíveis. Ainda assim, registou-se sempre speed-up. 

(Além disso, no EAC, o K-Médias é executado várias vezes sobre os mesmos dados o que permite que exista menos transferência de dados do que o que acontece na experiência destes resultados.)

## Results: Kmin rules
time: s

Vou passar agora para os resultados de um estudo mais extensivo, cujo objectivo foi analisar como é que os parâmetros do EAC variam com o aumento do conjunto de dados.

Forma usadas 4 regras para o intervalo [Kmin,Kmax] da produção da ensemble. A primeira, da raiz, já existia na literatura e as restantes são regras novas que nasceram das experiências que foram sendo feitas.

As duas últimas regras controlam o número de pontos em cada cluster de qualquer partição (em média). Na última regra, a _sk300_, o número de pontos está fixo em 300.

Eu vou-me concentrar nas tendências dos resultados e não em valores especificos. As mesmas tendências vão aparecer em diferentes parâmetros e, por isso, eu vou andar a um passo mais rápido, apresentando as mesmas tendências em parâmetros diferentes.

## Results: Kmin evolution
time: s

Vou começar pela evolução do Kmin. Podemos ver que o Kmin cresce com o número de pontos em todas as regras. 3 das regras estão paralelas enquando a _sk300_ começa por ser a regra com menor Kmin e acaba com o maior.

## Results: Production time per rule
time: s

A mesma tendência é observada com o tempo de produção da ensemble. A razão para isto é que um maior número de clusters corresponde a uma computação mais pesada do K-Médias.

## Results: Combination time per rule
time: s

No entanto, o inverso acontece com o tempo de combinação. Aqui a razão tem a haver com o efeito do Kmin no número de associações. Um Kmin maior significa que os clusters são mais compactos, o que por conseguinte, significa que cada ponto irá ter menos associações. Como existem menos associações, é necessário menos tempo para as processar.

## Results: SL times per rule
time: s

O mesmo acontece com a fase final pela mesma razão - não existem tantas associações para processar. Com estes três gráficos podemos já ver que existe um contraste no efeito que o Kmin tem nos tempos: acelerar a fase de produção atrasa as outras e vice versa.

## Results: Combination time per matrix format
time: s

Observemos agora o tempo de combinação nos diferentes formatos utilizados. Estes formatos são: uma matriz tradicional completa ou condensada (que aqui significa que apenas metade da matriz é preenchida) e matriz esparsa completa ou condensada. Como seria de esperar, a matriz tradicional é a mais rápida, especialmente quando é condensada. As matrizes esparsas são cerca de 10 vezes mais lentas.

## Results: SL times per matrix format
time: s

Uma análise semelhante é feita ao Single-Link. As diferentes abordagens são:
 - Single-Link baseado em MST usando uma matriz esparsa completa ou matriz condensada, que é duas vezes mais rápida;
 - SLINK, a implementação "clássica" mais rápida que trabalha sobre uma matriz tradicional, é mais lento que estas duas;
 - e Single-Link baseado em MST utilizando o disco, que é significativamente mais lenta que qualquer uma das outras versões.

## Results: Total time main memory
time: s

Observemos agora o tempo total do EAC usando matriz esparsa em memória principal. A tendência anterior não é clara aqui. A razão para isto parece ser um equilíbrio entre acelerar a primeira fase ou as duas últimas nas diferentes regras para conjuntos de dados grandes, com a excepção da regra da _raiz_ que acaba por ser mais lenta que todas as outras.

## Results: Total time disk
time: s

O mesmo não se constata quando se utiliza o Single-Link baseado em disco. Aqui, o tempo do Single-Link é tão lento comparado com o das outras fases, que essa tendência domina o tempo total.

## Results: Association density
time: s

Já apresentei alguns resultados relacionados o tempo computacional; agora vou falar da outra face do problema - a memória utilizada. Nesta figura está representada a densidade de associações relativa a _n^2_ (uma matriz quadrada). Podemos ver para conjuntos de dados muito grandes, a regra _sk300_ chega a ter uma densidade inferior a 0.1%. Também é interessante notar que a tendência aqui é a inversa à tendência da evolução do Kmin, o que quer dizer que o Kmin tem uma influência grande na esparsidade da matriz de co-associações.

## Results: Memory density
time: s

No entanto, a memória utilizada com uma matriz de co-associações não é apenas o valor das associações. É necessário uma estrutura de dados para suportar a matriz esparsa e isso vem com um custo. Esta figura mostra a memória utilizada para os diferentes formatos para a regra _sk300_ (a que produziu maior esparsidade de associações em problemas grandes) relativamente a n^2. A matriz esparsa condensada com um corte linear apresenta a menor utilização de memória, como seria de esperar, mas a "esparsidade" de memória não é tanta como no caso das associações (que chegava a 0.1%)

## Conclusions
time: s

Uma das lições aprendidas foi a dicotomia entre utilizar mais memória ou ser mais rápido. Isto apareceu várias vezes nas diferentes abordagens de optimização e nem sempre existe uma resposta certa: cada caso é um caso.

Outra impressão com que fiquei é que a GPU é um recurso pouco utilizado. Actualmente, GPUs são extremamente abundantes. Ainda assim, embora ofereçam uma grande capacidade de paralelismo, as bibliotecas de computação númerica, cientifica e as de Aprendizagem Automática não apreveitam esta capacidade.

Resumindo, o EAC foi escalado com sucesso:
 - Numa primeira fase usou-se uma versão paralela do K-Menas em GPUs;
 - A complexidade quadrática da matriz de co-associações foi endereçada com matrizes esparsas e uma nova estratégia para as contruir de forma eficiente;
 - Na fase final tomou-se partido da equivalência entre o Single-Link e a Minimum Spanning Tree para processar apenas associações que realmente existem; e,
 - usar o disco para guardar matrizes muito grandes e para operações muito pesadas em memória.

As principais contribuições foram:
 - Aumento do espectro de aplicabilidade do EAC - testaram-se datasets mais de 10 vezes maiores que o que anteriormente era possível e ainda era possível maior nas máquinas disponiveis;
 - Uma nova forma eficiente de contruir a matriz esparsa
 - Novas regras do Kmin para optimizar a esparsidade (e também uma compreensão maior de como é que estas regras afectam diferentes parâmetros do algoritmo)

Devo dizer ainda que foram identificadas bastantes direções que se podem explorar para optimizar para optimizar ainda mais.

Finalmente, quero só referir que uma selação do trabalho realizado foi compilado num artigo e submetido a uma conferência da área (Internation Conference on Pattern Recognition Applications and Methods).
