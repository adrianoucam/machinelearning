# Classificação de Câncer de Pele utilizando Deep Learning
<br>
Trabalho de Topicos avançados em Machine Learning para Saude  <br>
Professor : Flavio Luiz Seixas - fseixas@ic.uff.br <br>
Alunos : Adriano Lima e Souza -adrianoucam@gmail.com <br>
         Fernando Fernandes - fernando.fernandes2@gmai.com <br>
         <br>
## Motivação
Identificar utilizando Machine Learning com aprendizado profundo (Deep Learning) para classificar câncer de pele com base em imagens. <br>
<br>
### Exemplos de imagens
 <img src="../Cancer de Pele/23.jpg">
 <br> Cancer Benigno
<br>

<img src="../Cancer de Pele/6.jpg">
  <br> Cancer Maligno
<br>
 
 
O objetivo é criar um modelo de classificação de imagens para distinguir entre imagens benignas e malignas relacionadas ao câncer de pele. <br>

#Descrição do Dataset <br>
O dataset utilizado no projeto está disponível no Kaggle e pode ser encontrado no seguinte link: Skin Cancer: Malignant vs. Benign. <br>

![](https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign)

#Descrição dos Dados <br>
Os dados consistem em imagens de câncer de pele divididas em duas classes: maligna e benigna. As imagens estão organizadas em duas pastas, uma para o conjunto de treinamento e outra para o conjunto de teste. Cada pasta contém subpastas separadas para as classes maligna e benigna. <br>
<br>
<img src="../Cancer de Pele/train_grafico1.png">
  <br> TREINAMENTO - Imagens por tipo <br>
<br>
<img src="../Cancer de Pele/test_grafico1.png">
  <br> TESTE - Imagens por tipo <br>
  <br>
  <br>
  
#Etapas de Pré-processamento dos Dados <br>
## As imagens são pré-processadas de acordo com as seguintes etapas: <br>
<br>
<b>Normalização:</b> As imagens de treinamento e teste são normalizadas dividindo cada pixel pelo valor máximo (255) para que fiquem no intervalo [0, 1]. <br>

<b>Aumento de Dados:</b> As imagens de treinamento passam por um processo de aumento de dados (data augmentation) utilizando a classe ImageDataGenerator do Keras. <br>
Isso inclui <b>rotação, zoom e deslocamento horizontal </b> para aumentar a variabilidade do conjunto de treinamento. <br>

## Modelo de Aprendizado de Máquina <br>
O modelo de classificação de câncer de pele é construído utilizando o framework Keras com aprendizado profundo (Deep Learning). O modelo segue a seguinte arquitetura: <br>
<br>
<b> Camadas de Convolução 2D:</b>  As camadas de convolução 2D extraem características das imagens através de operações de convolução com filtros. <br>
<b> Camadas de Max Pooling:</b>  As camadas de max pooling reduzem a dimensionalidade das características selecionando os valores máximos em regiões específicas. <br>
<b> Camadas de Dropout:</b>  As camadas de dropout desativam aleatoriamente neurônios durante o treinamento para prevenir o overfitting. <br>
<b> Camadas Densas: </b> As camadas densas realizam a classificação final, sendo a última camada densa ativada pela função de ativação softmax para atribuir probabilidades às classes benigna e maligna. <br>
<br>
# Compilação e Treinamento do Modelo <br>
Durante o treinamento, o modelo é compilado com o otimizador Adam e a função de perda Sparse Categorical Crossentropy. As métricas de acurácia e perda são registradas para monitorar o desempenho do modelo durante o  <br> treinamento. O número de épocas e os callbacks para registro de métricas de treinamento são definidos para ajustar o modelo. <br>

## Método de Validação <br>
O desempenho do modelo é avaliado utilizando dados de validação separados do conjunto de treinamento. A acurácia e a perda são registradas para avaliar a capacidade de generalização do modelo.

## Resultados e Métricas de Desempenho <br>
Ao final do treinamento, os resultados são avaliados através de gráficos que mostram a acurácia e a perda do treinamento e validação em cada época. Essas métricas permitem avaliar a capacidade de aprendizado do modelo ao longo do treinamento. <br>

Execução 3 vezes (Epocas) <br>
 <img src="../Cancer de Pele/resultado2x.png">

 Execução 5 vezes (Epocas) <br>
 <img src="../Cancer de Pele/resultado5x.png">

 Execução 11 vezes (Epocas) <br>
 <img src="../Cancer de Pele/resultado11x.png">

 Execução 120 vezes (Epocas) <br>
 <img src="../Cancer de Pele/resultado120x.png">

 Execução 250 vezes (Epocas) <br>
 <img src="../Cancer de Pele/resultado251x.png">

## Conclusão
Através do desenvolvimento e treinamento de um modelo de classificação de câncer de pele utilizando deep learning, foi possível criar um sistema capaz de distinguir entre imagens benignas e malignas. O modelo pode ser utilizado como uma ferramenta de apoio para auxiliar profissionais da área médica na detecção precoce e diagnóstico do câncer de pele.<br>
<br>
# AUC 
O valor do AUC varia de 0,0 até 1,0 e o limiar entre a classe é 0,5. Ou seja, acima desse limite, o algoritmo classifica em uma classe e abaixo na outra classe. Quanto maior o AUC, melhor <br>
<br>
AUC: 0.7958333333333334
<br>
# RECALL 
O recall é o número de pessoas que o modelo identificou corretamente como tendo a doença dividido pelo número total de pessoas que realmente têm a doença nos seus dados. Ou seja, de todas as pessoas que ele poderia classificar como positivas, quantas ele acertou.<br>

# ROC

As curvas ROC são criadas plotando-se a sensibilidade (verdadeiro positivo) no eixo y contra 1 − especificidade (verdadeiro negativo) no eixo x para cada valor encontrado em uma amostra de indivíduos com e sem a doença.<br>
Para 120 epocas <br> 

 <img src="../Cancer de Pele/curva roc 120x.png">

# Recall <br>
Para 120 epocas <br>
[0.775      0.81666667]


# Precision Score
Para 120 epocas <br>

|[0.83532934 | 0.75153374] | <br>

|  	| precision 	| recall 	| f1-score 	| support 	|
|---	|---	|---	|---	|---	|
| 0 	| 0.84 	| 0.78 	| 0.80 	| 360 	|
| 1 	| 0.75 	| 0.82 	| 0.82 	| 300 	|
|  	|  	|  	|  	|  	|
   
   
# Acurácia
Acurácia: indica uma performance geral do modelo. <br>
Dentre todas as classificações, quantas o modelo classificou corretamente;  <br>
<br>Para 120 epocas <br>

|    accuracy |              |         |    0.79   |    660 |
|-------------|--------------|---------|-----------|--------| 
|   macro avg |      0.79    |   0.80  |    0.79   |    660 |
|weighted avg |       0.80   |   0.79  |    0.79   |    660 |


<br>
<br>
# Foi utilizado Python 3.11
Sobre os codigos fontes python <br>
## classifica1.py <br>
O arquivo criar os modelos de acordo com os parametros comentados acima  <br>
<br>
## cancer_teste.py 
Foi utilizada técnica de <b> transfer </b> learning para usar o aprendizado gravado <br>
O arquivo utiliza o modelo criado para classificar os arquivos da pasta escolhida - onde gera uma saida PREDS  [ 0 / 1 ]  <br>
Onde :<br>
0 seria Benigno<br>
1 seria Maligno<br>

# Exemplo abaixo da saida <br>
C://ML_SAUDE//cancer_pele//Photos-001//IMG_20230704_094559.jpg <br> 
Lendo o arquivo  C://ML_SAUDE//cancer_pele//Photos-001//IMG_20230704_094559.jpg <br>
Convertendo cor do arquivo  C://ML_SAUDE//cancer_pele//Photos-001//IMG_20230704_094559.jpg <br>
(1944, 2592, 3)

1/1 [==============================] - ETA: 0s <br>
1/1 [==============================] - 0s 75ms/step <br>
PREDS  [[1 0]] <br>
<b> [0] </b>

C://ML_SAUDE//cancer_pele//Photos-001//IMG_20230704_094610.jpg <br>
Lendo o arquivo  C://ML_SAUDE//cancer_pele//Photos-001//IMG_20230704_094610.jpg <br>
Convertendo cor do arquivo  C://ML_SAUDE//cancer_pele//Photos-001//IMG_20230704_094610.jpg <br>
(1944, 2592, 3) <br>

1/1 [==============================] - ETA: 0s <br>
1/1 [==============================] - 0s 61ms/step <br>
PREDS  [[0 1]] <br>
<b> [1] Maligno Maligno </b> <br>



