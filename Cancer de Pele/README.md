# Classificação de Câncer de Pele utilizando Deep Learning
## Motivação
Identificar utilizando Machine Learning com aprendizado profundo (Deep Learning) para classificar câncer de pele com base em imagens.

O objetivo é criar um modelo de classificação de imagens para distinguir entre imagens benignas e malignas relacionadas ao câncer de pele.

#Descrição do Dataset
O dataset utilizado no projeto está disponível no Kaggle e pode ser encontrado no seguinte link: Skin Cancer: Malignant vs. Benign.

#Descrição dos Dados
Os dados consistem em imagens de câncer de pele divididas em duas classes: maligna e benigna. As imagens estão organizadas em duas pastas, uma para o conjunto de treinamento e outra para o conjunto de teste. Cada pasta contém subpastas separadas para as classes maligna e benigna.

#Etapas de Pré-processamento dos Dados
## As imagens são pré-processadas de acordo com as seguintes etapas:

Normalização: As imagens de treinamento e teste são normalizadas dividindo cada pixel pelo valor máximo (255) para que fiquem no intervalo [0, 1].
Aumento de Dados: As imagens de treinamento passam por um processo de aumento de dados (data augmentation) utilizando a classe ImageDataGenerator do Keras. Isso inclui rotação, zoom e deslocamento horizontal para aumentar a variabilidade do conjunto de treinamento.

## Modelo de Aprendizado de Máquina
O modelo de classificação de câncer de pele é construído utilizando o framework Keras com aprendizado profundo (Deep Learning). O modelo segue a seguinte arquitetura:

Camadas de Convolução 2D: As camadas de convolução 2D extraem características das imagens através de operações de convolução com filtros.
Camadas de Max Pooling: As camadas de max pooling reduzem a dimensionalidade das características selecionando os valores máximos em regiões específicas.
Camadas de Dropout: As camadas de dropout desativam aleatoriamente neurônios durante o treinamento para prevenir o overfitting.
Camadas Densas: As camadas densas realizam a classificação final, sendo a última camada densa ativada pela função de ativação softmax para atribuir probabilidades às classes benigna e maligna.
Compilação e Treinamento do Modelo
Durante o treinamento, o modelo é compilado com o otimizador Adam e a função de perda Sparse Categorical Crossentropy. As métricas de acurácia e perda são registradas para monitorar o desempenho do modelo durante o treinamento. O número de épocas e os callbacks para registro de métricas de treinamento são definidos para ajustar o modelo.

## Método de Validação
O desempenho do modelo é avaliado utilizando dados de validação separados do conjunto de treinamento. A acurácia e a perda são registradas para avaliar a capacidade de generalização do modelo.

## Resultados e Métricas de Desempenho
Ao final do treinamento, os resultados são avaliados através de gráficos que mostram a acurácia e a perda do treinamento e validação em cada época. Essas métricas permitem avaliar a capacidade de aprendizado do modelo ao longo do treinamento.

Execução 3 vezes (Epocas)
 <img src="../Cancer de Pele/resultado2x.png">

 Execução 5 vezes (Epocas)
 <img src="../Cancer de Pele/resultado5x.png">

 Execução 11 vezes (Epocas)
 <img src="../Cancer de Pele/resultado11x.png">

 Execução 120 vezes (Epocas)
 <img src="../Cancer de Pele/resultado120x.png">

 Execução 250 vezes (Epocas)
 <img src="../Cancer de Pele/resultado251x.png">

## Conclusão
Através do desenvolvimento e treinamento de um modelo de classificação de câncer de pele utilizando deep learning, foi possível criar um sistema capaz de distinguir entre imagens benignas e malignas. O modelo pode ser utilizado como uma ferramenta de apoio para auxiliar profissionais da área médica na detecção precoce e diagnóstico do câncer de pele.

# AUC 
O valor do AUC varia de 0,0 até 1,0 e o limiar entre a classe é 0,5. Ou seja, acima desse limite, o algoritmo classifica em uma classe e abaixo na outra classe. Quanto maior o AUC, melhor
AUC: 0.7958333333333334

# RECALL 
O recall é o número de pessoas que o modelo identificou corretamente como tendo a doença dividido pelo número total de pessoas que realmente têm a doença nos seus dados. Ou seja, de todas as pessoas que ele poderia classificar como positivas, quantas ele acertou.

# ROC

As curvas ROC são criadas plotando-se a sensibilidade (verdadeiro positivo) no eixo y contra 1 − especificidade (verdadeiro negativo) no eixo x para cada valor encontrado em uma amostra de indivíduos com e sem a doença.

 <img src="../Cancer de Pele/curva roc 120x.png">

Recall
[0.775      0.81666667]


Precision Score

[0.83532934 0.75153374]
              precision    recall  f1-score   support

           0       0.84      0.78      0.80       360
           1       0.75      0.82      0.78       300

#Acurácia

Acurácia: indica uma performance geral do modelo. Dentre todas as classificações, quantas o modelo classificou corretamente;
    accuracy                           0.79       660
   macro avg       0.79      0.80      0.79       660
weighted avg       0.80      0.79      0.79       660
