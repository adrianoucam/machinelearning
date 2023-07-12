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

## Conclusão
Através do desenvolvimento e treinamento de um modelo de classificação de câncer de pele utilizando deep learning, foi possível criar um sistema capaz de distinguir entre imagens benignas e malignas. O modelo pode ser utilizado como uma ferramenta de apoio para auxiliar profissionais da área médica na detecção precoce e diagnóstico do câncer de pele.
