
Motivação
Identificar utilizando Machine Learning com aprendizado profundo (Deep Learning) para classificar cancer de pele com base em imagens

O objetivo é criar um modelo de classificação de imagens para distinguir entre imagens benignas e malignas relacionadas ao câncer de pele.

Descrição do Dataset
https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign

Descrição dos dados

Etapas de preparação dos dados ou pré-processamento.
As imagens estão disponiveis em 2 pastas
Treinamento
  +Maligno
  +Benigno
Teste
  +Maligno
  +Benigno

Modelo de aprendizado de máquina

Formalização matemática.
Pré-processamento de Dados:

Normalização: As imagens de treinamento e teste são normalizadas dividindo cada pixel pelo valor máximo (255) para que fiquem no intervalo [0, 1].

Pré-processamento de Dados:
   - As imagens de treinamento e teste são normalizadas dividindo-as pelo valor máximo de pixel (255).
   - As imagens de treinamento passam por aumento de dados (data augmentation) usando a classe `ImageDataGenerator` do Keras. Isso inclui rotação, zoom e deslocamento horizontal.

Construção do Modelo:

Construção do Modelo:
   - O modelo de classificação de imagens é construído utilizando a classe `Sequential` do Keras.
   - O modelo contém camadas de convolução (`Conv2D`), camadas de max pooling (`MaxPool2D`), camadas de dropout (`Dropout`), uma camada flatten (`Flatten`) e camadas densas (`Dense`).
   - A última camada utiliza a função de ativação softmax para a classificação de duas classes (benigna e maligna).

Camadas de Convolução: O modelo utiliza camadas de convolução 2D para extrair características das imagens. Cada camada de convolução aplica uma operação de convolução em um conjunto de filtros para gerar mapas de características.
Camadas de Max Pooling: As camadas de max pooling reduzem a dimensionalidade dos mapas de características, selecionando o valor máximo em uma região específica.
Camadas de Dropout: As camadas de dropout ajudam a prevenir o overfitting, desligando aleatoriamente um percentual de neurônios durante o treinamento.
Camadas Densas: As camadas densas são responsáveis por realizar a classificação final. A última camada densa utiliza a função de ativação softmax para atribuir probabilidades às classes benigna e maligna.
Compilação e Treinamento do Modelo:

Otimizador Adam: O modelo é compilado com o otimizador Adam, que utiliza o algoritmo de otimização estocástica baseado em gradientes para ajustar os pesos da rede neural durante o treinamento.
Função de Perda Sparse Categorical Crossentropy: A função de perda utilizada é a Sparse Categorical Crossentropy, que é adequada para problemas de classificação com múltiplas classes. Ela compara as probabilidades preditas pelo modelo com as classes reais e calcula a perda.
Métricas de Avaliação: Durante o treinamento, são registradas as métricas de acurácia e perda tanto para o conjunto de treinamento quanto para o conjunto de validação.


Método de validação.
Compilação e Treinamento do Modelo:
   - O modelo é compilado com o otimizador Adam e a função de perda `SparseCategoricalCrossentropy`.
   - O modelo é treinado utilizando o método `fit`, passando os dados de treinamento e validação, o número de épocas e os callbacks para registro de métricas de treinamento.
Medidas de desempenho.

Avaliação

Amostras usadas para treinamento, validação e teste.

Medidas de desempenho.

Conclusão
 - A acurácia e a perda do treinamento e validação são registradas durante o treinamento e plotadas em gráficos.
 - O modelo final é salvo em um arquivo.


Bibliotecas Importadas:
   - O código importa várias bibliotecas necessárias para a criação e treinamento do modelo de classificação, incluindo:
     - `matplotlib.pyplot` e `seaborn` para visualização de dados e gráficos.
     - `keras` para criação do modelo de rede neural.
     - `sklearn.metrics` para métricas de avaliação do modelo.
     - `tensorflow` para manipulação de tensores e execução do modelo.
     - `cv2` para processamento de imagens utilizando OpenCV.
     - `os` para manipulação de arquivos e diretórios.
     - `numpy` para operações numéricas e manipulação de arrays.

Funções Auxiliares:
   - `get_data(data_dir)`: Essa função é responsável por carregar os dados de treinamento e teste. Ela percorre os diretórios fornecidos para cada classe (benigna e maligna), lê as imagens, redimensiona-as para um tamanho específico e as armazena em uma lista junto com as respectivas classes.
   




  
