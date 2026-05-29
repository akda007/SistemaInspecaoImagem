# Sistema de Inspeção Visual Automática - Cenário A

Este repositório contém o código-fonte do Projeto Final da disciplina de Visão Computacional. O objetivo é implementar um pipeline clássico para a classificação automática de frutas (maçãs, bananas e laranjas) em duas categorias: dentro do padrão (fresh) e fora do padrão (rotten).

## Estrutura do Repositório

O projeto está organizado da seguinte forma:

- /dataset: Diretório contendo as imagens originais do dataset (deve ser baixado e extraído aqui).
- /notebooks: Contém os scripts interativos que compõem o pipeline do projeto.
  - 01_segmentacao.ipynb: Comparação entre métodos de segmentação (Otsu e HSV) para isolamento do objeto.
  - 02_features.ipynb: Extração de features manuais (forma, inerciais de Hu, cor e textura GLCM).
  - 03_classificacao.ipynb: Análise de features, divisão estratificada de dados e treinamento dos classificadores clássicos (Random Forest e SVM).
  - 04_xai_bonus.ipynb: Análise de explicabilidade do modelo Random Forest utilizando a biblioteca SHAP (nível avançado/bônus).
- /outputs: Diretório gerado automaticamente que armazena as tabelas de features extraídas (X.csv e y.csv), matrizes de confusão e gráficos salvos.
- requirements.txt: Lista de dependências e bibliotecas necessárias para a execução do código.

## Metodologia

O sistema segue rigorosamente o pipeline clássico de visão computacional:
1. Segmentação da imagem via Limiarização Global (Otsu).
2. Extração de vetor de features para cada item segmentado.
3. Padronização dos dados (StandardScaler) ajustada exclusivamente no conjunto de treino para evitar vazamento de dados.
4. Classificação binária avaliada por acurácia, precisão, recall, F1-score e matriz de confusão.

## Como Executar (Instruções em 3 Comandos)

Para garantir a reprodutibilidade do projeto, abra o terminal na raiz deste repositório e execute os comandos abaixo. 

Recomenda-se o uso de um ambiente virtual (venv) previamente ativado, caso esteja utilizando Linux.
```bash
1. Instale as dependências necessárias:
pip install -r requirements.txt

2. Inicie o servidor do Jupyter:
jupyter notebook

3. No navegador, acesse a pasta `notebooks/` e execute os arquivos `.ipynb` em ordem sequencial (do 01 ao 04).
```



> Nota: Certifique-se de que o dataset base está descompactado dentro da pasta `/dataset` antes de rodar o notebook `02_features.ipynb`, pois ele depende dessas imagens para popular a pasta `/outputs` com os dados de treinamento.