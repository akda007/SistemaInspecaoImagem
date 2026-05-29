# Sistema de Inspecao Visual Automatica

## Processamento de Imagens e Deep Learning para Triagem Industrial

### Visao Geral
Este repositorio documenta a implementacao de um sistema automatizado para triagem de qualidade de frutos (macas, laranjas e bananas) em ambientes de distribuicao. O projeto integra tecnicas de Visao Computacional classica e modelos de Deep Learning (Transfer Learning) para garantir alta acuracia na identificacao de anomalias mercadologicas, superando desafios comuns como variacoes de iluminacao e artefatos de fundo.

### Estrutura do Repositorio
```text
.
├── notebooks/                # Jupyter Notebooks de desenvolvimento
│   ├── 01_segmentacao.ipynb  # Processamento de imagem e segmentacao Otsu
│   ├── 02_extracao.ipynb     # Extracao de descritores geometricos
│   ├── 03_classificacao.ipynb# Modelos Random Forest e SVM
│   └── 04_cnn_xai_bonus.ipynb# MobileNetV2 e explicabilidade (SHAP)
├── outputs/                  # Artefatos, modelos e logs gerados
│   ├── modelos/              # Pesos (.h5) e objetos serializados (.pkl)
│   └── shap_graficos.png     # Visualizacoes de explicabilidade
├── app.py                    # Aplicacao Streamlit para demonstracao
├── requirements.txt          # Dependencias do ambiente
└── README.md                 # Documentacao tecnica

```

### Pipeline Tecnologico

#### 1. Abordagem Classica (Pipeline de Atributos)

* Segmentacao: Implementacao de limiarizacao adaptativa e metodo de Otsu para isolamento de regiao de interesse (ROI).
* Extracao: Extracao de caracteristicas de forma, textura (GLCM) e cor (Espaco HSV/Hu Moments).
* Classificacao: Modelos baseados em Ensemble Learning (Random Forest) e Maquinas de Vetores de Suporte (SVM).

#### 2. Abordagem Moderna (Deep Learning)

* Arquitetura: Transfer Learning utilizando a rede MobileNetV2 pre-treinada no ImageNet.
* Processamento: Normalizacao de entrada e descongelamento de camadas densas para ajuste fino (fine-tuning).
* Explicabilidade: Auditoria de modelos via biblioteca SHAP, permitindo a visualizacao da relevancia de cada pixel e atributo na predicao final.

### Instalacao e Execucao

Para replicar o ambiente de desenvolvimento, utilize as seguintes instrucoes:

1. Clone o repositorio:
```bash
git clone https://github.com/seu-usuario/projeto-inspecao-frutas.git
cd projeto-inspecao-frutas
```
2. Configure o ambiente virtual:
```bash
python -m venv .venv
source .venv/bin/activate
```
3. Instale as dependencias:
```bash
pip install -r requirements.txt
```
4. Inicie o Dashboard:
```bash
streamlit run app.py
```

### Desempenho e Validacao

O modelo foi validado sob uma amostragem estratificada de 1.600 imagens (800 Fresh / 800 Rotten). A MobileNetV2 demonstrou robustez superior na generalizacao de dados fora da distribuicao original, mantendo uma matriz de confusao com baixa taxa de falsos negativos, essencial para garantir a seguranca no varejo.

### Licenca

Este projeto e de carater academico, desenvolvido como requisito parcial para a disciplina de Visao Computacional.

Autor: Andrey Koch de Araujo
Instituicao: Universidade Positivo

