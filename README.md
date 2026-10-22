# Análise e Otimização do Auxílio Estudantil por meio de Grafos 🎓

Repositório oficial do Trabalho de Conclusão de Curso (TCC) do curso de Sistemas de Informação pela **Universidade Federal de Itajubá (UNIFEI)**.

**Autor:** Armando Manuel da Silva Correia Júnior

## 📌 Sobre o Projeto
Este sistema é uma plataforma computacional em Python (via Streamlit) desenvolvida para otimizar o processamento e a distribuição do auxílio estudantil. Ao substituir as tradicionais planilhas lineares por **Teoria dos Grafos** e algoritmos de Machine Learning, a aplicação mapeia vulnerabilidades socioeconómicas interconectadas de forma justa e livre de vieses.

## 🚀 Principais Funcionalidades
- **Deteção de Comunidades (Louvain):** Agrupamento inteligente de alunos com base nas suas afinidades socioeconómicas.
- **Grafo Interativo Multidimensional:** Visualização da rede estrutural com filtros dinâmicos.
- **Auditoria de Algoritmos:** Validação e comparação entre Coeficiente de Silhueta (K-Means) e Modularidade.
- **Gráfico de Radar:** Avaliação qualitativa normalizada (0 a 1) para cada comunidade delineada.
- **Simulador de Orçamento Institucional:** Criação dinâmica de linhas de corte baseadas nas dotações financeiras da Pró-Reitoria.
- **Métricas de Rede (Centralidade):** Extração avançada da relevância dos nós no modelo de vulnerabilidade.

## 🛠️ Instalação e Execução
1. Certifique-se que o Python 3.9+ está instalado na sua máquina.
2. Clone este repositório para o seu ambiente local.
3. Instale os pacotes de dependência necessários no terminal:
   ```bash
   pip install pandas streamlit networkx python-louvain scikit-learn pyvis plotly pdfplumber
