import streamlit as st
import pandas as pd

df = pd.read_csv("./database/personalized_learning_dataset.csv")

st.markdown('<h1 style="text-align: center;">📖 Análise de Evasão Escolar</h1>', unsafe_allow_html=True)

st.divider()

st.markdown('<h2>✒️Sobre o projeto</h2>', unsafe_allow_html=True)
st.write('Este projeto tem como objetivo analisar dados de desempenho acadêmico em cursos online para prever a evasão dos alunos e identificar padrões que podem levar ao sucesso acadêmico.')
st.dataframe(df)
st.markdown('<a href="https://www.kaggle.com/datasets/adilshamim8/personalized-learning-and-adaptive-education-dataset">Acesse a base de dados no Kaggle</a>', unsafe_allow_html=True)
