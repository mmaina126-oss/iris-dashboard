import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris

st.title("Iris Dataset Dashboard")

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["species"] = pd.Categorical.from_codes(
    iris.target,
    iris.target_names
)

st.write("### Iris Dataset")
st.dataframe(df)

st.write("### Scatter Plot")
st.scatter_chart(
    df,
    x="sepal length (cm)",
    y="sepal width (cm)"
)

st.write("### Bar Chart")
st.bar_chart(
    df[["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]]
)