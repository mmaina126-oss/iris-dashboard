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
st.write("### Line Chart")
st.line_chart(
    df[["petal length (cm)", "petal width (cm)"]]
)
st.write("### Area Chart")

st.area_chart(
    df[["petal length (cm)", "petal width (cm)"]]
)
st.write("### Species Distribution")

species_count = df["species"].value_counts()

st.bar_chart(species_count)
st.write("### Summary Statistics")

st.write(df.describe())
if st.checkbox("Show Raw Data"):
    st.dataframe(df)
sepal_length = st.slider(
    "Select Minimum Sepal Length",
    float(df["sepal length (cm)"].min()),
    float(df["sepal length (cm)"].max())
)

filtered_data = df[df["sepal length (cm)"] >= sepal_length]

st.write(filtered_data)
st.metric("Total Flowers", len(df))
st.metric("Number of Features", len(df.columns))
st.sidebar.title("Iris Dashboard Menu")

st.sidebar.write("Use this menu to explore the dashboard.")
