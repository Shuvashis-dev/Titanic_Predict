import pickle
import pandas as pd
import streamlit as st

model = pickle.load(open("titanic_model.pkl", "rb"))

st.title("Titanic Survival Predictor")

pclass = st.selectbox("Ticket class", [1, 2, 3])
sex = st.selectbox("Sex", ["female", "male"])
age = st.slider("Age", 0, 100, 25)
sibsp = st.number_input("Siblings or parter on ship", 0, 10)
parch = st.number_input("Parents/children on ship", 0, 10)
fare = st.number_input("Ticket fare", 0.0, 600.0, 30.0)

if st.button("Check"):
    data = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [1 if sex == "male" else 0],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
    })
    result = model.predict(data)[0]
    st.write("Survived" if result == 1 else "Did not survive")
