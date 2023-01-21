# Contents of ~/my_app/pages/page_3.py
import streamlit as st

# some functions
def format_model(s):
    return "model_"+"_".join((s[:1].lower()+s[1:]).split(" "))


st.markdown("# Prediction by user input")


col11, col21 = st.columns(2)

with col11:
    sl = st.text_input(
        "Sepal length (cm)",
        5.1,
        placeholder="Insert a value for sepal length",
        key="sl",
    )
    if sl!="": sl = float(sl)

with col21:
    sw = st.text_input(
        "Sepal width (cm)",
        3.5,
        placeholder="Insert a value for sepal width",
        key="sw",
    )
    if sw!="": sw = float(sw)



col12, col22 = st.columns(2)

with col12:
    pl = st.text_input(
        "Petal length (cm)",
        1.4,
        placeholder="Insert a value for petal length",
        key="pl",
    )
    if pl!="": pl = float(pl)

with col22:
    pw = st.text_input(
        "Petal width (cm)",
        0.2,
        placeholder="Insert à value for petal width",
        key="pw",
    )
    if pw!="": pw = float(pw)

st.selectbox(
    'Select the type of model for prediction:',
    ('Kmeans', 'Decision tree', 'Random forest',
    'Logistic regression', 'Neural network'),
    key="model_type")


class_names = dict(zip(list(range(3)), ['Setosa', 'Versicolor', 'Virginica']))


st.markdown("##")
st.markdown("## Prediction with *:green[%s]*" % st.session_state.model_type)
st.markdown("The model parameters will be the ones specified in the Classification page")
model_name = format_model(st.session_state.model_type)

try:
    model = st.session_state[model_name]
    disabled=False
except Exception as e:
    st.markdown("#### :red[Please specify first the model in the *classification* page]")
    disabled = True


if st.button("Predict", disabled=disabled):
    predicted_value = model.predict([[sl, sw, pl, pw]])[0]
    st.write(class_names[predicted_value])