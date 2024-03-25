import pandas as pd
import streamlit as st
from PIL import Image
import numpy as np
from model import open_data, preprocess_data, split_data, load_model_and_predict


def process_main_page():
    show_main_page()
    process_side_bar_inputs()


def show_main_page():
    image = Image.open('data/house_pic.jfif')

    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Demo Titanic",
        page_icon=image,

    )

    st.header(
        """
         :rainbow[Узнайте стоимость дома!]
        """
    )

    st.image(image)


def write_user_data(df):
    st.subheader("Характеристики дома")
    st.write(df)


def write_prediction(prediction):
    st.write("# :blue[Приблизительная цена, $]")
    st.write(f'## {prediction}')

def process_side_bar_inputs():
    st.sidebar.write(':red[Укажите параметры дома]')
    user_input_df = sidebar_input_features()


    train_df = open_data()
    train_X_df, _ = split_data(train_df)
    full_X_df = pd.concat((user_input_df, train_X_df), axis=0)
    preprocessed_X_df = preprocess_data(full_X_df, test=False)

    user_X_df = preprocessed_X_df[:1]
    write_user_data(user_X_df)

    prediction = load_model_and_predict(user_X_df)
    prediction = np.round(prediction, 2)
    write_prediction(prediction[0])


def sidebar_input_features():
    building = st.sidebar.selectbox("Тип дома", ("Отдельный дом на одну семью", "Дом на две семьи",
                                                     "Таунхаус на две квартиры", "Крайняя квартира в таунхаусе",
                                                     "Средняя квартира в таунхаусе"))
    utilities = st.sidebar.selectbox("Удобства", ("Все удобства", "Электричество, газ и водоснабжение",
                                                  "Электричество и газ", "Электричество"))
    condition = st.sidebar.slider('Состояние дома по шкале от 1 до 10, где 1 значит "отличное", а 10 "очень плохое"',
                                  min_value=1, max_value=10, value=3, step=1)

    living_area = st.sidebar.number_input("Жилая площадь, кв.м. (:red[целое] число)", min_value=10, step=1)
    living_area = living_area * 10.764 # square meters to sq feet

    translation = {
        "Отдельный дом на одну семью": "1Fam",
        "Дом на две семьи": "2FmCon",
        "Таунхаус на две квартиры": "Duplx",
        "Крайняя квартира в таунхаусе": "TwnhsE",
        "Средняя квартира в таунхаусе": "TwnhsI",
        "Все удобства": "AllPub",
        "Электричество, газ и водоснабжение": "NoSewr",
        "Электричество и газ": "NoSeWa",
        "Электричество": "ELO"
    }

    data = {
        "BldgType": translation[building],
        "Utilities": translation[utilities],
        "OverallCond": condition,
        "GrLivArea": living_area
    }

    df = pd.DataFrame(data, index=[0])

    return df


if __name__ == "__main__":
    process_main_page()