import streamlit as st
import yfinance as yf
import requests
import pandas as pd
import numpy as np
import fundamentalanalysis as fa
import pickle
import time
from yahooquery import Ticker
import matplotlib.pyplot as plt
import plotly.tools
import seaborn as sns
from ml import *

api_key = "fca6cce17f6634c8b204f50c5006b4d4"
key = 'E790UV655NTI4FL6'

user_tckr = []
# # загрузка данных о компании
class DummyFile:
    def __init__(self, name):
        self.name = name

def load_data(user_tckr):
    user_data = pd.DataFrame()
    tickers = Ticker(user_tckr)
    df = pd.DataFrame(tickers.quotes).T
    data = tickers.get_modules("summaryProfile price")
    df = pd.DataFrame.from_dict(data).T
    dataframes = [pd.json_normalize([x for x in df[module] if isinstance(x, dict)]) for module in ['summaryProfile', 'price']]
    df = pd.concat(dataframes, axis=1)
    gl_info = df[['symbol','country', 'exchange', 'sector']].dropna().set_index('symbol')
    for ticker in user_tckr:
        user_df = pd.DataFrame()
        if not ticker in gl_info.index.tolist():
            st.error(f'Нету данных для тикера {ticker}', icon="🚨")
            pass
        else:
            fa_df = fa.key_metrics(ticker, api_key, period="annual").T.drop(['period'], axis = 1)
            if fa_df.empty:
                st.error(f'Нету финансовой информации для тикера {ticker}', icon="🚨")
                pass
            else:
                fa_df['symbol'] = ticker
                fa_df = fa_df.reset_index().set_index('symbol')
                fa_df = fa_df.join(gl_info)
                cols = fa_df.columns.tolist()
                user_df = fa_df[cols[-4:] + cols[:-4]]             
                user_data = pd.concat([user_data, user_df], axis=0).reset_index()
    user_data = user_data.rename(columns={'index':'year'})
    if 'level_0' in user_data.columns.tolist():
        user_data['symbol'] = user_data['symbol'].fillna(user_data['level_0'])
        user_data = user_data.drop(['level_0'], axis=1).set_index('symbol')
    else:
        user_data = user_data.set_index('symbol')
    return user_data


# Main app engine
if __name__ == "__main__":
    st.title("demo")
    st.write("""В данном проекте можно получить предсказания для цены акции, ROE, P/E и качестве доходов на основе показателей финансовой отчетности с помощью градиентного бустинга [CatBoost](https://share.streamlit.io/mesmith027/streamlit_webapps/main/MC_pi/streamlit_app.py)""")
    st.write()


    number_inputs = 1

    data = pd.DataFrame()

    
    if 'df' not in st.session_state:
        st.session_state.df = data

    input_values = [st.text_input(f'Введите тикер компании', help='Тикер в формате YNDX или yndx')
            for i in range(number_inputs)]
    
    hide_dataframe_row_index = """
    <style>
    .row_heading.level0 {display:none}
    .blank {display:none}
    </style>
    """
    hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """

    if st.button("Добавить в список компаний", key="button_update"):
        st.session_state.df = pd.concat(
                    [st.session_state.df, pd.DataFrame({'Тикер': input_values})],
                    ignore_index=True)
        st.success('Тикер добавлен')
        data_df = st.session_state.df.copy()
        st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)
        st.table(data_df)
    
    if "predict_state" not in st.session_state:
     st.session_state.predict_state = False

    if st.button('Получить данные и спрогнозировать') or st.session_state.predict_state:
        st.session_state.predict_state = True
        user_tckr = st.session_state.df['Тикер'].unique()
        user_tckr = [x.upper() for x in user_tckr]
        user_df = load_data(user_tckr)
        display_df = user_df

        st.markdown(hide_table_row_index, unsafe_allow_html=True)

        st.dataframe(display_df, use_container_width=True)
        
        csv = display_df.to_csv().encode('utf-8')
        st.download_button( 

            label="Скачать данные как CSV",
            data=csv,
            file_name='sample_df.csv',
            mime='text/csv',

        )

        with st.spinner('Идет предсказание'):
            time.sleep(5)
        st.success('Предсказания готовы')

        user_df = data_beautifier(user_df)
        display_df.to_csv('display_df.csv')

        df = pd.read_csv('aaplsbux.csv')

        predict_options = ['ROE', 'Цена акции', 'Качество дохода', 'P/E']
        predict = st.selectbox('Выберите показатель:', predict_options)
        
        ticker_options = df['Компания'].unique().tolist()
        ticker = st.multiselect('Выберите компанию:', ticker_options, default=ticker_options[0])

        df = df[df['Компания'].isin(ticker)]
        fig, ax = plt.subplots()
        ax = sns.lineplot(data=df, x="Год", y=predict, hue="Компания", style="Компания", markers=True)

        # ax.plot(df["Год"], df[predict],   marker='o')
        
        ax.set_title(f"{predict}")

        st.write('Предсказания указаны на конец финансового года')
        st.pyplot(fig)

        st.download_button( 

            label="Скачать график как PNG",
            data=csv,
            file_name='sample_df.csv',
            mime='text/csv',

        )

    #     pricePKL = pickle.load(open('pricePredict.sav' , 'rb')).predict(user_df[0])
    #     ebitPKL = pickle.load(open('ebitgrowth.pkl', 'rb'))
    #     epsPKL = pickle.load(open('epsgrowth.pkl', 'rb'))
    #     netIncomePKL = pickle.load(open('netIncomeGrowth.pkl', 'rb'))
    #     operatingIncomePKL = pickle.load(open('operatingIncomeGrowth.pkl', 'rb'))
    #     revenuePKL = pickle.load(open('revenueGrowth.pkl', 'rb'))
    #     print(ebitPKL.predict(user_df))

    #     for i in range(len(user_tckr)):
    #         st.write(user_tckr[i])
    #         c1, c2, c3, c4, c5 = st.columns(5)
    #         c1.metric(label = 'Рост EBIT', value=round(ebitPKL.predict(user_df)[i],3))
    #         c2.metric(label = 'Рост EPS', value=round(epsPKL.predict(user_df)[i], 3))
    #         c3.metric(label = 'Рост Чистой прибыли', value=round(netIncomePKL.predict(user_df)[i],3))
    #         c4.metric(label = 'Рост Оперативной прибыли', value=round(operatingIncomePKL.predict(user_df)[i],3))
    #         c5.metric(label = 'Рост  Выручки', value=round(revenuePKL.predict(user_df)[i],3))

    # st.write(":rabbit:")

from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb


def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))


def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
     .stApp { bottom: 105px; }
    </style>
    """

    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="black",
        text_align="center",
        height="auto",
        opacity=1
    )

    style_hr = styles(
        display="block",
        margin=px(8, 8, "auto", "auto"),
        border_style="inset",
        border_width=px(2)
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)


def footer():
    myargs = [
        "Сделано в ",
        image('https://www.hse.ru/mirror/pubs/share/522215913',
              width=px(25), height=px(25)),
        # link("https://vk.com/vsohdaampoo", "Зайцева Стася"),
        br(),
        link("https://vk.com/vsohdaampoo", text='Возникли вопросы? Нашли баг?'),
    ]
    layout(*myargs)


if __name__ == "__main__":
    footer()
    st.markdown("")
    st.markdown("")
    st.markdown("")
    
 



    #     # pickled_model = pickle.load(open('demo_ml.pkl', 'rb'))
    #     # st.text(pickled_model.predict(user_df)[0])



