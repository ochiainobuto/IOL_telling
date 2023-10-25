import streamlit as st
import pandas as pd
import openai
import json
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
from pandasai.middlewares.streamlit import StreamlitMiddleware

# from pandasai import SmartDataframe
# from pandasai.llm import OpenAI
# import matplotlib.pyplot as plt
# import matplotlib.figure

#pip install poetry
#pip install pandasai
#conda activate sdxl
#pip install openpyxl

# OpenAIのAPIキーを設定
API_KEY = st.secrets["openai_key"]
openai.api_key = API_KEY

llm=OpenAI(api_token=API_KEY)

pandas_ai = PandasAI(
        llm,
        verbose=True,
        enable_cache=False,
        enforce_privacy=False,
        conversational=True,
        middlewares=[StreamlitMiddleware()],
    )

#st.session_stateの初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

def translate_to_english(input_text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f'以下を英訳してください。”{input_text}"'}
        ]
    )
    return response['choices'][0]['message']['content']

def translate_to_japanese(input_text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f'以下を日本語訳してください。”{input_text}"'}
        ]
    )
    return response['choices'][0]['message']['content']

######Title##########################################################
st.title('IOL telling')

# エクセルファイルのアップロード
uploaded_file = st.file_uploader("", type="xlsx")

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("Excel Data")
    st.write(df)

    header_list = df.columns.tolist()
    sel1 = st.multiselect('比較データ項目', options = header_list)
    st.write(f'選択されたのは、{sel1}です。')
    new_df = df[sel1]

    if len(new_df) > 0:
        st.write(new_df)

    # アプリの再実行の際に履歴のチャットメッセージを表示
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input(): # Streamlit のチャット入力がある場合に実行する
        st.chat_message("user").markdown(prompt)
        e_prompt=translate_to_english(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"): # アシスタントの応答を表示するためのブロックを開始する
            response = pandas_ai.run(new_df, prompt=e_prompt)
            jp_response = translate_to_japanese(response)
            st.write(jp_response) # 応答をStreamlitのチャットに表示する

        st.session_state.messages.append({"role": "assistant", "content": jp_response})

        #レスポンスをテキストファイルとしてダウンロード可能にする
        # st.download_button(
        #     label="Download Response",
        #     data=json.dumps(st.session_state.messages),
        #     file_name="gpt_response.txt",
        #     mime="text/plain",
        # )



        

    
