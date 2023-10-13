import streamlit as st
import pandas as pd
import openai

# OpenAIのAPIキーを設定
openai.api_key = st.secrets["openai_key"]

def get_response_from_gpt4(prompt, df):
    df_str = df.to_string()

    """GPT-4からの応答を取得する"""
    response = openai.ChatCompletion.create(
        model="gpt-4",  # 使用するモデル名を指定
        messages=[  # <--- 'messages'引数を追加
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{prompt}\nPlease answer based on the following data:\n{df_str}"}
        ]
    )
    print(prompt)  
    print(response)  
    return response['choices'][0]['message']['content']


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
    st.write(new_df)

    # GPT-4に問い合わせるプロンプトの入力
    user_input = st.text_area("Enter a prompt for GPT-4:")

    if user_input:
        # GPT-4からの応答を取得して表示
        response = get_response_from_gpt4(user_input, new_df)
        st.write(f"Response from GPT-4: {response}")

        # レスポンスをテキストファイルとしてダウンロード可能にする
        st.download_button(
            label="Download Response",
            data=response.encode(),
            file_name="gpt4_response.txt",
            mime="text/plain",
        )

    
