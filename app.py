import streamlit as st
from streamlit_chat import message
import os
import matplotlib.collections
import matplotlib.pyplot as plt
import japanize_matplotlib

# From here down is all the StreamLit UI.
st.set_page_config(page_title="IOL telling", page_icon="📊")
st.header("IOL telling")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []
    
import io
from langchain.agents import load_tools, initialize_agent, AgentType, Tool, tool
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
import pandas as pd
# from langchain.agents import create_pandas_dataframe_agent
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import (
    HumanMessage,
)
from typing import Any, Dict, List

# OpenAIのAPIキーを設定
API_KEY = st.secrets["openai_key"]
os.environ['OPENAI_API_KEY'] = API_KEY

df = pd.DataFrame([])
data = st.file_uploader("", type="xlsx")

# st.download_button(label='サンプルデータをダウンロードする',data='https://drive.google.com/file/d/1wuSx35y3-hjZew1XhrM78xlAGIDTd4fp/view?usp=drive_open',mime='text/csv')

header_num = 0
index_num = 0
index_list = [i for i in range(index_num)]

if data:
    df = pd.read_excel(data, header=header_num, index_col=index_list)
    st.dataframe(df)

def get_text():
    input_text = st.text_input("You: ", "メーカーの分布を円グラフで表示して", key="input")
    return input_text

def get_state(): 
     if "state" not in st.session_state: 
         st.session_state.state = {"memory": ConversationBufferMemory(memory_key="chat_history")} 
     return st.session_state.state 
state = get_state()

prompt = PromptTemplate(
    input_variables=["chat_history","input"], 
    template='Based on the following chat_history, Please reply to the question in format of markdown. history: {chat_history}. question: {input}'
)

class SimpleStreamlitCallbackHandler(BaseCallbackHandler):
    """ Copied only streaming part from StreamlitCallbackHandler """
    
    def __init__(self) -> None:
        self.tokens_area = st.empty()
        self.tokens_stream = ""
        
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        self.tokens_stream += token
        self.tokens_area.markdown(self.tokens_stream)

ask_button = ""

if df.shape[0] > 0:
    agent = create_pandas_dataframe_agent(OpenAI(temperature=0, max_tokens=1000), df, memory=state['memory'], verbose=True, return_intermediate_steps=True)
    user_input = get_text()
    ask_button = st.button('ask')
else:
    pass

language = st.selectbox('language',['日本語','English'])

import json
import re
from collections import namedtuple
AgentAction = namedtuple('AgentAction', ['tool', 'tool_input', 'log'])

def format_action(action, result):
    action_fields = '\n'.join([f"{field}: {getattr(action, field)}"+'\n' for field in action._fields])
    return f"{action_fields}\nResult: {result}\n"

if ask_button:
    st.write("Input:", user_input)
    with st.spinner('typing...'):
        prefix = f'You are the best explainer. please answer in {language}. User: '
        handler = SimpleStreamlitCallbackHandler()
        response = agent({"input":user_input})
        answer = json.dumps(response['output'],ensure_ascii=False).replace('"', '')

        actions = response['intermediate_steps']
        actions_list = []

        #st.session_state.past.append(user_input)
        
        for action, result in actions:
            text = f"""Tool: {action.tool}\n
               Input: {action.tool_input}\n
               Log: {action.log}\nResult: {result}\n
            """
            if result is not None:
                st.set_option('deprecation.showPyplotGlobalUse', False)
                print(">>>>>>>>>>",result)
                if isinstance(result, matplotlib.collections.PathCollection):                    
                    st.pyplot(clear_figure=False)

                    # グラフをバイトストリームとして保存
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)

                    # ダウンロードボタンを作成
                    st.download_button(
                        label="このグラフをダウンロード",
                        data=buf,
                        file_name="plot.png",
                        mime="image/png"
                    )
                    #st.session_state.generated.append(buf.getvalue())

                elif isinstance(result, matplotlib.axes.Axes):
                    st.pyplot(clear_figure=False)

                    # グラフをバイトストリームとして保存
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)

                    # ダウンロードボタンを作成
                    st.download_button(
                        label="このグラフをダウンロード",
                        data=buf,
                        file_name="plot.png",
                        mime="image/png"
                    )
                    #st.session_state.generated.append(buf.getvalue())
                else:
                    st.write(result)
                    #st.session_state.generated.append(answer)

            text = re.sub(r'`[^`]+`', '', text)
            actions_list.append(text)
            
        if language == 'English':
            with st.expander('ℹ️ Show details', expanded=False):
                st.write('\n'.join(actions_list))
        else:
            with st.expander('ℹ️ 詳細を見る', expanded=False):
                st.write('\n'.join(actions_list))
            
        st.session_state.past.append(user_input)
        st.session_state.generated.append(answer)
        
if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        # try:
        #     st.image(st.session_state["generated"][i], caption=f"Graph {i + 1}")
        #     # ダウンロードボタンを作成
        #     # st.download_button(
        #     #     label="このグラフをダウンロード",
        #     #     data=st.session_state["generated"][i],
        #     #     file_name="plot.png",
        #     #     mime="image/png"
        #     # )
        # except:
        #     message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
        # message(st.session_state["past"][i], is_user=True, key=str(i) + "_user", avatar_style="thumbs")
