import os
import streamlit as st
# from decouple import config
from langchain.prompts import PromptTemplate
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

# try:
#     os.environ["GOOGLE_API_KEY"] = config("GOOGLE_API_KEY")
# except:
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

st.set_page_config(
    page_title='Stock IA',
    page_icon='🤖'
)
st.header('Assistente de Estoque')


model_options = [
    'gemini',
]

st.sidebar.selectbox(
    label='Selecione IA',
    options=model_options,
    disabled=True,
    accept_new_options=False
)

st.sidebar.markdown('### Sobre')
st.sidebar.markdown('Esta aplicação tem como objetivo entregar insights sobre estoque, produtos, entradas e saídas')
st.sidebar.markdown('* Versão 1.0.0')

st.write('Faça uma pergunta sobre Estoque, Produtos...')
user_question = st.text_input('O que você gostaria de saber?')


llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
)

db = SQLDatabase.from_uri('sqlite:///estoque.db')

toolkit = SQLDatabaseToolkit(
    db=db,
    llm=llm,
)

system_message = hub.pull('hwchase17/react')

agent = create_react_agent(
    llm=llm,
    tools=toolkit.get_tools(),
    prompt=system_message,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=toolkit.get_tools(),
    verbose=True
)

prompt = '''
Use the necessary tools to answer questions related to product inventory.
You will provide insights about products, prices, restocking, and reports as requested by the user.
The final answer should have a user-friendly formatting for visualization.
Always respond in Brazilian Portuguese.
Question: {q}
'''

prompt_template = PromptTemplate.from_template(prompt)



if st.button('Consultar'):
    if user_question:
        with st.spinner('Montando insights...'):
            formatted_prompt = prompt_template.format(q=user_question)
            output = agent_executor.invoke(
                {
                    'input': formatted_prompt
                }
            )

            st.markdown(output.get('output'))
    else:
        st.warning('Your need to make a question')