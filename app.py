import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain.chains import RetrievalQA
import tempfile

# Configuração da chave de API
os.environ["GOOGLE_API_KEY"] = 'AIzaSyAD37RNob17phhOJP2QcIG6NPC-Jxc1frw'#st.secrets["GOOGLE_API_KEY"]

st.set_page_config(page_title='IA', page_icon='🤖')
st.header('Assistente virtual')

st.sidebar.selectbox(
    label='Selecione IA',
    options=['gemini'],
    disabled=True
)

st.sidebar.markdown('### Sobre')
st.sidebar.markdown('Esta aplicação entrega insights sobre documentos importados')
st.sidebar.markdown('* Versão 1.0.0')

# Upload de Arquivo para RAG
uploaded_file = st.file_uploader("📂 Faça upload de um arquivo (txt, pdf, csv)", type=['txt', 'pdf', 'csv'])
rag_docs = []

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

        # Detectar tipo e carregar
        if uploaded_file.name.endswith('.txt'):
            loader = TextLoader(tmp_path)
        elif uploaded_file.name.endswith('.pdf'):
            loader = PyPDFLoader(tmp_path)
        elif uploaded_file.name.endswith('.csv'):
            loader = CSVLoader(tmp_path)
        else:
            loader = None

        if loader:
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            rag_docs = text_splitter.split_documents(documents)

# Inicialização do LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Banco de dados
db = SQLDatabase.from_uri('sqlite:///estoque.db')
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# Agente com ferramentas SQL
system_message = hub.pull('hwchase17/react')
agent = create_react_agent(llm=llm, tools=toolkit.get_tools(), prompt=system_message)
agent_executor = AgentExecutor(agent=agent, tools=toolkit.get_tools(), verbose=True, handle_parsing_errors=True)

# Template do Prompt
prompt = '''
Use the necessary tools to answer questions related to product inventory.
You will provide insights about products, prices, restocking, and reports as requested by the user.
The final answer should have a user-friendly formatting for visualization.
Always respond in Brazilian Portuguese.
Question: {q}
'''
prompt_template = PromptTemplate.from_template(prompt)

# Caixa de texto da pergunta
st.write('Faça uma pergunta...')
user_question = st.text_input('O que você gostaria de saber?')

# Botão de consulta
if st.button('Consultar'):
    if user_question:
        with st.spinner('Montando insights...'):
            formatted_prompt = prompt_template.format(q=user_question)

            # Se houver documentos para RAG, usa RetrievalQA
            if rag_docs:
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                vectorstore = FAISS.from_documents(rag_docs, embedding=embeddings)
                retriever = vectorstore.as_retriever()

                rag_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=retriever,
                    return_source_documents=True
                )

                result = rag_chain.invoke({'query': formatted_prompt})
                st.markdown("### 📌 Resposta com base nos documentos")
                st.markdown(result["result"])
            else:
                output = agent_executor.invoke({'input': formatted_prompt})
                st.markdown("### 📌 Resposta com base no banco de dados")
                st.markdown(output.get('output'))
    else:
        st.warning('Você precisa fazer uma pergunta.')
