import getpass, os
import regex as re
import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.messages.utils import convert_to_messages
from langchain_core.tools.retriever import create_retriever_tool
from langgraph.graph import MessagesState
from langchain_text_splitters import RecursiveCharacterTextSplitter


if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass("Please enter your HF token:\n")

st.set_page_config(page_title="KI-Demo", page_icon="🤖")
st.title("🤖 KI-Demo Chatbot")
prompt = st.chat_input("Stelle mir eine Frage:\n")


def create_index(documents, embeddings, index_path="vectorstore_index"):
    if os.path.exists(path=index_path):
        print("Loading existing index...")
        vector_store = FAISS.load_local(
            folder_path=index_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True # allow Pickle files
        )
    else:
        print("Building new index...")
        vector_store = FAISS.from_documents(documents=documents,embedding=embeddings)
        vector_store.save_local(folder_path=index_path)
    return vector_store


@st.cache_resource
def load_pipeline():
    loader = DirectoryLoader(
    path="./data",
    glob="*.pdf",
    loader_cls=PyPDFLoader,
    loader_kwargs={"mode": "single"}
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=25,
    add_start_index=True,
    )
    all_splits = text_splitter.split_documents(documents=docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = create_index(documents=all_splits, embeddings=embeddings)

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5,
            "score_threshold": 0.5
        }
    )

    retriever_tool = create_retriever_tool(
        retriever=retriever,
        name="PDF_retriever",
        description="Retrieve topic-related information." # change to the desired topic
    )

    model = HuggingFaceEndpoint(
        repo_id="HuggingFaceTB/SmolLM3-3B",
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
        provider="auto",
    )

    chatbot = ChatHuggingFace(llm=model)

    return vector_store, chatbot, retriever_tool

vector_store, chatbot, retriever_tool = load_pipeline()

GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks in German language."
    "Use ONLY information retrieved from the context in the database to answer the question."
    "If you don't know the answer, reply with the following sentence:" \
    "'Leider habe ich nicht genug Informationen, um diese Frage zu beantworten.'"
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)

def generate_query(state: MessagesState):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    response = (chatbot.bind_tools([retriever_tool]).invoke(state["messages"]))
    return {"messages": [response]}


def generate_answer(state: MessagesState):
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = chatbot.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}


column1, column2 = st.columns([2, 1]) # ratio 2:1 (response wider than context)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for role, msg in st.session_state["messages"]:
    with st.chat_message(role):
        st.markdown(msg)

if prompt:
    st.session_state["messages"].append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
    
    retrieved_docs = vector_store.similarity_search_with_score(query=prompt, k=5)
    retrieved_info = [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "filename": doc.metadata.get("source", "Keine Quellenangabe").split("/")[-1],
            "score": score,
        }
        for doc, score in retrieved_docs
    ]
    
    model_input = {
        "messages": convert_to_messages(
            [
                {
                    "role": "user",
                    "content": prompt,
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "1",
                            "name": "PDF_retriever",
                            "args": {"query": prompt}
                        }
                    ],
                },
            ]
        )
    }

    response = generate_answer(model_input)
    model_output = response["messages"][-1].content
    model_name = response["messages"][-1].response_metadata.get("model_name", "Unbekanntes LLM")
    regex = r"<think>(.*?)<\/think>\s*(.*)"
    match = re.search(regex, model_output, re.DOTALL)

    if match:
        reasoning = match.group(1)
        answer = match.group(2).strip()
    else:
        answer = model_output.strip()

    with column1:
        with st.chat_message("assistant"):
            st.markdown(answer)
        with st.expander(f"Model Info: {model_name}"):
            st.json(response["messages"][-1].response_metadata)

    st.session_state["messages"].append(("assistant", answer))

    with column2:
        st.markdown("### Dokumentenkontext")
        for i, doc in enumerate(retrieved_info, 1):
            with st.expander(f"Dokument {i} - {doc["metadata"].get("source", "Keine Quellenangabe")} (Score: {doc['score']:.2f})"):
                st.markdown(f"- **Inhalt:** {doc['content'][:100]}...")
