import getpass, os
import regex as re
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_core.messages.utils import convert_to_messages
from langchain_core.tools.retriever import create_retriever_tool
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import MessagesState
from langchain_text_splitters import RecursiveCharacterTextSplitter


if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass("Please enter your HF token:\n")

print("Hallo! Ich bin Ihr virtueller Assistent. Wie kann ich Ihnen helfen? (Tippe 'exit' zum Beenden)\n")
# query = input("Hallo! Ich bin Ihr virtueller Assistent. Wie kann ich Ihnen helfen?\n")

# loader = PyPDFLoader(file_path="./data/erfuellungsbericht-100.pdf", mode="single")
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
vector_store = InMemoryVectorStore.from_documents(documents=all_splits, embedding=embeddings)
# vectore_store.add_documents(documents=all_splits)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5,
        "score_threshold": 0.5
    }
)

# retrieved_docs = vectore_store.similarity_search_with_score(query=query, k=5)
"""retrieved_info = [
    {
        "content": doc.page_content,
        "metadata": doc.metadata,
        "score": score
    }
    for doc, score in retrieved_docs
]"""

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


while True:
    user_query = input("Stelle mir eine Frage:\n")
    if user_query.lower() in ["exit", "stop", "quit"]:
        print("Chat beendet.")
        break

    retrieved_docs = vector_store.similarity_search_with_score(query=user_query, k=5)
    retrieved_info = [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": score
        }
        for doc, score in retrieved_docs
    ]

    model_input = {
        "messages": convert_to_messages(
            [
                {
                    "role": "user",
                    "content": user_query,
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "1",
                            "name": "PDF_retriever",
                            "args": {"query": user_query}
                        }
                    ],
                },
            ]
        )
    }

    response = generate_answer(model_input)
    
    regex = r"<think>(.*?)<\/think>\s*(.*)"
    match = re.search(regex, str(response["messages"][-1].content), re.DOTALL)

    if match:
        reasoning = match.group(1)
        model_response = match.group(2)
        # print(f"Model Reasoning: {reasoning}\n")
        print(f"Model Response: {model_response}")
    else:
        response["messages"][-1].pretty_print()

    # print(f"\n\n{retrieved_info}")
