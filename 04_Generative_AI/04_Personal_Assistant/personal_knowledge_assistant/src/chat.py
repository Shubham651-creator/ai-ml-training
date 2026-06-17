from dotenv import load_dotenv

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()
conversation_memory = []

CHROMA_DIR = "../data/chroma_db"


def get_vector_db():
    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

    return vectordb


def retrieve_context(vectordb, query):

    docs = vectordb.similarity_search(
        query,
        k=3
    )

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    sources = []

    for doc in docs:
        page = doc.metadata.get("page", "Unknown")
        source = doc.metadata.get("source", "Unknown")

        sources.append(
            f"{source} | page {page}"
        )

    return context, sources


def ask_gemini(question, context, chat_history=""):

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash"
    )

    prompt = f"""
You are an expert teacher.

Conversation History:
{chat_history}

Document Context:
{context}

Current Question:
{question}

Answer using the document context.
Use conversation history when needed.
"""

    response = llm.invoke(prompt)

    return response.content

def get_recent_context():

    return "\n".join(
        conversation_memory[-6:]
    )


def main():

    vectordb = get_vector_db()

    while True:

        question = input("\nAsk a question (or 'exit'): ")

        if question.lower() == "exit":
            break

        context, sources = retrieve_context(
            vectordb,
            question
        )

        history = get_recent_context()
        answer = ask_gemini(
            question,
            context,
            history
        )

        conversation_memory.append(
            f"User: {question}"
        )
        conversation_memory.append(
            f"Assistant: {answer}"
        )

        print("\nAnswer:\n")
        print(answer)

        print("\nSources:\n")
        for source in sources:
            print(f"- {source}")


if __name__ == "__main__":
    main()