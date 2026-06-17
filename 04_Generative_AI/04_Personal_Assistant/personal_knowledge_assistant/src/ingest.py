from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma


PDF_DIR = Path("../data/pdfs")
CHROMA_DIR = "../data/chroma_db"


def load_documents():
    documents = []

    pdf_files = list(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDFs found.")
        return []

    for pdf_file in pdf_files:
        print(f"Loading: {pdf_file.name}")

        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()

        documents.extend(docs)

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    return chunks


def create_vector_db(chunks):
    print("Loading embedding model...")

    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    print("Creating ChromaDB...")

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    print("Vector database created successfully.")

    return vectordb


def main():
    documents = load_documents()

    if not documents:
        return

    print(f"Loaded {len(documents)} pages")

    chunks = split_documents(documents)

    create_vector_db(chunks)


if __name__ == "__main__":
    main()