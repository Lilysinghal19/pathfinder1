import os
import json
import shutil
import hashlib
import pandas as pd
import chromadb
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma          
import config
from dotenv import load_dotenv
load_dotenv()                                

_hf = os.getenv("HF_TOKEN") or os.getenv("hf_token")
if _hf:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = _hf
    os.environ["HF_TOKEN"] = _hf
    print("HF_TOKEN: loaded")
else:
    print("HF_TOKEN: NOT loaded — check .env")

def load_data(csv_path: str = config.DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["job_role", "skills_text"])
    df = df[df["job_role"].str.strip()    != ""]
    df = df[df["skills_text"].str.strip() != ""]
    print(f"Loaded {len(df)} rows | {df['job_role'].nunique()} roles")
    return df


def detect_chunk_size(text: str) -> int:
    length = len(text)
    if length < config.SHORT_DOC_CHARS:
        return length
    elif length < config.MEDIUM_DOC_CHARS:
        return 300
    else:
        return 500


def build_documents(df: pd.DataFrame) -> list[Document]:
    documents = []

    for _, row in df.iterrows():
        
        skills_list = [
            s.strip().lower()
            for s in str(row["skills_text"]).split(", ")
            if s.strip()
        ]
        skills_joined = ", ".join(skills_list)

    
        content = (
            f"Job Role: {row['job_role']}\n"
            f"Skills: {skills_joined}\n"
            f"Certifications: {row.get('Certifications', 'None')}\n"
            f"Education: {row.get('Education_Level', '')} "
            f"in {row.get('Field_of_Study', '')}"
        ).strip()

        
        metadata = {
            "job_role"        : str(row["job_role"]),
            "skills_text"     : skills_joined,   # normalized lowercase
            "experience_years": str(row.get("Experience_Years", 0)),
            "education_level" : str(row.get("Education_Level", "")),
            "field_of_study"  : str(row.get("Field_of_Study", "")),
            "certifications"  : str(row.get("Certifications", "None")),
            "source"          : str(row.get("source", "csv")),
            "schema_ver"      : config.SCHEMA_VERSION,
        }

        documents.append(
            Document(page_content=content, metadata=metadata)
        )

    print(f"Built {len(documents)} documents")
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    all_chunks  = []
    size_counts = {"short": 0, "medium": 0, "long": 0}

    for doc in documents:
        chunk_size = detect_chunk_size(doc.page_content)
        if chunk_size == len(doc.page_content):
            all_chunks.append(doc)
            size_counts["short"] += 1
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size    = chunk_size,
                chunk_overlap = config.CHUNK_OVERLAP,
                separators    = config.CHUNK_SEPARATORS,
            )
            chunks = splitter.split_documents([doc])
            all_chunks.extend(chunks)
            if chunk_size == 300:
                size_counts["medium"] += 1
            else:
                size_counts["long"]   += 1

    print(f"Chunking → short:{size_counts['short']} "
          f"medium:{size_counts['medium']} "
          f"long:{size_counts['long']} "
          f"| total:{len(all_chunks)}")
    return all_chunks


FINGERPRINT_FILE = os.path.join(config.CHROMA_PATH, ".schema_fingerprint")

def compute_fingerprint(df: pd.DataFrame) -> str:
    data = {
        "columns"        : sorted(df.columns.tolist()),
        "schema_version" : config.SCHEMA_VERSION,
        "embedding_model": config.EMBEDDING_MODEL,
        "row_count"      : len(df),
    }
    return hashlib.md5(
        json.dumps(data, sort_keys=True).encode()
    ).hexdigest()

def should_rebuild(fingerprint: str) -> bool:
    if not os.path.exists(config.CHROMA_PATH):
        print("ChromaDB not found — building fresh")
        return True
    if not os.path.exists(FINGERPRINT_FILE):
        print("No fingerprint found — rebuilding")
        return True
    with open(FINGERPRINT_FILE) as f:
        stored = f.read().strip()
    if stored != fingerprint:
        print("Schema changed — rebuilding ChromaDB")
        return True
    print("Schema unchanged — loading existing ChromaDB")
    return False

def save_fingerprint(fingerprint: str):
    os.makedirs(config.CHROMA_PATH, exist_ok=True)
    with open(FINGERPRINT_FILE, "w") as f:
        f.write(fingerprint)

def load_embedding() -> HuggingFaceEmbeddings:
    print(f"Loading embedding model: {config.EMBEDDING_MODEL}")
    emb = HuggingFaceEmbeddings(
        model_name    = config.EMBEDDING_MODEL,
        model_kwargs  = {"device": config.EMBEDDING_DEVICE},
        encode_kwargs = {"normalize_embeddings": True},
    )
    print("Embedding model ready")
    return emb

def build_vectorstore(
    chunks     : list[Document],
    embeddings : HuggingFaceEmbeddings,
) -> Chroma:
    if os.path.exists(config.CHROMA_PATH):
        shutil.rmtree(config.CHROMA_PATH)
        print("Deleted old ChromaDB")

    print(f"Storing {len(chunks)} chunks in ChromaDB...")
    vs = Chroma.from_documents(
        documents           = chunks,
        embedding           = embeddings,
        persist_directory   = config.CHROMA_PATH,
        collection_name     = config.COLLECTION_NAME,
        collection_metadata = {"hnsw:space": "cosine"},
    )
    print(f"Stored {vs._collection.count()} chunks")
    return vs

def load_vectorstore(embeddings: HuggingFaceEmbeddings) -> Chroma:
    return Chroma(
        client = chroma_client,
        embedding_function = embeddings,
        collection_name    = config.COLLECTION_NAME,
    )

def verify_store(vs: Chroma, df: pd.DataFrame):
    top_role   = df["job_role"].value_counts().index[0]
    sample_row = df[df["job_role"] == top_role].iloc[0]
    test_query = (
        f"Job Role: {sample_row['job_role']}\n"
        f"Skills: {sample_row['skills_text']}"
    )
    print(f"\nVerification:")
    print(f"  Role  : {top_role}")
    print(f"  Query : {test_query[:80]}...")
    results = vs.similarity_search_with_score(
        query  = test_query,
        k      = 5,
        filter = {"job_role": top_role},
    )
    for doc, score in results:
        sim = round(1 - score, 4)
        print(f"  [{sim}] {doc.metadata['skills_text'][:55]}")
    print(f"  {len(results)} results returned")


def main(csv_path: str = config.DATA_PATH):
    print("=" * 50)
    print("BLOCK 4 — embeddings.py")
    print("=" * 50)

    df          = load_data(csv_path)
    fingerprint = compute_fingerprint(df)
    rebuild     = should_rebuild(fingerprint)
    embeddings  = load_embedding()

    if rebuild:
        documents = build_documents(df)
        chunks    = chunk_documents(documents)
        vs        = build_vectorstore(chunks, embeddings)
        save_fingerprint(fingerprint)
    else:
        vs = load_vectorstore(embeddings)
        print(f"Loaded {vs._collection.count()} chunks")

    verify_store(vs, df)
    print(f"\nDone | chunks: {vs._collection.count()} "
          f"| schema: {config.SCHEMA_VERSION}")
    return vs

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else config.DATA_PATH
    main(path)