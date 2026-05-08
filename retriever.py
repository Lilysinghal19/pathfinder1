import os
from collections import Counter
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

_hf = os.getenv("HF_TOKEN") or os.getenv("hf_token")
if _hf:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = _hf
    os.environ["HF_TOKEN"] = _hf
import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import config

GARBAGE_ROLES = {
    "challenging", "unknown", "nurse", "nursing",
    "opportunity as a br", "a", "an", "the",
}

def is_valid_role(role: str) -> bool:
    return role.strip().lower() not in GARBAGE_ROLES


_embeddings  = None
_vectorstore = None

def get_vectorstore() -> Chroma:
    global _embeddings, _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    if not os.path.exists(config.CHROMA_PATH):
        raise RuntimeError("ChromaDB not found. Run embeddings.py first.")

    print("Loading ChromaDB")
    _embeddings = HuggingFaceEmbeddings(
        model_name    = config.EMBEDDING_MODEL,
        model_kwargs  = {"device": config.EMBEDDING_DEVICE},
        encode_kwargs = {"normalize_embeddings": True},
    )
    chroma_client=chromadb.PersistentClient(path=config.CHROMA_PATH)
    _vectorstore = Chroma(
        client = chroma_client,
        embedding_function = _embeddings,
        collection_name    = config.COLLECTION_NAME,
    )
    print(f"ChromaDB loaded — {_vectorstore._collection.count()} chunks")
    return _vectorstore

def get_embeddings() -> HuggingFaceEmbeddings:
    
    get_vectorstore()
    return _embeddings


def build_query(skills: str, role: str) -> str:
    skills_clean = ", ".join([
        s.strip().lower()
        for s in skills.split(",")
        if s.strip()
    ])
    return (
        f"Job Role: {role}\n"
        f"Skills: {skills_clean}"
    )


def get_available_roles() -> list[str]:
    vs = get_vectorstore()
    data = vs.get(include=["metadatas"])
    return sorted(set(
        m["job_role"]
        for m in data["metadatas"]
        if m.get("job_role", "").strip()
        and is_valid_role(m["job_role"])
    ))

def normalize_role(typed: str, available: list[str]) -> str | None:
    typed_lower = typed.strip().lower()
    for r in available:
        if r.lower() == typed_lower:
            return r
    typed_words = set(typed_lower.split())
    for r in available:
        if typed_words.issubset(set(r.lower().split())):
            return r
    for r in available:
        if typed_lower in r.lower():
            return r
    return None

def find_closest_roles(skills: str, role: str, top_n: int = 5) -> list[dict]:
    vs      = get_vectorstore()
    query   = build_query(skills, role)
    results = vs.similarity_search_with_score(query, k=top_n * 4)
    seen = {}
    for doc, score in results:
        r = doc.metadata.get("job_role", "")
        if not is_valid_role(r):
            continue
        sim = round(1 - score, 4)
        if r not in seen:
            seen[r] = sim
    closest = sorted(seen.items(), key=lambda x: x[1], reverse=True)
    return [{"role": r, "similarity": s} for r, s in closest[:top_n]]


def retrieve_role_docs(
    skills    : str,
    role      : str,
    top_k     : int   = config.TOP_K,
    threshold : float = config.SIMILARITY_THRESHOLD,
) -> list[Document]:
    vs    = get_vectorstore()
    query = build_query(skills, role)

    scored = vs.similarity_search_with_score(
        query  = query,
        k      = top_k,
        filter = {"job_role": role},
    )
    filtered = [
        doc for doc, score in scored
        if round(1 - score, 4) >= threshold
    ]
    if not filtered and scored:
        print(f"  Threshold {threshold} too strict — using top 3")
        filtered = [doc for doc, _ in scored[:3]]

    print(f"  Retrieved {len(filtered)} docs for '{role}'")
    return filtered

def extract_role_skills(
    docs      : list[Document],
    threshold : float = 0.10,
) -> set[str]:
    skill_counts = Counter()
    total_docs   = len(docs)

    for doc in docs:
        skills_in_doc = set(
            s.strip().lower()
            for s in doc.metadata.get("skills_text", "").split(", ")
            if s.strip()
        )
        for skill in skills_in_doc:
            skill_counts[skill] += 1

    min_count = max(2, int(total_docs * threshold))
    core = {s for s, c in skill_counts.items() if c >= min_count}

    print(f"  Role skills: {len(core)} core "
          f"(threshold {threshold*100:.0f}%, "
          f"min {min_count}/{total_docs} docs)")
    return core

def semantic_match(
    user_skills : list[str],
    role_skills : set[str],
    sem_threshold: float = 0.60,
) -> tuple[list[str], list[str]]:
    """
    Compare user skills vs role skills using embedding similarity.

    Returns (matching_user_skills, missing_role_skills)

    matching: user skills that semantically match a role skill
    missing : role skills not covered by any user skill
    """
    if not user_skills or not role_skills:
        return [], list(role_skills)

    emb = get_embeddings()      # safe — always initialized

    role_skills_list = sorted(role_skills)

    # Embed both lists
    user_vecs = emb.embed_documents(user_skills)
    role_vecs = emb.embed_documents(role_skills_list)

    # N×M similarity matrix
    sim_matrix = cosine_similarity(user_vecs, role_vecs)

    matched_role_indices = set()
    matching_user        = []

    for i, user_skill in enumerate(user_skills):
        best_score = 0
        best_j     = -1
        for j in range(len(role_skills_list)):
            if sim_matrix[i][j] > best_score:
                best_score = sim_matrix[i][j]
                best_j     = j

        if best_score >= sem_threshold:
            matching_user.append(user_skill)
            matched_role_indices.add(best_j)
            role_name = role_skills_list[best_j]
            if user_skill != role_name:
                print(f"  Semantic match: '{user_skill}' "
                      f"→ '{role_name}' (score {best_score:.2f})")

    missing = [
        role_skills_list[j]
        for j in range(len(role_skills_list))
        if j not in matched_role_indices
    ]

    return matching_user, missing


def build_context_string(
    docs    : list[Document],
    max_docs: int = 5,
) -> str:
    lines = []
    for i, doc in enumerate(docs[:max_docs], 1):
        m = doc.metadata
        lines.append(
            f"Profile {i}: {m.get('job_role','')} | "
            f"Skills: {m.get('skills_text','')[:80]} | "
            f"Exp: {m.get('experience_years','N/A')} yrs"
        )
    return "\n".join(lines)


def compute_skill_gap(
    user_skills_str : str,
    target_role     : str,
    top_k           : int   = config.TOP_K,
    threshold       : float = config.SIMILARITY_THRESHOLD,
) -> dict:

    print(f"\n{'─'*50}")
    print(f"Target role : {target_role}")
    print(f"User skills : {user_skills_str[:70]}")

    # Parse and normalize user skills to lowercase
    user_skills_list = [
        s.strip().lower()
        for s in user_skills_str.split(",")
        if s.strip()
    ]

    # Normalize role name
    available   = get_available_roles()
    matched     = normalize_role(target_role, available)

    if not matched:
        return {
            "error"          : f"Role '{target_role}' not found.",
            "suggestions"    : find_closest_roles(user_skills_str, target_role),
            "target_role"    : target_role,
            "user_skills"    : user_skills_list,
            "available_roles": available,
        }

    target_role = matched
    print(f"Matched role: {target_role}")

    # Retrieve relevant docs
    docs = retrieve_role_docs(user_skills_str, target_role, top_k, threshold)
    if not docs:
        return {
            "error"      : f"No documents retrieved for '{target_role}'.",
            "target_role": target_role,
            "user_skills": user_skills_list,
        }

    
    role_skills = extract_role_skills(docs, threshold=0.10)

   
    matching, missing = semantic_match(
        user_skills  = user_skills_list,
        role_skills  = role_skills,
        sem_threshold= 0.60,
    )

    total     = len(role_skills)
    match_pct = round(len(matching) / total * 100, 1) if total else 0.0
    gap_pct   = round(100 - match_pct, 1)

    return {
        "target_role"   : target_role,
        "user_skills"   : user_skills_list,
        "role_skills"   : sorted(role_skills),
        "matching"      : sorted(matching),
        "missing"       : sorted(missing),
        "match_pct"     : match_pct,
        "gap_pct"       : gap_pct,
        "total_required": total,
        "docs_retrieved": len(docs),
        "context_str"   : build_context_string(docs),
        "context_docs"  : docs,
    }


if __name__ == "__main__":

    roles = get_available_roles()
    print(f"\n{len(roles)} roles available:")
    for i, r in enumerate(roles, 1):
        print(f"  {i:2}. {r}")

    print("\n" + "="*50)
    user_skills = input("Your skills (comma separated): ").strip()
    target_role = input("Target role                  : ").strip()

    report = compute_skill_gap(user_skills, target_role)

    print(f"\n{'='*50}")
    if "error" in report:
        print(f"ERROR : {report['error']}")
        if "suggestions" in report:
            print("Did you mean:")
            for s in report["suggestions"]:
                print(f"  {s['role']} (sim: {s['similarity']})")
    else:
        print(f"Role    : {report['target_role']}")
        print(f"Match   : {report['match_pct']}%")
        print(f"Gap     : {report['gap_pct']}%")
        print(f"Docs    : {report['docs_retrieved']} retrieved")
        print(f"\nMatching ({len(report['matching'])}):")
        print("  " + (", ".join(report["matching"]) or "None"))
        print(f"\nMissing ({len(report['missing'])}):")
        print("  " + ", ".join(report["missing"][:12]))
        print(f"\nContext:")
        print(report["context_str"])

