import os
import json
from groq import Groq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from retriever import compute_skill_gap, get_available_roles, get_vectorstore
import config
from langchain_groq import ChatGroq
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()


os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("hf_token")

groq_client = Groq(api_key=config.GROQ_API_KEY)

# ══════════════════════════════════════════════════════════════
# PROMPT TEMPLATES FOR GAP ANALYSIS
# All templates MUST include all format_prompt variables:
# {context} {target_role} {user_skills} {matching_skills}
# {missing_skills} {match_pct} {gap_pct} {max_missing}
# ══════════════════════════════════════════════════════════════
PROMPT_TEMPLATES = {

    "concise": """
You are a career assistant.

RETRIEVED CONTEXT:
{context}

CANDIDATE:
- Target Role : {target_role}
- Skills      : {user_skills}
- Matching    : {matching_skills} ({match_pct}%)
- Missing     : {missing_skills}
- Gap         : {gap_pct}%

Give SHORT output:
1. Current Level (max 2 lines)
2. Top {max_missing} Missing Skills (bullet points)
3. What to focus next (2 lines)

Be direct and specific. No long explanations.
""".strip(),

    "encouraging": """
You are a warm career advisor helping candidates grow.

RETRIEVED CONTEXT:
{context}

CANDIDATE:
- Target Role   : {target_role}
- Current Skills: {user_skills}
- Already Has   : {matching_skills} ({match_pct}% match)
- Still Needs   : {missing_skills}
- Gap           : {gap_pct}%

Provide clearly numbered sections:
1. ASSESSMENT — where they stand (2-3 sentences)
2. TOP {max_missing} SKILLS TO LEARN — ordered by importance
3. FREE RESOURCES — one resource per skill
4. TIMELINE — phases if studying 2hrs/day
5. MOTIVATION — personalised note based on {matching_skills}
""".strip(),

    "strict": """
You are a senior technical recruiter giving direct honest feedback.

RETRIEVED CONTEXT:
{context}

CANDIDATE:
- Target Role   : {target_role}
- Current Skills: {user_skills}
- Qualified In  : {matching_skills} ({match_pct}%)
- Gaps          : {missing_skills}
- Overall Gap   : {gap_pct}%

Provide:
1. HONEST ASSESSMENT (2 sentences)
2. CRITICAL GAPS — Top {max_missing} disqualifying skills
3. MINIMUM VIABLE SKILLSET before applying
4. TIMELINE — realistic, no sugarcoating
5. VERDICT — Ready / Not Yet / Needs X months
""".strip(),

    "technical": """
You are a principal engineer reviewing a technical profile.

RETRIEVED CONTEXT:
{context}

TECHNICAL PROFILE:
- Targeting    : {target_role}
- Has          : {user_skills}
- Overlaps     : {matching_skills} ({match_pct}%)
- Gaps         : {missing_skills}
- Gap severity : {gap_pct}%

Provide:
1. TECHNICAL ASSESSMENT — stack alignment
2. PRIORITY GAPS — Top {max_missing} by industry demand
3. LEARNING PATH — specific repos, docs, projects
4. PROJECT IDEAS — 2 portfolio projects
5. SPRINT PLAN — 2-week sprints
""".strip(),
}


def get_prompt_template(tone: str = config.PROMPT_TONE) -> PromptTemplate:
    if tone not in PROMPT_TEMPLATES:
        print(f"Unknown tone '{tone}' — using 'concise'")
        tone = "concise"
    return PromptTemplate(
        input_variables=[
            "context", "target_role", "user_skills",
            "matching_skills", "missing_skills",
            "match_pct", "gap_pct", "max_missing",
        ],
        template=PROMPT_TEMPLATES[tone],
    )


def format_prompt(
    gap_report  : dict,
    max_missing : int = config.MAX_MISSING_SHOWN,
    tone        : str = config.PROMPT_TONE,
) -> str:
    template = get_prompt_template(tone)
    return template.format(
        context         = gap_report.get("context_str", ""),
        target_role     = gap_report["target_role"],
        user_skills     = ", ".join(gap_report["user_skills"]),
        matching_skills = ", ".join(gap_report["matching"]) or "None yet",
        missing_skills  = ", ".join(gap_report["missing"][:max_missing])
                          or "None — fully qualified!",
        match_pct       = gap_report["match_pct"],
        gap_pct         = gap_report["gap_pct"],
        max_missing     = max_missing,
    )


# ══════════════════════════════════════════════════════════════
# CALL GROQ — single turn (gap analysis)
# ══════════════════════════════════════════════════════════════
def call_groq(
    prompt      : str,
    model       : str   = config.GROQ_MODEL,
    temperature : float = config.GROQ_TEMPERATURE,
    max_tokens  : int   = config.GROQ_MAX_TOKENS,
    output_fmt  : str   = config.OUTPUT_FORMAT,
) -> str:
    if output_fmt == "json":
        prompt += (
            "\n\nReturn as valid JSON with keys: "
            "assessment, priority_skills, resources, timeline, closing"
        )

    print(f"Calling GROQ: {model} | temp={temperature}")

    response = groq_client.chat.completions.create(
        model    = model,
        messages = [
            {
                "role"   : "system",
                "content": (
                    "You are a helpful career advisor. "
                    "Ground all advice in the retrieved profiles. "
                    "Be specific and actionable."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature = temperature,
        max_tokens  = max_tokens,
    )

    raw = response.choices[0].message.content
    if output_fmt == "json":
        try:
            json.loads(raw)
        except json.JSONDecodeError:
            print("GROQ invalid JSON — returning as text")
    return raw


# ══════════════════════════════════════════════════════════════
# CHAT MEMORY STORE
# Persists across multiple calls in the same Python process
# ══════════════════════════════════════════════════════════════
store       = {}
MAX_HISTORY = 10
def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    history = store[session_id]
    # Keep only last MAX_HISTORY messages
    if len(history.messages) > MAX_HISTORY:
        history.messages = history.messages[-MAX_HISTORY:]
    return history


# ══════════════════════════════════════════════════════════════
# BUILD CHATBOT CHAIN
# Built ONCE per insight — cached so it's not rebuilt each message
# Uses existing ChromaDB vectorstore as retriever
# ══════════════════════════════════════════════════════════════
_chain_cache: dict = {}


def build_chatbot_chain(extra_context: str = "") -> RunnableWithMessageHistory:
    """
    Builds a conversational RAG chain with:
    - History-aware retriever (rewrites question using chat history)
    - ChromaDB vectorstore for skill/role context
    - extra_context: AI insight + gap data injected into system prompt
    - Last 10 messages memory via RunnableWithMessageHistory

    Cached by extra_context so it's not rebuilt on every message.
    """
    # Use cache — only rebuild if extra_context changes
    cache_key = extra_context[:200]
    if cache_key in _chain_cache:
        return _chain_cache[cache_key]

    vs        = get_vectorstore()
    retriever = vs.as_retriever(
        search_type   = "similarity",
        search_kwargs = {"k": 5},
    )

    llm = ChatGroq(
        groq_api_key = config.GROQ_API_KEY,
        model_name   = "llama-3.1-8b-instant",   # stable GROQ model name
        temperature  = 0.3,
    )

    # Step 1 — rewrite user question using chat history
    # so follow-up questions like "tell me more" still retrieve correctly
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Given the chat history and the latest user question, "
            "rewrite it as a standalone question that can be understood "
            "without the chat history. Do NOT answer it. "
            "Just rewrite it if needed, otherwise return it as-is.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # Step 2 — answer using retrieved docs + gap context
    qa_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            f"""You are a helpful career advisor with memory of this conversation.

Candidate's analysis context:
{extra_context}

Use the retrieved documents below as additional knowledge about skills and roles.
{{context}}

Rules:
- Answer ONLY questions about skills, career development, and learning paths
- Be specific and refer to the candidate's actual skills and gaps when relevant
- Keep responses concise — 2-4 short paragraphs maximum
- Be encouraging but honest
- If asked something unrelated to careers, politely redirect""",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    doc_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, doc_chain)

    conversational_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key   = "input",
        history_messages_key = "chat_history",
        output_messages_key  = "answer",
    )

    _chain_cache[cache_key] = conversational_chain
    return conversational_chain


# ══════════════════════════════════════════════════════════════
# CHAT WITH KNOWLEDGE BASE
# Called by Streamlit on every user message
# extra_context = AI insight + gap data (passed from app.py)
# ══════════════════════════════════════════════════════════════
def chat_with_knowledge_base(
    question   : str,
    insight    : str  = "",
    session_id : str  = "default",
) -> str:
    """
    Multi-turn chat with:
    - Memory of last 10 messages
    - ChromaDB vectorstore for skill/role retrieval
    - insight: AI gap analysis result injected as context

    Returns the assistant's reply as a string.
    """
    chain = build_chatbot_chain(extra_context=insight)

    response = chain.invoke(
        {"input": question},
        config={"configurable": {"session_id": session_id}},
    )

    return response["answer"]


# ══════════════════════════════════════════════════════════════
# MASTER FUNCTION — Streamlit calls this for gap analysis
# ══════════════════════════════════════════════════════════════
def analyze_skill_gap(
    user_skills_str : str,
    target_role     : str,
    tone            : str   = config.PROMPT_TONE,
    model           : str   = config.GROQ_MODEL,
    output_fmt      : str   = config.OUTPUT_FORMAT,
    max_missing     : int   = config.MAX_MISSING_SHOWN,
    top_k           : int   = config.TOP_K,
    threshold       : float = config.SIMILARITY_THRESHOLD,
) -> dict:

    gap_report = compute_skill_gap(
        user_skills_str = user_skills_str,
        target_role     = target_role,
        top_k           = top_k,
        threshold       = threshold,
    )

    if "error" in gap_report:
        return {
            "gap_report" : gap_report,
            "explanation": None,
            "prompt_used": None,
            "error"      : gap_report["error"],
            "suggestions": gap_report.get("suggestions", []),
        }

    prompt      = format_prompt(gap_report, max_missing, tone)
    explanation = call_groq(
        prompt     = prompt,
        model      = model,
        output_fmt = output_fmt,
    )

    return {
        "gap_report" : gap_report,
        "explanation": explanation,
        "prompt_used": prompt,
        "error"      : None,
        "suggestions": [],
    }


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    roles = get_available_roles()
    print(f"\n{len(roles)} roles available:")
    for i, r in enumerate(roles, 1):
        print(f"  {i:2}. {r}")

    user_skills = input("Your skills (comma separated): ").strip()
    target_role = input("Target role                  : ").strip()

    result = analyze_skill_gap(user_skills, target_role)

    if result["error"]:
        print(f"\nERROR: {result['error']}")
        if result.get("suggestions"):
            for s in result["suggestions"]:
                print(f"  {s['role']} (sim: {s['similarity']})")
    else:
        gap = result["gap_report"]
        print(f"\nMatch: {gap['match_pct']}% | Gap: {gap['gap_pct']}%")
        print(result["explanation"])