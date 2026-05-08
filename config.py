import os
from dotenv import load_dotenv

load_dotenv()
DATA_PATH       = "cleaned_job_skills_technical_only.csv"
CHROMA_PATH     = "chroma_store"
COLLECTION_NAME = "resume_skills"


EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-mpnet-base-v2"   
)
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")


CHUNK_OVERLAP    = 50
CHUNK_SEPARATORS = ["\n", ", ", " "]


SHORT_DOC_CHARS  = 350   
MEDIUM_DOC_CHARS = 600    



TOP_K              = int(os.getenv("TOP_K", 15))
SIMILARITY_THRESHOLD = float(os.getenv("SIM_THRESHOLD", 0.3))
SKILL_FREQUENCY_THRESHOLD = float(os.getenv("SKILL_FREQ_THRESHOLD", 0.15))
                     
MMR_FETCH_MULTIPLIER = 3   


GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
GROQ_MODEL    = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", 0.7))
GROQ_MAX_TOKENS  = int(os.getenv("GROQ_MAX_TOKENS", 300))


PROMPT_TONE   = os.getenv("PROMPT_TONE", "concise")


MAX_MISSING_SHOWN = int(os.getenv("MAX_MISSING_SHOWN", 10))


OUTPUT_FORMAT = os.getenv("OUTPUT_FORMAT", "text")

SCHEMA_VERSION = "v1.1"
# MySQL — add these to your .env
MYSQL_HOST     = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT     = int(os.getenv("MYSQL_PORT", 3306))
MYSQL_USER     = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "skill_gap_db")


if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not set — add it to your .env file"
    )