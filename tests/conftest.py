import os

from dotenv import load_dotenv

load_dotenv()

# Tests must never send traces: LLM seams are mocked, but with the .env keys
# loaded above, graph-level LangChain tracing still fires on mock runs —
# burning LangSmith trace quota (it did, 2026-07-15).
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGSMITH_TRACING_V2"] = "false"

# The results root is env-overridable so the cloud box can write to its
# persistent volume; an operator's .env must not redirect test artifacts there.
os.environ.pop("PRELUDE_RESULTS_DIR", None)
os.environ.pop("PRELUDE_REGISTRY_STAGE", None)
