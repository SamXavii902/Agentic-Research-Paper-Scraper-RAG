import os
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

def initialize_cache(db_path=".langchain_cache.db"):
    """
    Initializes a local SQLite cache for all LLM calls.
    This saves massive amounts of API tokens during iterative testing
    by instantly returning previously generated responses for identical prompts.
    """
    print(f"Initializing SQLite LLM Cache at {db_path}...")
    set_llm_cache(SQLiteCache(database_path=db_path))
