"""Provider-agnostic LLM factory — single source of provider logic."""
from langchain_core.language_models import BaseChatModel

from app.core.config import settings


def get_llm(streaming: bool = False) -> BaseChatModel:
    provider = settings.llm_provider.lower()
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            streaming=streaming,
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
            streaming=streaming,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            streaming=streaming,
        )
    raise ValueError(f"Unknown LLM provider: {settings.llm_provider!r}. Set LLM_PROVIDER=groq|anthropic|openai.")
