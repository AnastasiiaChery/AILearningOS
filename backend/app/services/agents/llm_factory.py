"""Provider-agnostic LLM factory — single source of provider logic."""
from langchain_core.language_models import BaseChatModel

from app.core.config import settings


def get_llm(streaming: bool = False, temperature: float | None = None) -> BaseChatModel:
    # temperature=None → provider default. Pass 0 for deterministic, low-variance
    # tasks (e.g. HyDE generation, where we want a stable hypothetical doc so the
    # eval compares identical retrievals across runs).
    extra = {} if temperature is None else {"temperature": temperature}
    provider = settings.llm_provider.lower()
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            streaming=streaming,
            **extra,
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
            streaming=streaming,
            **extra,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            streaming=streaming,
            **extra,
        )
    raise ValueError(f"Unknown LLM provider: {settings.llm_provider!r}. Set LLM_PROVIDER=groq|anthropic|openai.")
