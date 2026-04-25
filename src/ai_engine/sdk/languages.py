"""Language models and catalog for the ai-engine SDK."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class LanguageCode(str, Enum):
    """Supported language code for generated game content."""

    EN = "en"


class LanguageInfo(BaseModel):
    """Language metadata used by SDK generation models.

    Attributes:
        code: ISO 639-1 language code.
        language_id: Stable identifier for generated models and persistence.
        name: Language name in English.
        native_name: Language self-name.
    """

    code: LanguageCode
    language_id: str
    name: str
    native_name: str


LANGUAGE_CATALOG: dict[LanguageCode, LanguageInfo] = {
    LanguageCode.EN: LanguageInfo(
        code=LanguageCode.EN,
        language_id="lang-en",
        name="English",
        native_name="English",
    ),
}


def get_language_info(language: LanguageCode | str) -> LanguageInfo:
    """Return canonical language metadata from code.

    Args:
        language: ISO language code as enum or string.

    Returns:
        The catalog entry for the language.

    Raises:
        ValueError: If language code is not supported by the SDK.
    """

    try:
        code = (
            language if isinstance(language, LanguageCode) else LanguageCode(language)
        )
    except ValueError as exc:
        supported = ", ".join(item.value for item in LanguageCode)
        raise ValueError(
            f"Unsupported language {language!r}. Supported values: {supported}"
        ) from exc
    return LANGUAGE_CATALOG[code]
