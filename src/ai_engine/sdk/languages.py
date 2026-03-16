"""Language models and catalog for the ai-engine SDK."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class LanguageCode(str, Enum):
    """Supported language codes for generated game content."""

    ES = "es"
    EN = "en"
    FR = "fr"
    DE = "de"
    IT = "it"
    PT = "pt"
    CA = "ca"


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
    LanguageCode.ES: LanguageInfo(
        code=LanguageCode.ES,
        language_id="lang-es",
        name="Spanish",
        native_name="Espanol",
    ),
    LanguageCode.EN: LanguageInfo(
        code=LanguageCode.EN,
        language_id="lang-en",
        name="English",
        native_name="English",
    ),
    LanguageCode.FR: LanguageInfo(
        code=LanguageCode.FR,
        language_id="lang-fr",
        name="French",
        native_name="Francais",
    ),
    LanguageCode.DE: LanguageInfo(
        code=LanguageCode.DE,
        language_id="lang-de",
        name="German",
        native_name="Deutsch",
    ),
    LanguageCode.IT: LanguageInfo(
        code=LanguageCode.IT,
        language_id="lang-it",
        name="Italian",
        native_name="Italiano",
    ),
    LanguageCode.PT: LanguageInfo(
        code=LanguageCode.PT,
        language_id="lang-pt",
        name="Portuguese",
        native_name="Portugues",
    ),
    LanguageCode.CA: LanguageInfo(
        code=LanguageCode.CA,
        language_id="lang-ca",
        name="Catalan",
        native_name="Catala",
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
