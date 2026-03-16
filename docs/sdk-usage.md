# SDK Usage

This guide describes the lightweight SDK included in `ai_engine.sdk` for
parsing and validating objects returned by `POST /generate`.

## Goals

- Provide typed client-side objects for generated games.
- Keep quiz and pasapalabra models separated.
- Include language metadata with a stable `language_id`.

## Supported Models

### Quiz model

`GeneratedQuiz` supports three question variants:

- `multiple_choice` (2 to 4 options)
- `true_false`
- `best_answer` (2 to 4 options, most-correct option)

### Pasapalabra model

`GeneratedPasapalabra` stores entries with:

- `letter`
- `relation` (`starts_with` or `contains`)
- `word`
- `definition`

## Language section

Every generated SDK object includes metadata:

- `generation_id`: unique identifier for the generated object
- `language`: ISO 639-1 code
- `language_id`: stable language identifier (`lang-es`, `lang-en`, ...)

## Basic usage

```python
from ai_engine.sdk import LanguageCode, parse_generate_response

payload = {
    "game_type": "pasapalabra",
    "game": {
        "game_type": "pasapalabra",
        "title": "Science Rosco",
        "topic": "Science",
        "words": [
            {
                "letter": "A",
                "hint": "Basic unit of matter",
                "answer": "Atom",
                "starts_with": True,
            }
        ],
    },
}

result = parse_generate_response(payload, language=LanguageCode.EN)
print(result.metadata.language_id)  # lang-en
```
