from functools import lru_cache
from lib.conf_omnivoice_languages import LANGUAGES


@lru_cache(maxsize=1)
def load_omnivoice_languages()->tuple[dict[str, str], dict[str, dict[str, object]]]:
    """Return ISO-639-3 -> OmniVoice ID and UI metadata mappings."""
    engine_languages:dict[str, str] = {}
    language_metadata:dict[str, dict[str, object]] = {}
    for language_id, language_name, iso3 in LANGUAGES:
        engine_languages[iso3] = language_id
        language_metadata[iso3] = {
            'name': language_name,
            'native_name': language_name,
            'max_chars': 182,
            'script': 'latin',
        }
    if len(engine_languages) != 646:
        raise RuntimeError(
            f'Expected 646 OmniVoice languages, found {len(engine_languages)}'
        )
    return engine_languages, language_metadata
