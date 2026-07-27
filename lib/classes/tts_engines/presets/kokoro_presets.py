from lib.conf_models import TTS_ENGINES, default_engine_settings

# Kokoro has no fine-tuned checkpoints: one stock 82M model, many stock voices.
# 'voice' here is a Kokoro voice id (see default_engine_settings voices), not a
# wav path — the adapter overrides _set_voice() to resolve ids.
models = {
    "internal": {
        "lang": "multi",
        "repo": default_engine_settings[TTS_ENGINES['KOKORO']]['repo'],
        "sub": "",
        "voice": "bm_george",
        "files": default_engine_settings[TTS_ENGINES['KOKORO']]['files'],
        "samplerate": default_engine_settings[TTS_ENGINES['KOKORO']]['samplerate']
    }
}
