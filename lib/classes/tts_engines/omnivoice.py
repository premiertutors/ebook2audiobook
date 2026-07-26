from lib.classes.tts_engines.common.headers import *
from lib.classes.tts_engines.common.preset_loader import load_engine_presets


class OmniVoiceEngine(TTSUtils, TTSRegistry, name="omnivoice"):

    def __init__(self, session: DictProxy):
        try:
            self.session = session
            self.models = load_engine_presets(self.session["tts_engine"])
            self.params = {
                "samplerate": default_engine_settings[TTS_ENGINES["OMNIVOICE"]][
                    "samplerate"
                ],
                "voice_prompts": {},
            }
            self.audio_segments = []
            self.language = self.session.get("language")
            if self.session.get("translate_enabled") and self.session.get("translate"):
                self.language = self.session["translate"]
            engine_languages = default_engine_settings[
                TTS_ENGINES["OMNIVOICE"]
            ]["languages"]
            if self.language not in engine_languages:
                raise ValueError(f"Language {self.language} not supported by OmniVoice.")
            self.omnivoice_language = engine_languages[self.language]
            fine_tuned = self.session.get("fine_tuned")
            if fine_tuned not in self.models:
                raise ValueError(
                    f"Invalid fine_tuned model {fine_tuned}. "
                    f"Available models: {list(self.models)}"
                )
            self.model_cfg = self.models[fine_tuned]
            self.tts_key = self.session["model_cache"]
            self.device = self._resolve_device()
            self.engine = self.load_engine()
        except Exception as e:
            raise ValueError(f"__init__() error: {e}") from e

    def _resolve_device(self) -> str:
        selected = self.session["device"]
        if selected in (
            devices["CUDA"]["proc"],
            devices["ROCM"]["proc"],
            devices["JETSON"]["proc"],
        ):
            return "cuda"
        if selected == devices["MPS"]["proc"]:
            return "mps"
        if selected == devices["XPU"]["proc"]:
            return "xpu"
        return "cpu"

    def load_engine(self) -> Any:
        try:
            import torch
            try:
                from omnivoice import OmniVoice
            except ImportError as e:
                raise RuntimeError(
                    "OmniVoice is not installed correctly. Re-run the "
                    "ebook2audiobook dependency installer."
                ) from e
            self.cleanup_memory()
            engine = loaded_tts.get(self.tts_key)
            if engine is None:
                dtype = (
                    torch.float16
                    if self.device in ("cuda", "xpu")
                    else torch.float32
                )
                engine = OmniVoice.from_pretrained(
                    self.model_cfg["repo"],
                    device_map=self.device,
                    dtype=dtype,
                )
                loaded_tts[self.tts_key] = engine
            return engine
        except Exception as e:
            raise RuntimeError(f"load_engine() error: {e}") from e

    def convert(self, sentence_file: str, sentence: str, **kwargs) -> tuple:
        try:
            import torch
            from lib.classes.tts_engines.common.audio import is_audio_data_valid

            self.params["block_voice"] = kwargs.get(
                "block_voice", self.session.get("voice")
            )
            self.params["current_voice"], error = self._set_voice(
                self.params["block_voice"]
            )
            if self.params["current_voice"] is None and error is not None:
                return False, error
            self.audio_segments = []
            for part in self._split_sentence_on_sml(sentence):
                part = part.strip()
                if not part:
                    continue
                if SML_TAG_PATTERN.fullmatch(part):
                    success, error = self._convert_sml(part)
                    if not success:
                        return False, error
                    continue
                if not any(character.isalnum() for character in part):
                    continue
                generate_args = {
                    "text": part,
                    "language": self.omnivoice_language,
                }
                current_voice = self.params["current_voice"]
                if current_voice is not None:
                    prompt = self.params["voice_prompts"].get(current_voice)
                    if prompt is None:
                        prompt = self.engine.create_voice_clone_prompt(
                            ref_audio=current_voice
                        )
                        self.params["voice_prompts"][current_voice] = prompt
                    generate_args["voice_clone_prompt"] = prompt
                with torch.inference_mode():
                    result = self.engine.generate(**generate_args)
                if not result or not is_audio_data_valid(result[0]):
                    return False, "OmniVoice returned invalid audio."
                self.audio_segments.append(
                    torch.as_tensor(result[0], dtype=torch.float32).unsqueeze(0)
                )
            if not self.audio_segments:
                return False, "OmniVoice returned no audio."
            audio = torch.cat(self.audio_segments, dim=-1)
            samplerate = int(
                getattr(self.engine, "sampling_rate", self.params["samplerate"])
            )
            if not self.audio_save(sentence_file, audio, samplerate):
                return False, f"Cannot save {sentence_file}"
            self.audio_segments = []
            return True, None
        except Exception as e:
            self.cleanup_memory()
            self.audio_segments = []
            return False, self.log_exception(
                f"{self.__class__.__name__}.convert()", e
            )

    def create_vtt(self, all_sentences: list) -> bool:
        return self._build_vtt_file(all_sentences)
