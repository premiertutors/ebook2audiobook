from lib.classes.tts_engines.common.headers import *
from lib.classes.tts_engines.common.preset_loader import load_engine_presets

#sys.stderr = StdoutFilter(sys.stdout)

class Kokoro(TTSUtils, TTSRegistry, name='kokoro'):
    '''
    Adapter for hexgrad/Kokoro-82M via the `kokoro` pip package (KPipeline API).

    Capability flags (honest):
      - NO voice cloning: a `--voice` wav file is mapped to the nearest stock
        voice by detected gender, with a loud warning. A non-existent voice
        value that is not a stock voice id is refused with the list of ids.
      - NO fine-tuned models: only the 'internal' preset exists.
      - CPU-first: 82M params run 5-20x realtime on CPU. Only 'cuda' is passed
        through; any other accelerator (mps/rocm/xpu/jetson) falls back to CPU.
      - Stock voice selection: E2A_KOKORO_VOICE=<voice_id> (e.g. bm_lewis), or
        a `--voice` path whose stem is a stock id (e.g. .../bm_lewis.wav).
    '''

    def __init__(self, session:DictProxy):
        try:
            self.session = session
            self.cache_dir = tts_dir
            self.speakers_path = None
            self.speaker = None
            self.tts_key = self.session['model_cache']
            self.tts_zs_key = default_vc_model.rsplit('/', 1)[-1]
            self.pth_voice_file = None
            self.resampler_cache = {}
            self.resampled_wav_cache = {}
            self.audio_segments = []
            self.tts_engine = self.session['tts_engine']
            self.models = load_engine_presets(self.tts_engine)
            self.params = {}
            self.language = self.session.get('language')
            self.language_iso1 = self.session.get('language_iso1')
            if self.session.get('translate_enabled'):
                if self.session.get('translate'):
                    self.language = self.session['translate']
                if self.session.get('translate_iso1'):
                    self.language_iso1 = self.session['translate_iso1']
            if self.tts_engine not in default_engine_settings:
                error = f'Invalid tts_engine {self.tts_engine}.'
                raise ValueError(error)
            self.engine_langs = default_engine_settings[self.tts_engine].get('languages', {})
            if self.language not in self.engine_langs:
                error = f'Language {self.language} not supported by engine {self.tts_engine}.'
                raise ValueError(error)
            fine_tuned = self.session.get('fine_tuned')
            if fine_tuned not in self.models:
                error = f'Invalid fine_tuned model {fine_tuned}. Available models: {list(self.models.keys())}'
                raise ValueError(error)
            model_cfg = self.models[fine_tuned]
            self.params['samplerate'] = model_cfg['samplerate']
            self.params['speed'] = default_engine_settings[self.tts_engine]['speed']
            self.repo_id = model_cfg['repo']
            self.voice_ids = set(default_engine_settings[self.tts_engine]['voices'].keys())
            self.default_voice_id = model_cfg['voice']
            env_voice = os.environ.get('E2A_KOKORO_VOICE')
            if env_voice:
                if env_voice not in self.voice_ids:
                    error = f'E2A_KOKORO_VOICE={env_voice!r} is not a Kokoro stock voice. Available: {sorted(self.voice_ids)}'
                    raise ValueError(error)
                self.default_voice_id = env_voice
            # CPU-first: kokoro's KModel supports cpu/cuda; anything else (mps,
            # rocm, xpu, jetson) is not wired here — CPU is fast enough (5-20x
            # realtime) and keeps the GPU free for heavier engines.
            requested_device = self.session['device']
            self.device = requested_device if requested_device == devices['CUDA']['proc'] else devices['CPU']['proc']
            if self.device != requested_device:
                msg = f"Kokoro: device '{requested_device}' not supported by this adapter, using CPU."
                print(msg)
            # The loaded_tts cache is process-wide: key by device so a CPU-loaded
            # model is never handed to a CUDA session (or vice versa) unchanged.
            self.tts_key = f"{self.tts_key}-{self.device}"
            self.pipelines = {}
            self._mapped_voice_cache = {}
            # Duration substituted for a fragment kokoro cannot phonemize —
            # long enough to register as a beat in the narration, short enough
            # not to read as a gap. Overridable for tests.
            self._silence_ms = float(os.environ.get('E2A_KOKORO_SILENCE_MS', '250'))
            self.engine = self.load_engine()
        except Exception as e:
            error = f'__init__() error: {e}'
            raise ValueError(error)

    def load_engine(self)->Any:
        try:
            msg = f'Loading TTS {self.tts_key} model, it takes a while, please be patient…'
            print(msg)
            self.cleanup_memory()
            engine = loaded_tts.get(self.tts_key)
            if not engine:
                from kokoro import KModel
                engine = KModel(repo_id=self.repo_id).to(self.device).eval()
                loaded_tts[self.tts_key] = engine
            if engine:
                msg = f'TTS {self.tts_key} Loaded!'
                print(msg)
                return engine
            error = 'load_engine(): engine is None'
            raise RuntimeError(error)
        except Exception as e:
            error = f'load_engine() error: {e}'
            raise RuntimeError(error) from e

    def _set_voice(self, voice:str|None)->tuple:
        # Resolve any incoming voice value (None, stock id, id-named path, or a
        # cloning wav) to a Kokoro stock voice id. Kokoro cannot clone voices.
        if voice is None:
            return self.default_voice_id, None
        if voice in self.voice_ids:
            return voice, None
        stem = Path(voice).stem
        if stem in self.voice_ids:
            return stem, None
        if os.path.exists(voice):
            if voice in self._mapped_voice_cache:
                return self._mapped_voice_cache[voice], None
            from lib.classes.tts_engines.common.audio import detect_gender
            gender = detect_gender(voice)
            if gender:
                mapped = 'bf_emma' if gender == 'female' else 'bm_george'
                detail = ''
            else:
                # Inconclusive pitch: honour the configured default (which the
                # warning below claims was used) rather than assuming male.
                mapped = self.default_voice_id
                detail = ' Pitch analysis was inconclusive, so the default voice was used.'
            msg = (
                f"WARNING: Kokoro cannot clone voices. Reference voice {Path(voice).name!r} "
                f"is mapped to the nearest stock voice '{mapped}'.{detail} "
                f"Use E2A_KOKORO_VOICE=<id> to pick one of: {sorted(self.voice_ids)}"
            )
            print(msg)
            self._mapped_voice_cache[voice] = mapped
            return mapped, None
        error = (
            f"Kokoro has no voice {voice!r} and cannot clone. "
            f"Available stock voice ids: {sorted(self.voice_ids)}"
        )
        return None, error

    def _get_pipeline(self, voice_id:str)->Any:
        # One KPipeline per lang code, all sharing the single KModel. The lang
        # code drives G2P ('a' en-US vs 'b' en-GB) and is derived from the
        # voice id prefix so af_/am_ voices get American G2P automatically.
        prefix = voice_id[:1]
        lang_code = prefix if prefix in ('a', 'b') else self.engine_langs[self.language]
        pipeline = self.pipelines.get(lang_code)
        if pipeline is None:
            from kokoro import KPipeline
            pipeline = KPipeline(lang_code=lang_code, repo_id=self.repo_id, model=self.engine)
            self.pipelines[lang_code] = pipeline
        return pipeline

    def convert(self, sentence_file:str, sentence:str, **kwargs)->tuple:
        try:
            import torch
            from lib.classes.tts_engines.common.audio import is_audio_data_valid
            if self.engine:
                sentence_parts = self._split_sentence_on_sml(sentence)
                self.params['block_voice'] = kwargs.get('block_voice', self.session['voice'])
                if self.params.get('inline_voice'):
                    self.params['current_voice'] = self.params['inline_voice']
                else:
                    self.params['current_voice'], error = self._set_voice(self.params['block_voice'])
                    if self.params['current_voice'] is None and error is not None:
                        return False, error
                self.audio_segments = []
                for part in sentence_parts:
                    part = part.strip()
                    if not part:
                        continue
                    if SML_TAG_PATTERN.fullmatch(part):
                        success, error = self._convert_sml(part)
                        if not success:
                            return False, error
                        continue
                    if not any(c.isalnum() for c in part):
                        continue
                    else:
                        if part.endswith("'"):
                            part = part[:-1]
                        # inline [voice: path] tags store a raw path; resolve it
                        voice_id = self.params['current_voice']
                        if voice_id not in self.voice_ids:
                            voice_id, error = self._set_voice(voice_id)
                            if voice_id is None and error is not None:
                                return False, error
                        try:
                            pipeline = self._get_pipeline(voice_id)
                            chunks = []
                            # split_pattern=None: the fork feeds one sentence per
                            # call; KPipeline still yields multiple results when
                            # a sentence exceeds the 510-phoneme model window.
                            with torch.inference_mode():
                                for result in pipeline(part, voice=voice_id, speed=self.params['speed'], split_pattern=None):
                                    audio_chunk = result.audio
                                    if audio_chunk is not None and audio_chunk.numel() > 0:
                                        chunks.append(audio_chunk.detach().cpu())
                            if not chunks:
                                # A fragment whose phonemization yields nothing
                                # (bare syllable notation like "-yôr-", stray
                                # markup) must not abort a whole book: emit a
                                # short silence so the sync map keeps a real,
                                # nonzero duration for the fragment, and say so.
                                msg = (f'WARNING: unsynthesizable fragment, '
                                       f'emitting {int(self._silence_ms)} ms silence for: {part!r}')
                                print(msg)
                                chunks.append(torch.zeros(
                                    int(self.params['samplerate'] * self._silence_ms / 1000.0)
                                ))
                            audio_part = torch.cat(chunks, dim=-1)
                            if not is_audio_data_valid(audio_part):
                                error = 'audio_part not valid'
                                return False, error
                            part_tensor = self._tensor_type(audio_part).unsqueeze(0)
                            if part_tensor.numel() == 0:
                                error = 'part_tensor not valid'
                                return False, error
                            self.audio_segments.append(part_tensor)
                        except IndexError as e:
                            error = f'convert() error at segment "{part}": {e}'
                            return False, error
                        except Exception as e:
                            return False, self.log_exception(f'{self.__class__.__name__}.convert() part loop', e)
                if self.audio_segments:
                    segment_tensor = torch.cat(self.audio_segments, dim=-1)
                    if not self.audio_save(sentence_file, segment_tensor, self.params['samplerate']):
                        error = f'audio_save() error: cannot save {sentence_file}'
                        return False, error
                    self.audio_segments = []
                    if not os.path.exists(sentence_file):
                        error = f'Cannot create {sentence_file}'
                        return False, error
                return True, None
            else:
                error = f"TTS engine {self.tts_engine} failed to load!"
                return False, error
        except Exception as e:
            self.cleanup_memory()
            self.audio_segments = []
            return False, self.log_exception(f'{self.__class__.__name__}.convert()', e)

    def create_vtt(self, all_sentences:list)->bool:
        if self._build_vtt_file(all_sentences):
            return True
        return False
