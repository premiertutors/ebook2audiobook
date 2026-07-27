"""Sentence-parallel synthesis for engines that declare it safe.

The per-block sentence loop in core is embarrassingly parallel for stateless
engines: each sentence renders to its own numbered file inside the block dir,
and core reconciles a block by scanning for the files already on disk — so
out-of-order completion needs no new bookkeeping.

The one piece of cross-sentence engine state is an inline [voice: …] scope.
Two workers cannot share it, so the parent replays the tag stream itself
(resolve_inline_voices) and tells each task which scope it starts in — the
worker installs that before rendering instead of inheriting whatever its
previous sentence left behind. Every block therefore stays in this lane; no
block needs a sequential fallback, and the parent never loads a model.

Engines opt in with `"parallel_safe": True` in their default settings; the
worker count comes from E2A_PARALLEL_WORKERS (0/1 = the classic sequential
path, untouched). Each worker process builds its own TTSManager — models are
NOT shared across processes — so this is only sensible for small models
(kokoro's 82M is ~400 MB per worker; XTTS would not be).

Spawn safety: workers are plain module functions in this file, the session is
a multiprocessing Manager proxy (picklable), and app.py guards __main__, so
the spawn re-import is inert.
"""

from __future__ import annotations

import os

_WORKER_TTS = None


def _init_worker(session, torch_threads: int) -> None:
    # One TTSManager (and thus one model) per worker process, built once.
    # Capping intra-op torch threads stops N workers x M threads thrashing
    # a machine that is also assembling audio.
    global _WORKER_TTS
    import torch

    torch.set_num_threads(max(1, torch_threads))
    # TTSRegistry.ENGINES is filled by import side effect: every engine class
    # self-registers through TTSRegistry.__init_subclass__ when its module is
    # imported, and the engines package __init__ imports them all. The main
    # process gets this for free (core imports tts_engines.common.*, which
    # executes the package __init__); a spawned worker imports only
    # tts_manager, so without this the registry is empty and TTSManager raises
    # "Invalid tts_engine ... Expected one of: ".
    import lib.classes.tts_engines  # noqa: F401  (registers every engine)
    from lib.classes.tts_manager import TTSManager

    _WORKER_TTS = TTSManager(session)


def resolve_inline_voices(sentences, indices) -> dict:
    """Map sentence index -> the ``[voice:...]`` scope in force at its START.

    Engines keep the inline voice on the engine *instance*
    (``self.params['inline_voice']``): an opening ``[voice:path]`` sets it and
    only a ``[/voice]`` clears it, so in the sequential lane the scope simply
    carries from one ``convert()`` call to the next. Replaying the tag stream
    here lets the parent hand each sentence its own starting scope, so a block
    whose scope spans sentences can still be rendered out of order.

    ``indices`` must be the sentences the engine actually converts, in order —
    tags inside skipped sentences are never seen by the engine, so they must
    not move this state machine either.

    Mirrors ``TTSUtils._convert_sml``: values are ``os.path.abspath``'d, and
    ``[/voice]`` reverts flatly to "no inline voice" — the engine keeps no
    stack, so nested scopes collapse rather than restoring an outer one.
    """
    from lib.conf_models import SML_TAG_PATTERN

    entry = {}
    inline = None
    for j in indices:
        entry[j] = inline
        for m in SML_TAG_PATTERN.finditer(sentences[j]):
            if m.group("tag") != "voice":
                continue
            if m.group("close"):
                inline = None
            else:
                value = (m.group("value") or "").strip()
                if value:
                    inline = os.path.abspath(value)
    return entry


def _convert_one(sentence_file: str, sentence: str, block_voice, inline_voice) -> tuple:
    try:
        # A worker is reused for unrelated sentences, so the sticky inline
        # [voice: …] scope must never bleed from one task into the next. Setting
        # it per task both clears the previous sentence's scope and restores the
        # one this sentence actually starts in. The tags stay in the sentence
        # text: a mid-sentence voice change renders as several parts
        # concatenated into one file, which a single scalar could not express.
        params = getattr(_WORKER_TTS.engine, "params", None)
        if isinstance(params, dict):
            params["inline_voice"] = inline_voice
        # Render to a sibling temp file and rename only on success. Resume in
        # this lane treats a sentence file's existence as "done", so a file
        # half-written when the process was killed would be skipped forever and
        # fed to ffmpeg as-is. os.replace is atomic within the directory, so a
        # file under the real name is always a complete render. The suffix
        # keeps the extension (engines pick the container from it) and is
        # dot-prefixed so it can never be mistaken for sentence `N`.
        head, tail = os.path.split(sentence_file)
        stem, ext = os.path.splitext(tail)
        tmp_file = os.path.join(head, f".{stem}.part{ext}")
        try:
            ok, err = _WORKER_TTS.convert_sentence2audio(
                tmp_file, sentence, block_voice=block_voice
            )
            if ok and os.path.exists(tmp_file):
                os.replace(tmp_file, sentence_file)
                return ok, err
            return False, err or f"no audio produced for {os.path.basename(sentence_file)}"
        finally:
            if os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except OSError:
                    pass
    except Exception as e:  # surfaced to the parent as an error result
        return False, f"parallel convert failed for {os.path.basename(sentence_file)}: {e}"


class ParallelSentencePool:
    """A lazily-created, process-wide pool reused across blocks.

    Creating the pool per block would pay the worker init (torch import +
    model load) thousands of times over a long book; created once, workers
    live for the whole conversion.
    """

    def __init__(self, session, workers: int):
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp

        cpu = os.cpu_count() or 4
        torch_threads = max(1, cpu // workers)
        self.workers = workers
        self.executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
            initializer=_init_worker,
            initargs=(session, torch_threads),
        )

    def convert_block(self, tasks, cancelled, on_done) -> tuple:
        """Render `tasks` = [(sentence_file, sentence, block_voice, inline_voice)].

        Calls on_done(sentence_file) as each finishes (progress display).
        Returns (True, None) or (False, error) on the first failure; queued
        futures are cancelled and the already-running ones are drained before
        returning, so no worker is still writing sentence files once the caller
        sees the failure. Files written by then are kept for resume.
        """
        from concurrent.futures import as_completed, wait

        futures = {
            self.executor.submit(_convert_one, f, s, v, iv): f for (f, s, v, iv) in tasks
        }
        error = None
        for future in as_completed(futures):
            ok, err = future.result()
            if not ok or cancelled():
                error = err if not ok else "Conversion Cancelled"
                for pending in futures:
                    pending.cancel()
                # cancel() is a no-op for a task already running; wait() lets
                # those finish (one sentence each) so a retry cannot race a
                # surviving worker on the same paths.
                wait(futures)
                break
            on_done(futures[future])
        if error:
            return False, error
        return True, None

    def shutdown(self) -> None:
        # wait=True so the pool is fully torn down before the conversion call
        # reports completion; convert_block has already drained, so this only
        # costs anything on the exception path.
        self.executor.shutdown(wait=True, cancel_futures=True)
