"""Sentence-parallel synthesis for engines that declare it safe.

The per-block sentence loop in core is embarrassingly parallel for stateless
engines: each sentence renders to its own numbered file inside the block dir,
and core reconciles a block by scanning for the files already on disk — so
out-of-order completion needs no new bookkeeping.

The one piece of cross-sentence engine state is an inline [voice: …] scope.
Core keeps any block whose scope stays open across a sentence boundary in the
sequential lane, since two workers cannot share that state.

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


def _convert_one(sentence_file: str, sentence: str, block_voice) -> tuple:
    try:
        # A worker is reused for unrelated sentences, so an inline [voice: …]
        # scope must never bleed from one task into the next. Core keeps any
        # block whose scope crosses a sentence boundary out of this lane, so
        # clearing the sticky param here is always the correct starting state.
        params = getattr(_WORKER_TTS.engine, "params", None)
        if isinstance(params, dict):
            params["inline_voice"] = None
        return _WORKER_TTS.convert_sentence2audio(
            sentence_file, sentence, block_voice=block_voice
        )
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
        """Render `tasks` = [(sentence_file, sentence, block_voice)] concurrently.

        Calls on_done(sentence_file) as each finishes (progress display).
        Returns (True, None) or (False, error) on the first failure; remaining
        futures are cancelled, already-written files are kept for resume.
        """
        from concurrent.futures import as_completed

        futures = {
            self.executor.submit(_convert_one, f, s, v): f for (f, s, v) in tasks
        }
        error = None
        for future in as_completed(futures):
            ok, err = future.result()
            if not ok or cancelled():
                error = err if not ok else "Conversion Cancelled"
                for pending in futures:
                    pending.cancel()
                break
            on_done(futures[future])
        if error:
            return False, error
        return True, None

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
