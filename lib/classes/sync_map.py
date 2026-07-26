"""Emit the read-along sync map (premiertutors/books `contracts/schemas/sync-map.schema.json`).

Timings come from the same source as the WebVTT: the duration of each per-fragment
audio clip, summed in reading order. Everything else the contract needs — stable
content-hash ids, chapter boundaries, and source character offsets — is derived
here, because upstream keeps none of it.

Two deliberate choices:

* ``audio.durationMs`` is the running sum of clip durations, not a probe of the
  exported m4b. The contract requires the last fragment to end exactly at
  ``durationMs``; the AAC re-encode differs from the clip sum by well under a
  millisecond, and a probe would break that invariant for no gain in accuracy.
* Offsets index into the block's normalised text (the string held in
  ``blocks[].text``), which is exactly what §2.3 of the plan specifies. They are
  recomputed by re-running the deterministic splitter, so a resumed run that never
  called it in this process still emits correct offsets. When the splitter cannot
  produce spans (a language pysbd does not cover, or the legacy packer path) the
  sync map is NOT written — a wrong offset silently highlights the wrong words.
"""

import hashlib
import json
import os
import re

from datetime import datetime, timezone
from pathlib import Path

from lib.conf import default_audio_proc_format, sync_map_schema_version, prog_version
from lib.conf_models import SML_TAG_PATTERN


def _normalised_text(raw: str) -> str:
    """Text as spoken: SML tags removed, whitespace collapsed. Also the hash input."""
    return re.sub(r'\s+', ' ', SML_TAG_PATTERN.sub('', str(raw))).strip()


def _slug(value: str) -> str:
    slug = re.sub(r'[^a-z0-9]+', '-', str(value).lower()).strip('-')
    slug = re.sub(r'^[^a-z0-9]+', '', slug)
    return slug or 'book'


def _sha256_file(path: str) -> str | None:
    try:
        digest = hashlib.sha256()
        with open(path, 'rb') as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b''):
                digest.update(chunk)
        return f'sha256:{digest.hexdigest()}'
    except Exception:
        return None


def _audio_stream_info(final_file: str) -> tuple[int, int]:
    """(sampleRate, channels) of the exported audiobook, with sane fallbacks."""
    try:
        from mutagen import File as MutagenFile
        info = getattr(MutagenFile(final_file), 'info', None)
        rate = int(getattr(info, 'sample_rate', 0) or 0)
        channels = int(getattr(info, 'channels', 0) or 0)
        if rate >= 8000 and channels in (1, 2):
            return rate, channels
    except Exception:
        pass
    return 24000, 1


def build_sync_map_file(session: dict, sync_map_path: str, get_sentences, session_id: str,
                        final_file: str, block_indices: set = None) -> tuple:
    """Write ``sync_map_path``. Returns (ok, error).

    The block/fragment traversal mirrors ``build_vtt_file`` exactly so the two
    artefacts can never disagree about which clip is which.
    """
    try:
        from lib.classes.tts_engines.common.audio import get_audiolist_duration

        sentences_dir = Path(session['sentences_dir'])
        blocks = session['blocks_current']['blocks']
        sources = list(session.get('block_sources') or [])

        chapters: list[dict] = []
        chapter_source_block: dict[int, int] = {}
        fragments: list[dict] = []
        pending: list[tuple[Path, str, int, int, str]] = []  # file, text, charStart, charEnd, href

        chapter_index = 0
        for i, block in enumerate(blocks):
            if not (block['keep'] and block['text'].strip()):
                continue
            if block_indices is not None and i not in block_indices:
                continue
            block_dir = sentences_dir / str(block['id'])
            if not block_dir.is_dir():
                return False, f"Missing audio directory for block {i} (id {block['id']}): {block_dir}"

            block_sentences = block.get('sentences') or []
            spans = None
            recomputed = get_sentences(session_id, block['text'], with_spans=True)
            if isinstance(recomputed, tuple):
                resplit, spans = recomputed
                if spans is not None and resplit != list(block_sentences):
                    # The splitter is deterministic; a mismatch means the audio on
                    # disk was produced by different settings, so the offsets would
                    # not describe it. Refuse rather than emit a plausible lie.
                    return False, (
                        f"block {i} re-split into {len(resplit)} fragments but "
                        f"{len(block_sentences)} were synthesised — sync map would be misaligned"
                    )
            if spans is None:
                return False, (
                    f"block {i}: no character spans available (language "
                    f"{session.get('language_iso1')!r} is outside the sentence splitter); "
                    "refusing to emit fragments[].src offsets that would be guesses"
                )

            # block_sources is aligned with blocks_orig, and blocks_current is a
            # deepcopy of it, so the block's own index is the right key — not the
            # count of kept chapters.
            source = sources[i] if i < len(sources) else None
            if not source or not source.get('href'):
                return False, (
                    f"block {i}: no source href recorded for this chapter — "
                    "cannot fill the required chapters[].href / fragments[].src.href"
                )
            href = source['href']

            first_fragment = len(pending)
            for sentence_idx, sentence in enumerate(block_sentences):
                if not any(c.isalnum() for c in str(sentence)):
                    continue
                audio_file = block_dir / f'{sentence_idx}.{default_audio_proc_format}'
                if not audio_file.is_file():
                    return False, f"Missing audio file for block {i}, sentence {sentence_idx}: {audio_file}"
                char_start, char_end = spans[sentence_idx]
                pending.append((audio_file, str(sentence), char_start, char_end, href))
            if len(pending) == first_fragment:
                continue

            chapters.append({
                'index': chapter_index,
                'id': f'ch{chapter_index}',
                'title': str(source.get('title') or ''),
                'href': href,
                'startMs': 0,
                'endMs': 0,
                'firstFragment': first_fragment,
                'lastFragment': len(pending) - 1,
            })
            chapter_source_block[chapter_index] = i
            chapter_index += 1

        if not pending or not chapters:
            return False, 'no fragments to write'

        durations = get_audiolist_duration([str(p) for p, *_ in pending])

        # Cumulative seconds rounded to whole ms at each boundary: every fragment's
        # start is the previous fragment's end by construction, so the tiling
        # invariant in contracts/validate.py holds exactly.
        chapter_of = {}
        for chapter in chapters:
            for seq in range(chapter['firstFragment'], chapter['lastFragment'] + 1):
                chapter_of[seq] = chapter
        ordinal_in_chapter: dict[int, int] = {}
        elapsed = 0.0
        for seq, (audio_file, raw_text, char_start, char_end, href) in enumerate(pending):
            start_ms = round(elapsed * 1000)
            elapsed += durations.get(os.path.realpath(audio_file), 0.0)
            end_ms = round(elapsed * 1000)
            if end_ms <= start_ms:
                return False, f'fragment {seq} has a non-positive duration ({audio_file})'
            chapter = chapter_of[seq]
            ci = chapter['index']
            ordinal = ordinal_in_chapter.get(ci, 0)
            ordinal_in_chapter[ci] = ordinal + 1
            text = _normalised_text(raw_text)
            if not text:
                return False, f'fragment {seq} normalised to an empty string'
            digest = hashlib.sha256(text.encode('utf-8')).hexdigest()[:8]
            fragments.append({
                'id': f"{chapter['id']}-f{ordinal}-{digest}",
                'ci': ci,
                'seq': seq,
                'startMs': start_ms,
                'endMs': end_ms,
                'text': text,
                'src': {'href': href, 'charStart': char_start, 'charEnd': char_end},
            })

        for chapter in chapters:
            chapter['startMs'] = fragments[chapter['firstFragment']]['startMs']
            chapter['endMs'] = fragments[chapter['lastFragment']]['endMs']

        sample_rate, channels = _audio_stream_info(final_file)
        voice_path = session.get('voice')
        if not voice_path:
            try:
                from lib.classes.tts_engines.common.preset_loader import load_engine_presets
                voice_path = (load_engine_presets(session['tts_engine'])
                              .get(session['fine_tuned'], {})
                              .get('voice'))
            except Exception:
                voice_path = None

        engine = {
            'name': str(session['tts_engine']),
            'version': str(prog_version),
        }
        voice_hash = _sha256_file(voice_path) if voice_path else None
        if voice_hash:
            engine['voiceHash'] = voice_hash

        book_id = _slug(session.get('filename_noext') or Path(final_file).stem)
        voice_label = _slug(Path(voice_path).stem) if voice_path else _slug(str(session['fine_tuned']))
        build_id = '__'.join([
            datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%SZ'),
            str(session['tts_engine']),
            f'voice-{voice_label}',
        ])

        document = {
            'schemaVersion': sync_map_schema_version,
            'bookId': book_id,
            'buildId': build_id,
            'audio': {
                'container': str(session['output_format']),
                'sampleRate': sample_rate,
                'channels': channels,
                'durationMs': fragments[-1]['endMs'],
            },
            'engine': engine,
            'text': {
                'normalizer': 'e2a-normalize_text+pysbd',
                'normalizerVersion': str(prog_version),
            },
            'wordLevel': False,
            'chapters': chapters,
            'fragments': fragments,
        }

        with open(sync_map_path, 'w', encoding='utf-8') as handle:
            json.dump(document, handle, ensure_ascii=False, indent=2)
        print(f'Sync map written: {sync_map_path} ({len(fragments)} fragments, {len(chapters)} chapter(s))')

        # fragments[].src indexes into each chapter's NORMALISED text, and the
        # reader has no other way to obtain that string — the EPUB it renders holds
        # the source text, not this. Ship it beside the sync map so the offsets are
        # resolvable; the schema sets additionalProperties:false, so it cannot live
        # inside the document itself.
        try:
            normalised_path = str(sync_map_path).replace('.sync-map.json', '.normalised-text.json')
            payload = {}
            for chapter in chapters:
                block = blocks[chapter_source_block[chapter['index']]]
                payload[chapter['id']] = {'href': chapter['href'], 'text': block['text']}
            with open(normalised_path, 'w', encoding='utf-8') as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            print(f'Normalised chapter text written: {normalised_path}')
        except Exception as e:
            print(f'build_sync_map_file(): normalised text sidecar not written: {e}')
        return True, None
    except Exception as e:
        return False, f'build_sync_map_file(): {e}'
