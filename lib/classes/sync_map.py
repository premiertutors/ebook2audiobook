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
import shutil
import subprocess

from datetime import datetime, timezone
from pathlib import Path

from lib.conf import default_audio_proc_format, sync_map_schema_version, prog_version
from lib.conf_models import SML_TAG_PATTERN


def _normalised_text(raw: str) -> str:
    """Text as spoken: SML tags removed, whitespace collapsed. Also the hash input."""
    return re.sub(r'\s+', ' ', SML_TAG_PATTERN.sub('', str(raw))).strip()


def _normalise_with_map(source: str) -> tuple[str, list[int]]:
    """Normalise a chapter and keep an index map onto the result.

    ``fragments[].src`` offsets must slice the normalised chapter text and
    reproduce the fragment's ``text`` verbatim, so the offsets have to be
    expressed in the coordinates of the *normalised* string — not the block text,
    which still carries the [pause]/[break] directives that are never spoken.

    Returns ``(normalised, index_map)`` where ``index_map[i]`` is the normalised
    index that ``source[i]`` maps to; a deleted character maps to the position the
    next surviving character occupies. The transform is character-for-character
    equivalent to :func:`_normalised_text` minus the final strip, which the caller
    applies once per chapter.
    """
    out: list[str] = []
    index_map = [0] * (len(source) + 1)
    i = 0
    length = len(source)
    while i < length:
        tag = SML_TAG_PATTERN.match(source, i)
        if tag and tag.end() > i:
            for k in range(i, tag.end()):
                index_map[k] = len(out)
            i = tag.end()
            continue
        index_map[i] = len(out)
        char = source[i]
        if char.isspace():
            # A whitespace run collapses to one space, exactly as re.sub(r'\s+', ' ')
            # would — including a leading run, which the caller's strip removes.
            if not out or out[-1] != ' ':
                out.append(' ')
        else:
            out.append(char)
        i += 1
    index_map[length] = len(out)
    return ''.join(out), index_map


def _slug(value: str) -> str:
    text = str(value)
    slug = re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')
    # Any non-ASCII letter/digit (CJK, Arabic, Cyrillic, ...) is dropped by the
    # substitution above and carries no representation in `slug`. Titles that
    # differ only in that dropped script — e.g. "中文1" and "Книга 1" both
    # slugging to "1" — would otherwise collide in any consumer keyed by
    # bookId, so fold a stable hash of the original value into the id whenever
    # slugging lost that information, not just when it emptied out entirely.
    lossy = any(ch.isalnum() and ord(ch) > 127 for ch in text)
    if slug and not lossy:
        return slug
    digest = hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]
    return f'{slug}-{digest}' if slug else f'book-{digest}'


def _sha256_file(path: str) -> str | None:
    try:
        digest = hashlib.sha256()
        with open(path, 'rb') as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b''):
                digest.update(chunk)
        return f'sha256:{digest.hexdigest()}'
    except Exception:
        return None


def _resolve_voice_path(session: dict, block_voice: str | None) -> str | None:
    """The voice file actually used for a block: its own override, else the
    session default, else the fine-tuned preset's voice — the same fallback
    chain ``convert_chapters2audio`` applies when it synthesises the block."""
    voice_path = block_voice or session.get('voice')
    if not voice_path:
        try:
            from lib.classes.tts_engines.common.preset_loader import load_engine_presets
            voice_path = (load_engine_presets(session['tts_engine'])
                          .get(session['fine_tuned'], {})
                          .get('voice'))
        except Exception:
            voice_path = None
    return voice_path


def _audio_stream_info(final_file: str) -> tuple[int, int]:
    """(sampleRate, channels) of the exported audiobook, with sane fallbacks.

    Mutagen cannot read stream properties for every container (notably
    Matroska/WebM), so ffprobe is tried next before falling back to the
    hardcoded default.
    """
    try:
        from mutagen import File as MutagenFile
        info = getattr(MutagenFile(final_file), 'info', None)
        rate = int(getattr(info, 'sample_rate', 0) or 0)
        channels = int(getattr(info, 'channels', 0) or 0)
        if rate >= 8000 and channels in (1, 2):
            return rate, channels
    except Exception:
        pass
    try:
        ffprobe = shutil.which('ffprobe')
        if ffprobe:
            probe = subprocess.run(
                [ffprobe, '-v', 'error', '-select_streams', 'a:0',
                 '-show_entries', 'stream=sample_rate,channels',
                 '-of', 'default=nokey=1:noprint_wrappers=1', final_file],
                capture_output=True, text=True,
            )
            if probe.returncode == 0:
                values = probe.stdout.split()
                if len(values) >= 2:
                    rate, channels = int(values[0]), int(values[1])
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
        normalised_chapters: dict[int, dict] = {}
        fragments: list[dict] = []
        pending: list[tuple[Path, str, int, int, str]] = []  # file, text, charStart, charEnd, href
        used_voice_paths: set = set()

        # Chapter ids must stay stable across split output parts: build_sync_map_file
        # is called once per part with a disjoint block_indices subset, so a counter
        # local to this call would renumber every chapter from ch0 in each part and
        # change ids whenever the split boundary moves. Numbering against the full,
        # unfiltered block list instead ties each id to the block's own identity.
        global_chapter_index: dict[int, int] = {}
        gci = 0
        for gi, gblock in enumerate(blocks):
            if gblock['keep'] and gblock['text'].strip():
                global_chapter_index[gi] = gci
                gci += 1

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

            # Re-express the spans in the coordinates of the NORMALISED chapter
            # text, which is the string the sidecar ships and the contract slices.
            chapter_text, index_map = _normalise_with_map(block['text'])
            lead = len(chapter_text) - len(chapter_text.lstrip())
            chapter_text = chapter_text.strip()

            first_fragment = len(pending)
            for sentence_idx, sentence in enumerate(block_sentences):
                if not any(c.isalnum() for c in str(sentence)):
                    continue
                audio_file = block_dir / f'{sentence_idx}.{default_audio_proc_format}'
                if not audio_file.is_file():
                    return False, f"Missing audio file for block {i}, sentence {sentence_idx}: {audio_file}"
                span_start, span_end = spans[sentence_idx]
                # A span that starts inside the stripped leading run (an SML tag,
                # whitespace, or both) maps to a position at or before `lead`; that
                # is the start of the stripped chapter_text, i.e. offset 0 — not a
                # negative index, which Python would silently wrap from the end.
                char_start = max(0, min(len(chapter_text), index_map[span_start] - lead))
                char_end = max(0, min(len(chapter_text), index_map[span_end] - lead))
                # Stripping the tags can expose whitespace at either edge; the
                # fragment's own text is stripped, so the span must be too.
                while char_start < char_end and chapter_text[char_start].isspace():
                    char_start += 1
                while char_end > char_start and chapter_text[char_end - 1].isspace():
                    char_end -= 1
                # The invariant the sidecar exists for. Checking it here rather than
                # trusting the arithmetic is the difference between a bad offset
                # failing the build and a bad offset drifting the highlight.
                expected = _normalised_text(sentence)
                if chapter_text[char_start:char_end] != expected:
                    return False, (
                        f"block {i} sentence {sentence_idx}: offsets "
                        f"[{char_start},{char_end}) slice "
                        f"{chapter_text[char_start:char_end]!r}, expected {expected!r}"
                    )
                pending.append((audio_file, str(sentence), char_start, char_end, href))
            if len(pending) == first_fragment:
                continue
            used_voice_paths.add(_resolve_voice_path(session, block.get('voice')))
            normalised_chapters[chapter_index] = {'href': href, 'text': chapter_text}

            chapters.append({
                'index': chapter_index,
                'id': f'ch{global_chapter_index[i]}',
                'title': str(source.get('title') or ''),
                'href': href,
                'startMs': 0,
                'endMs': 0,
                'firstFragment': first_fragment,
                'lastFragment': len(pending) - 1,
            })
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
        voice_path = _resolve_voice_path(session, None)

        engine = {
            'name': str(session['tts_engine']),
            'version': str(prog_version),
        }
        # engine.voiceHash identifies the voice(s) actually used by the included
        # blocks, not just the session default — a block-level override (or a
        # mix of them) must not be reported as the session voice.
        voice_paths = sorted({vp for vp in used_voice_paths if vp})
        if len(voice_paths) == 1:
            voice_hash = _sha256_file(voice_paths[0])
            if voice_hash:
                engine['voiceHash'] = voice_hash
        elif len(voice_paths) > 1:
            per_voice_hashes = [h for h in (_sha256_file(vp) for vp in voice_paths) if h]
            if per_voice_hashes:
                engine['voiceHash'] = 'sha256:' + hashlib.sha256(
                    '|'.join(per_voice_hashes).encode('utf-8')
                ).hexdigest()

        book_stem = session.get('filename_noext') or Path(final_file).stem
        if session.get('translate_enabled') and session.get('translate'):
            book_stem = f"{book_stem}_{session['translate']}"
        # The stem alone is the basename, so two different ebooks both called book.epub
        # would collide on one bookId and cross-contaminate sync data keyed on it. Fold in
        # an identity digest of the source file: same file -> same id (re-compiling with a
        # different voice stays stable); different files sharing a name diverge. A re-issued
        # edition (different bytes) becomes a new book, which is the honest reading of the
        # contract -- its offsets would not resolve against the old text anyway.
        source_path = session.get('ebook')
        source_hash = _sha256_file(source_path) if source_path else None
        if source_hash:
            book_stem = f"{book_stem}-{source_hash[len('sha256:'):][:8]}"
        book_id = _slug(book_stem)
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

        # fragments[].src indexes into each chapter's NORMALISED text, and the reader
        # has no other way to obtain that string — the EPUB it renders holds source
        # text, not this. contracts/schemas/normalised-text.schema.json freezes the
        # companion document; buildId ties the pair together, because offsets are
        # only valid against the normalisation that produced them.
        normalised_document = {
            'schemaVersion': sync_map_schema_version,
            'bookId': book_id,
            'buildId': build_id,
            'chapters': {
                chapter['id']: normalised_chapters[chapter['index']]
                for chapter in chapters
            },
        }

        # A sync map on disk without its companion is a build whose offsets nothing
        # can resolve, so the pair is written or neither is: both documents go to
        # temporary files first and are only renamed into place once both writes
        # succeed, so a mid-write failure (e.g. disk full) can't leave a truncated
        # or mismatched sidecar behind.
        # Replace only the filename's suffix, not the whole path: an unrestricted
        # string replace would also rewrite a directory component that happens
        # to contain the literal ".sync-map.json".
        sync_map_path_obj = Path(sync_map_path)
        normalised_path = str(sync_map_path_obj.with_name(
            sync_map_path_obj.name.replace('.sync-map.json', '.normalised-text.json')))
        normalised_tmp = normalised_path + '.tmp'
        sync_map_tmp = str(sync_map_path) + '.tmp'
        try:
            with open(normalised_tmp, 'w', encoding='utf-8') as handle:
                json.dump(normalised_document, handle, ensure_ascii=False, indent=2)
            with open(sync_map_tmp, 'w', encoding='utf-8') as handle:
                json.dump(document, handle, ensure_ascii=False, indent=2)
            os.replace(normalised_tmp, normalised_path)
            os.replace(sync_map_tmp, sync_map_path)
        finally:
            for tmp_path in (normalised_tmp, sync_map_tmp):
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        print(f'Sync map written: {sync_map_path} ({len(fragments)} fragments, {len(chapters)} chapter(s))')
        print(f'Normalised chapter text written: {normalised_path}')
        return True, None
    except Exception as e:
        return False, f'build_sync_map_file(): {e}'
