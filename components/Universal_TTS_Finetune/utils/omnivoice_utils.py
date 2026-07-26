from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Callable


DEFAULT_BASE_MODEL = 'k2-fsa/OmniVoice'
DEFAULT_AUDIO_TOKENIZER = 'eustlb/higgs-audio-v2-tokenizer'


def _read_ljspeech_metadata(path:Path)->list[tuple[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f'OmniVoice metadata file not found: {path}')
    samples:list[tuple[str, str]] = []
    with path.open('r', encoding='utf-8', newline='') as handle:
        for row in csv.reader(handle, delimiter='|'):
            if len(row) < 2:
                continue
            sample_id = row[0].strip()
            text = (row[2] if len(row) > 2 and row[2].strip() else row[1]).strip()
            if sample_id and text:
                samples.append((sample_id, text))
    return samples


def create_omnivoice_manifests(
    dataset_dir:Path,
    workspace_dir:Path,
    language:str,
)->dict[str, Any]:
    """Convert a prepared LJSpeech dataset into OmniVoice train/dev JSONL."""
    workspace_dir.mkdir(parents=True, exist_ok=True)
    split_sources = {
        'train': dataset_dir / 'metadata_train.csv',
        'dev': dataset_dir / 'metadata_val.csv',
    }
    if not split_sources['train'].is_file():
        split_sources['train'] = dataset_dir / 'metadata.csv'

    result:dict[str, Any] = {'sample_counts': {}}
    seen_ids:set[str] = set()
    for split, metadata_path in split_sources.items():
        samples = (
            _read_ljspeech_metadata(metadata_path) if metadata_path.is_file() else []
        )
        output_path = workspace_dir / f'{split}.jsonl'
        count = 0
        with output_path.open('w', encoding='utf-8') as handle:
            for sample_id, text in samples:
                # Some older prepared datasets use the complete filename as ID.
                wav_name = sample_id if Path(sample_id).suffix else f'{sample_id}.wav'
                audio_path = (dataset_dir / 'wavs' / wav_name).resolve()
                if not audio_path.is_file():
                    raise FileNotFoundError(
                        f'Audio referenced by {metadata_path.name} was not found: {audio_path}'
                    )
                unique_key = str(audio_path)
                if split == 'dev' and unique_key in seen_ids:
                    continue
                if split == 'train':
                    seen_ids.add(unique_key)
                record = {
                    'id': sample_id,
                    'audio_path': str(audio_path),
                    'text': text,
                    'language_id': language,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                count += 1
        if split == 'train' and count == 0:
            raise ValueError(
                'The prepared dataset contains no OmniVoice training samples.'
            )
        result[f'{split}_jsonl'] = str(output_path)
        result['sample_counts'][split] = count
    return result


def build_omnivoice_configs(
    workspace_dir:Path,
    *,
    token_dir:Path,
    output_dir:Path,
    base_model:str,
    restore_path:str | None,
    steps:int,
    grad_accum:int,
    include_dev:bool,
    extra_overrides:dict[str, Any],
)->tuple[Path, Path, dict[str, Any]]:
    """Write self-contained OmniVoice train/data configs."""
    train_config:dict[str, Any] = {
        'llm_name_or_path': 'Qwen/Qwen3-0.6B',
        'audio_vocab_size': 1025,
        'audio_mask_id': 1024,
        'num_audio_codebook': 8,
        'audio_codebook_weights': [8, 8, 6, 6, 4, 4, 2, 2],
        'drop_cond_ratio': 0.1,
        'prompt_ratio_range': [0.0, 0.3],
        'mask_ratio_range': [0.0, 1.0],
        'language_ratio': 0.8,
        'use_pinyin_ratio': 0.0,
        'instruct_ratio': 0.0,
        'only_instruct_ratio': 0.0,
        'resume_from_checkpoint': restore_path,
        'init_from_checkpoint': None if restore_path else base_model,
        'learning_rate': 1e-5,
        'weight_decay': 0.01,
        'max_grad_norm': 1.0,
        'steps': steps,
        'seed': 42,
        'warmup_type': 'ratio',
        'warmup_ratio': 0.01,
        'warmup_steps': 0,
        'batch_tokens': 8192,
        'gradient_accumulation_steps': grad_accum,
        'num_workers': 2,
        'mixed_precision': 'bf16',
        'allow_tf32': True,
        'logging_steps': 25,
        'eval_steps': max(50, min(500, steps // 10 or 1)),
        'save_steps': max(50, min(500, steps // 10 or 1)),
        'keep_last_n_checkpoints': 3,
        # The official SDPA config differs by selecting this attention backend.
        'attn_implementation': 'sdpa',
        'max_sample_tokens': 2000,
        'min_sample_tokens': 50,
        'max_batch_size': 64,
    }
    allowed = set(train_config)
    unused = {
        key: value for key, value in extra_overrides.items() if key not in allowed
    }
    train_config.update(
        {key: value for key, value in extra_overrides.items() if key in allowed}
    )

    data_config:dict[str, Any] = {
        'train': [{'manifest_path': [str(token_dir / 'train' / 'data.lst')]}],
        'dev': (
            [{'manifest_path': [str(token_dir / 'dev' / 'data.lst')]}]
            if include_dev
            else []
        ),
    }
    train_path = workspace_dir / 'train_config.json'
    data_path = workspace_dir / 'data_config.json'
    train_path.write_text(json.dumps(train_config, indent=2), encoding='utf-8')
    data_path.write_text(json.dumps(data_config, indent=2), encoding='utf-8')
    return train_path, data_path, unused


def calculate_training_steps(epochs:int, sample_count:int, batch_size:int)->int:
    return max(1, int(epochs) * math.ceil(max(1, sample_count) / max(1, batch_size)))


def build_tokenize_command(
    python_executable:str,
    *,
    input_jsonl:Path,
    token_dir:Path,
    split:str,
    tokenizer_path:str,
)->list[str]:
    return [
        python_executable,
        '-m',
        'omnivoice.scripts.extract_audio_tokens',
        '--input_jsonl',
        str(input_jsonl),
        '--tar_output_pattern',
        str(token_dir / split / 'audios' / 'shard-%06d.tar'),
        '--jsonl_output_pattern',
        str(token_dir / split / 'txts' / 'shard-%06d.jsonl'),
        '--tokenizer_path',
        tokenizer_path,
        '--nj_per_gpu',
        '1',
        '--shuffle',
        'True' if split == 'train' else 'False',
    ]


def build_train_command(
    python_executable:str,
    *,
    train_config:Path,
    data_config:Path,
    output_dir:Path,
)->list[str]:
    return [
        python_executable,
        '-m',
        'accelerate.commands.launch',
        '--num_processes',
        '1',
        '-m',
        'omnivoice.cli.train',
        '--train_config',
        str(train_config),
        '--data_config',
        str(data_config),
        '--output_dir',
        str(output_dir),
    ]


def find_omnivoice_checkpoint(output_dir:Path)->Path:
    """Find the newest complete Hugging Face-style OmniVoice checkpoint."""
    candidates:list[Path] = []
    for config_path in output_dir.rglob('config.json'):
        directory = config_path.parent
        if (directory / 'model.safetensors').is_file() or any(
            directory.glob('model-*.safetensors')
        ):
            candidates.append(directory)
    if not candidates:
        raise FileNotFoundError(
            f'No complete OmniVoice checkpoint was produced under {output_dir}.'
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def package_omnivoice_checkpoint(
    checkpoint_dir:Path,
    ready_dir:Path,
    *,
    copy_function:Callable[..., Any] = shutil.copy2,
)->Path:
    destination = ready_dir / 'model'
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(checkpoint_dir, destination, copy_function=copy_function)
    return destination
