#!/usr/bin/env python3
"""
Скачивание модели Qwen3-ASR с HuggingFace Hub.

Использование:
    python download_model.py --model Qwen/Qwen3-ASR-0.6B --output ../models/qwen3-asr-0.6b

Зависимости:
    pip install huggingface_hub
"""

import argparse
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Ошибка: необходимо установить huggingface_hub:")
    print("   pip install huggingface_hub")
    exit(1)


def main():
    parser = argparse.ArgumentParser(description="Скачивание модели Qwen3-ASR")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="Qwen/Qwen3-ASR-0.6B",
        help="ID модели на HuggingFace (default: Qwen/Qwen3-ASR-0.6B)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Путь для сохранения модели"
    )
    parser.add_argument("--revision", type=str, default="main", help="Ветка или тег")

    args = parser.parse_args()

    # Определяем путь для сохранения
    if args.output:
        output_dir = Path(args.output)
    else:
        model_name = args.model.split("/")[-1].lower()
        output_dir = Path(__file__).parent.parent / "models" / model_name

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Скачивание модели: {args.model}")
    print(f"Директория: {output_dir}")
    print()

    # Скачиваем модель
    snapshot_download(
        repo_id=args.model,
        local_dir=str(output_dir),
        revision=args.revision,
        # Внимание: в репозиториях HF часто есть важные *.txt (например, merges.txt).
        ignore_patterns=["*.md", ".git*"],
    )

    print()
    print(f"Модель скачана в: {output_dir}")

    # Список файлов
    print("\nФайлы модели:")
    for f in sorted(output_dir.rglob("*")):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"   {f.relative_to(output_dir)} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
