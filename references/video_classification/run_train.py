"""train.pyのYAML config wrapper.

Usage:
    python run_train.py --config configs/baseline.yaml
    python run_train.py --config configs/baseline.yaml --override epochs=30 lr=0.001
    python run_train.py --config configs/baseline.yaml --tmux my_session
    python run_train.py --config configs/baseline.yaml --dry-run

Examples:
    # 基本実行
    python run_train.py -c configs/baseline.yaml

    # パラメータ上書き付き
    python run_train.py -c configs/baseline.yaml -o batch_size=16 epochs=30

    # tmuxセッション内で実行 (SSH切断耐性)
    python run_train.py -c configs/baseline.yaml --tmux train_exp1

    # コマンド確認のみ
    python run_train.py -c configs/baseline.yaml --dry-run
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

# train.pyの引数のうち、store_true で扱うbooleanフラグ
_BOOLEAN_FLAGS = frozenset(
    {
        "cache_dataset",
        "sync_bn",
        "test_only",
        "use_deterministic_algorithms",
        "amp",
    }
)


def load_config(config_path: Path) -> dict:
    """YAMLファイルを読み込み辞書として返す."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    """'key=value' 形式の上書きリストをconfigに適用する."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override format must be 'key=value', got: '{item}'")
        key, value = item.split("=", 1)
        if key not in config:
            raise ValueError(f"Unknown config key: '{key}'")
        original = config[key]
        config[key] = _cast_value(value, type(original))
    return config


def _cast_value(value: str, target_type: type) -> object:
    """文字列値を元の型にキャストする."""
    if target_type is bool:
        return value.lower() in ("true", "1", "yes")
    if target_type is list:
        return [int(x.strip()) for x in value.strip("[]").split(",")]
    return target_type(value)


def build_args(config: dict) -> list[str]:
    """config辞書をtrain.pyのCLI引数リストに変換する."""
    args: list[str] = []
    for key, value in config.items():
        cli_key = f"--{key.replace('_', '-')}"
        if key in _BOOLEAN_FLAGS:
            if value:
                args.append(cli_key)
        elif isinstance(value, list):
            args.append(cli_key)
            args.extend(str(v) for v in value)
        elif isinstance(value, str) and value == "":
            # 空文字列はスキップ (e.g. resume="")
            continue
        else:
            args.extend([cli_key, str(value)])
    return args


def build_command(train_args: list[str], script_dir: Path) -> list[str]:
    """train.pyの完全なコマンドを構築する."""
    train_py = str(script_dir / "train.py")
    return [sys.executable, train_py, *train_args]


def run_in_tmux(
    cmd: list[str], session_name: str, working_dir: Path, log_file: Path
) -> None:
    """tmuxセッション内でコマンドを実行する."""
    shell_cmd = " ".join(shlex.quote(c) for c in cmd)
    full_cmd = f"cd {shlex.quote(str(working_dir))} && {shell_cmd} 2>&1 | tee {shlex.quote(str(log_file))}"
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session_name, full_cmd],
        check=True,
    )
    print(f"tmuxセッション '{session_name}' で学習を開始しました")
    print(f"  ログ: {log_file}")
    print(f"  進捗確認: tmux capture-pane -t {session_name} -p | tail -20")
    print(f"  セッション接続: tmux attach -t {session_name}")


def parse_args() -> argparse.Namespace:
    """コマンドライン引数をパースする."""
    parser = argparse.ArgumentParser(
        description="train.py の YAML config wrapper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        required=True,
        help="YAML config file path",
    )
    parser.add_argument(
        "-o",
        "--override",
        nargs="*",
        default=[],
        help="key=value overrides",
    )
    parser.add_argument(
        "--tmux",
        type=str,
        metavar="SESSION",
        help="tmuxセッション内で実行",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="コマンドを表示するのみ (実行しない)",
    )
    return parser.parse_args()


def main() -> None:
    """configを読み込みtrain.pyを実行する."""
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    config = load_config(args.config)
    if args.override:
        config = apply_overrides(config, args.override)

    train_args = build_args(config)
    cmd = build_command(train_args, script_dir)

    print("=== Training Config ===")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"\n=== Command ===\n  {' '.join(shlex.quote(c) for c in cmd)}\n")

    if args.dry_run:
        return

    if args.tmux:
        log_file = script_dir / f"{args.tmux}.log"
        run_in_tmux(cmd, args.tmux, script_dir, log_file)
    else:
        os.chdir(script_dir)
        os.execv(sys.executable, cmd)


if __name__ == "__main__":
    main()
