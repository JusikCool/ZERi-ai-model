from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(r"C:\Users\user\Desktop\JERi\ZERi-ai-model\M2")
PREPROCESS_SCRIPT = ROOT_DIR / "preprocess_tft_fixed.py"
SMOKE_SCRIPT = ROOT_DIR / "smoke_test_tft_fixed.py"
TRAIN_SCRIPT = ROOT_DIR / "train_tft_fixed.py"
EVALUATE_SCRIPT = ROOT_DIR / "evaluate_tft_fixed.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full TFT-FIXED pipeline: preprocess, smoke test, train, and evaluate."
    )
    parser.add_argument("--input-data", type=Path, default=None)
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT_DIR / "artifacts")
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "runs" / "tft_fixed_pipeline")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--plot-ticker", type=str, default="AAPL")
    parser.add_argument("--skip-smoke-test", action="store_true")
    return parser.parse_args()


def run_stage(stage_name: str, command: list[str]) -> subprocess.CompletedProcess:
    print(f"\n=== {stage_name} ===")
    print(" ".join(command))
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.stdout:
        safe_text = result.stdout.encode("cp949", errors="replace").decode("cp949", errors="replace")
        print(safe_text, end="" if safe_text.endswith("\n") else "\n")
    if result.stderr:
        safe_err = result.stderr.encode("cp949", errors="replace").decode("cp949", errors="replace")
        print(safe_err, end="" if safe_err.endswith("\n") else "\n", file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"{stage_name} failed with exit code {result.returncode}.")
    return result


def extract_json_object(raw_text: str) -> dict:
    lines = raw_text.strip().splitlines()
    for start in range(len(lines)):
        candidate = "\n".join(lines[start:])
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError("Could not find a JSON object in the script output.")


def find_best_checkpoint(train_output_dir: Path, train_payload: dict) -> Path:
    best_path = train_payload.get("best_model_path")
    if best_path:
        best_checkpoint = Path(best_path)
        if best_checkpoint.exists():
            return best_checkpoint

    checkpoint_dir = train_output_dir / "checkpoints"
    checkpoints = sorted(checkpoint_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if checkpoints:
        return checkpoints[0]
    raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")


def main() -> None:
    args = parse_args()

    python_executable = sys.executable
    artifacts_dir = args.artifacts_dir
    train_output_dir = args.output_dir / "train"
    eval_output_dir = args.output_dir / "eval"

    preprocess_command = [python_executable, str(PREPROCESS_SCRIPT), "--output-dir", str(artifacts_dir)]
    if args.input_data is not None:
        preprocess_command.extend(["--input", str(args.input_data)])
    run_stage("Preprocess", preprocess_command)

    if not args.skip_smoke_test:
        run_stage("Smoke Test", [python_executable, str(SMOKE_SCRIPT)])

    train_command = [
        python_executable,
        str(TRAIN_SCRIPT),
        "--data-path",
        str(artifacts_dir / "tft_fixed_panel_ready.csv"),
        "--output-dir",
        str(train_output_dir),
    ]
    if args.max_epochs is not None:
        train_command.extend(["--max-epochs", str(args.max_epochs)])
    if args.batch_size is not None:
        train_command.extend(["--batch-size", str(args.batch_size)])
    if args.learning_rate is not None:
        train_command.extend(["--learning-rate", str(args.learning_rate)])

    train_result = run_stage("Train", train_command)
    train_payload = extract_json_object(train_result.stdout)
    best_checkpoint = find_best_checkpoint(train_output_dir=train_output_dir, train_payload=train_payload)

    evaluate_command = [
        python_executable,
        str(EVALUATE_SCRIPT),
        "--checkpoint-path",
        str(best_checkpoint),
        "--data-path",
        str(artifacts_dir / "tft_fixed_panel_ready.csv"),
        "--output-dir",
        str(eval_output_dir),
        "--plot-ticker",
        args.plot_ticker,
    ]
    evaluate_result = run_stage("Evaluate", evaluate_command)
    evaluate_payload = extract_json_object(evaluate_result.stdout)

    summary = {
        "artifacts_dir": str(artifacts_dir),
        "train_output_dir": str(train_output_dir),
        "eval_output_dir": str(eval_output_dir),
        "best_checkpoint_path": str(best_checkpoint),
        "prediction_csv_path": evaluate_payload.get("prediction_csv_path"),
        "quantile_metrics_csv_path": evaluate_payload.get("quantile_metrics_csv_path"),
        "summary_json_path": evaluate_payload.get("summary_json_path"),
        "primary_plot_path": evaluate_payload.get("primary_plot_path"),
    }

    print("\n=== Pipeline Completed ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nPipeline failed: {exc}", file=sys.stderr)
        raise
