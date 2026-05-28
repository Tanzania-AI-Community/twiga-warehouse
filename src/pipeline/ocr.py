import subprocess
from pathlib import Path


def ensure_ocr_pdf(
    input_path: Path,
    output_path: Path | None = None,
    *,
    force_ocr: bool = True,
) -> Path:
    resolved_output_path = output_path or input_path.with_name(f"{input_path.stem}_ocr{input_path.suffix}")
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    command = ["ocrmypdf"]
    if force_ocr:
        command.append("--force-ocr")
    command.extend([str(input_path), str(resolved_output_path)])

    try:
        subprocess.run(args=command, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ocrmypdf is not installed or not available in PATH. "
            "Install it from https://github.com/ocrmypdf/OCRmyPDF."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ocrmypdf failed with exit code {exc.returncode}") from exc

    return resolved_output_path
