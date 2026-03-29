from __future__ import annotations

from pathlib import Path


def find_default_dataset_root() -> Path | None:
    candidates = [
        Path("datasets/New Plant Diseases Dataset(Augmented)"),
        Path("dataset"),
        Path("datasets"),
    ]
    for candidate in candidates:
        if (candidate / "train").is_dir():
            return candidate
    return None


def list_model_candidates() -> list[str]:
    discovered: list[Path] = []

    model_dir = Path("models")
    if model_dir.is_dir():
        for suffix in ("*.pth", "*.pt", "*.ckpt"):
            discovered.extend(model_dir.rglob(suffix))

    workspace_root = Path(".")
    for suffix in ("*.pth", "*.pt", "*.ckpt"):
        discovered.extend(workspace_root.glob(suffix))

    unique = sorted({path.resolve() for path in discovered})
    return [str(path) for path in unique]


def parse_three_floats(raw_text: str, fallback: list[float]) -> list[float]:
    try:
        values = [float(item.strip()) for item in raw_text.split(",")]
        if len(values) != 3:
            return fallback
        return values
    except ValueError:
        return fallback


def load_class_names(dataset_root: str) -> list[str]:
    train_dir = Path(dataset_root) / "train"
    if not train_dir.is_dir():
        return []

    class_names = [folder.name for folder in train_dir.iterdir() if folder.is_dir()]
    return sorted(class_names)


def pretty_label(raw_label: str) -> str:
    if "___" in raw_label:
        plant, condition = raw_label.split("___", maxsplit=1)
    else:
        plant, condition = "Unknown", raw_label

    plant = plant.replace("_", " ").replace(",", "")
    condition = condition.replace("_", " ")
    return f"{plant} - {condition}"
