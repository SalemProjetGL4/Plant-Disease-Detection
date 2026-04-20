from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub


DATASET_HANDLE = "vipoooool/new-plant-diseases-dataset"

# Keep extraction path short on Windows to avoid MAX_PATH issues.
DOWNLOAD_DIR = Path("C:/kgh")

DATASETS_ROOT = Path("datasets")
NEW_PLANT_DISEASES_ROOT = DATASETS_ROOT / "New Plant Diseases"


def _find_split_dir(root: Path, split_name: str) -> Path:
	direct_path = root / split_name
	if direct_path.is_dir():
		return direct_path

	matches = [candidate for candidate in root.rglob(split_name) if candidate.is_dir()]
	if not matches:
		raise FileNotFoundError(f"Could not find '{split_name}' inside {root}")

	return sorted(matches, key=lambda item: (len(item.parts), str(item)))[0]


def _copy_split(source: Path, target: Path) -> None:
	target.parent.mkdir(parents=True, exist_ok=True)
	shutil.copytree(source, target, dirs_exist_ok=True)


def main() -> None:
	DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

	try:
		downloaded_root = Path(kagglehub.dataset_download(DATASET_HANDLE, output_dir=str(DOWNLOAD_DIR)))
	except FileNotFoundError as exc:
		raise RuntimeError(
			"Dataset extraction failed due to a long file path on Windows. "
			"Enable Win32 long paths or keep output_dir very short (e.g. C:/kgh)."
		) from exc

	train_source = _find_split_dir(downloaded_root, "train")
	valid_source = _find_split_dir(downloaded_root, "valid")
	test_source = _find_split_dir(downloaded_root, "test")

	train_target = NEW_PLANT_DISEASES_ROOT / "train"
	valid_target = NEW_PLANT_DISEASES_ROOT / "valid"
	test_target = DATASETS_ROOT / "test"

	_copy_split(train_source, train_target)
	_copy_split(valid_source, valid_target)
	_copy_split(test_source, test_target)

	print("Dataset prepared successfully:")
	print(f"- {train_target}")
	print(f"- {valid_target}")
	print(f"- {test_target}")


if __name__ == "__main__":
	main()