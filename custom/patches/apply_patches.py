"""
Apply patches to installed third-party packages in .venv.

These are compatibility fixes for package version mismatches that can't be
resolved through normal dependency pinning. Re-run this script any time
the venv is recreated.
"""

import pathlib
import sys


def patch_mmdet_mmcv_version_check(venv_root: pathlib.Path) -> None:
    """mmdet 3.3.0 hardcodes mmcv_maximum_version = '2.2.0' (exclusive),
    which rejects mmcv==2.2.0. Bump the cap to '2.3.0' to allow it."""
    target = venv_root / "lib" / "python3.12" / "site-packages" / "mmdet" / "__init__.py"
    if not target.exists():
        print(f"SKIP: {target} not found")
        return

    text = target.read_text()
    old = "mmcv_maximum_version = '2.2.0'"
    new = "mmcv_maximum_version = '2.3.0'"

    if new in text:
        print(f"OK (already patched): {target}")
        return
    if old not in text:
        print(f"WARNING: expected string not found in {target}, skipping")
        return

    target.write_text(text.replace(old, new))
    print(f"Patched: {target}")


def main() -> None:
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    venv_root = repo_root / ".venv"

    if not venv_root.exists():
        print(f"ERROR: venv not found at {venv_root}", file=sys.stderr)
        sys.exit(1)

    patch_mmdet_mmcv_version_check(venv_root)
    print("Done.")


if __name__ == "__main__":
    main()
