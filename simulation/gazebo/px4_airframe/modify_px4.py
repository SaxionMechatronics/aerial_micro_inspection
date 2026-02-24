#!/usr/bin/env python3

from pathlib import Path
import re

# ===================== USER SETTINGS =====================
FILE_PATH = "/opt/PX4-Autopilot/ROMFS/px4fmu_common/init.d-posix/airframes/CMakeLists.txt"
ITEM_TO_ADD = "2026_gz_x500_inspection"
MAKE_BACKUP = True
# =========================================================


def get_romfs_block(text: str) -> re.Match:
    m = re.search(r"px4_add_romfs_files\s*\(\s*(?P<body>[\s\S]*?)\n\)", text)
    if not m:
        raise ValueError("Could not find a px4_add_romfs_files( ... ) block.")
    return m


def item_exists_in_romfs_list(text: str, item: str) -> bool:
    m = get_romfs_block(text)
    body = m.group("body")
    return re.search(rf"(?m)^\s*{re.escape(item)}\s*$", body) is not None


def add_item_to_px4_add_romfs_files(text: str, item: str) -> str:
    m = get_romfs_block(text)
    body = m.group("body")

    if re.search(rf"(?m)^\s*{re.escape(item)}\s*$", body):
        return text

    lines = body.splitlines(keepends=True)

    insert_at = len(lines)
    while insert_at > 0:
        stripped = lines[insert_at - 1].strip()
        if stripped == "" or stripped.startswith("#"):
            insert_at -= 1
            continue
        break

    indent = "\t"
    for line in lines:
        s = line.strip()
        if s and not s.startswith("#"):
            indent = re.match(r"^\s*", line).group(0)
            break

    lines.insert(insert_at, f"{indent}{item}\n")

    new_body = "".join(lines)
    start, end = m.span("body")
    return text[:start] + new_body + text[end:]


def main() -> None:
    path = Path(FILE_PATH).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    original = path.read_text(encoding="utf-8")

    if item_exists_in_romfs_list(original, ITEM_TO_ADD):
        print(f"Already exists, no change: {ITEM_TO_ADD}")
        return

    updated = add_item_to_px4_add_romfs_files(original, ITEM_TO_ADD)

    if MAKE_BACKUP:
        backup = path.with_suffix(path.suffix + ".bak")
        backup.write_text(original, encoding="utf-8")
        print(f"Backup: {backup}")

    path.write_text(updated, encoding="utf-8")
    print(f"Inserted: {ITEM_TO_ADD}")


if __name__ == "__main__":
    main()
