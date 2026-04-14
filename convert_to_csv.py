import os
import pandas as pd
from pathlib import Path

places = ["Bhatagaon DCR", "DCR AIIMS", "IGKV DCR", "SILTARA DCR"]
extensions = {".xls", ".xlsx", ".xlsb"}

def safe_dest(dest_dir, stem, ext=".csv"):
    p = dest_dir / f"{stem}{ext}"
    counter = 1
    while p.exists():
        p = dest_dir / f"{stem}_{counter}{ext}"
        counter += 1
    return p

for place in places:
    dest_dir = Path(place)
    # Convert files from subfolders only (skip files already at root to avoid re-converting copies)
    all_files = [f for f in Path(place).rglob("*") if f.suffix.lower() in extensions]
    # Deduplicate by name - prefer subfolder originals, skip root-level copies we made earlier
    seen_names = set()
    unique_files = []
    # First pass: collect subfolder files
    for f in all_files:
        if f.parent != dest_dir:
            unique_files.append(f)
            seen_names.add(f.name)
    # Second pass: add root files only if no subfolder version exists
    for f in all_files:
        if f.parent == dest_dir and f.name not in seen_names:
            unique_files.append(f)
            seen_names.add(f.name)
    all_files = unique_files
    
    copied, failed = 0, 0
    for f in all_files:
        try:
            # xlsb needs pyxlsb engine, xls needs xlrd, xlsx/others use openpyxl
            if f.suffix.lower() == ".xlsb":
                sheets = pd.read_excel(f, sheet_name=None, engine="pyxlsb")
            elif f.suffix.lower() == ".xls":
                sheets = pd.read_excel(f, sheet_name=None, engine="xlrd")
            else:
                sheets = pd.read_excel(f, sheet_name=None, engine="openpyxl")

            sheet_names = list(sheets.keys())
            if len(sheet_names) == 1:
                out = safe_dest(dest_dir, f.stem)
                sheets[sheet_names[0]].to_csv(out, index=False)
            else:
                for sname, df in sheets.items():
                    safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in str(sname))
                    out = safe_dest(dest_dir, f"{f.stem}__{safe_name}")
                    df.to_csv(out, index=False)
            copied += 1
        except Exception as e:
            print(f"  FAILED: {f.name} -> {e}")
            failed += 1

    print(f"{place}: {copied} files converted, {failed} failed")
