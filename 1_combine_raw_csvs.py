import os
import glob
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────
STATION_FOLDERS = {
    "Bhatagaon DCR": "BHATAGAON",
    "DCR AIIMS":     "AIIMS",
    "IGKV DCR":      "IGKV",
    "SILTARA DCR":   "SILTARA",
}

OUTPUT_DIR  = "artifacts"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "data_15min.parquet")

IGNORE_KEYWORDS = ["MUX", "POWER OFF", "CALIBRATION", "MONTHLY"]


def find_header_row(filepath):
    """Return the 0-based line index of the row containing 'Date & Time'."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
        for i, line in enumerate(fh):
            if "Date & Time" in line:
                return i
    return 0


def process_file(filepath, station):
    filename = os.path.basename(filepath)

    if any(kw in filename.upper() for kw in IGNORE_KEYWORDS):
        print("  [SKIP]  " + filename)
        return None

    try:
        skip_idx = find_header_row(filepath)
        df = pd.read_csv(
            filepath,
            skiprows=skip_idx,
            encoding="utf-8",
            encoding_errors="ignore",
            on_bad_lines="skip",
        )

        if df.empty:
            print("  [EMPTY] " + filename)
            return None

        # Drop phantom Unnamed columns (trailing comma bug)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

        # Rename timestamp column
        if "Date & Time" in df.columns:
            df = df.rename(columns={"Date & Time": "timestamp"})

        if "timestamp" not in df.columns:
            print("  [WARN]  No 'Date & Time' column in " + filename + ", skipping.")
            return None

        # Drop units row (first data row after header)
        df = df.iloc[1:].reset_index(drop=True)

        # Drop summary rows
        mask = df["timestamp"].astype(str).str.contains(
            r"\bMin\b|\bMax\b|\bTotal\b", case=False, na=False
        )
        df = df[~mask]

        # Drop fully empty rows
        df = df.dropna(how="all")
        df = df[df["timestamp"].astype(str).str.strip() != ""]

        # Add metadata
        df["station"]      = station
        df["source_file"]  = filename
        df["source_sheet"] = "CSV"

        # Cast to str per-file to avoid OOM on final concat
        df = df.astype(str)

        print("  [OK]    " + filename + "  ->  " + station + "  (" + str(len(df)) + " rows)")
        return df

    except Exception as exc:
        print("  [ERROR] " + filename + ": " + str(exc))
        return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_frames = []

    for folder, station_label in STATION_FOLDERS.items():
        pattern = os.path.join(folder, "*.csv")
        csv_files = sorted(glob.glob(pattern))

        print("\n" + "=" * 60)
        print("Station: " + station_label + "  |  Folder: " + folder + "  |  Files: " + str(len(csv_files)))
        print("=" * 60)

        for fp in csv_files:
            df = process_file(fp, station_label)
            if df is not None:
                all_frames.append(df)

    if not all_frames:
        print("\n[ABORT] No data was collected.")
        return

    print("\nMerging " + str(len(all_frames)) + " dataframes...")
    combined = pd.concat(all_frames, ignore_index=True)

    print("Saving -> " + OUTPUT_FILE + "  (shape: " + str(combined.shape) + ")")
    combined.to_parquet(OUTPUT_FILE, index=False)
    print("\nDone!  Rows: " + str(combined.shape[0]) + "  |  Columns: " + str(combined.shape[1]))


if __name__ == "__main__":
    main()
