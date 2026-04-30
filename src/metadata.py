from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd


AUDACITY_NS = {"audacity": "http://audacity.sourceforge.net/xml/"}


def clean_roughness_table(excel_path: Path) -> pd.DataFrame:
    """Load roughness measurements and keep only numeric experiment rows."""
    roughness = pd.read_excel(excel_path, header=1)
    roughness = roughness.dropna(how="all").dropna(axis=1, how="all")

    numeric_cols = [
        "Feed (mm)",
        "Speed (rpm)",
        "DoC (mm)",
        "Ra",
        "Rz",
        "Rzmax",
        "RSm",
        "Rq",
        "Rp",
        "Rt",
        "R3z",
        "Pc",
        "Pt",
        "Rmr",
        "Rk",
        "Rpk",
        "Rvk",
        "Mr1",
        "Mr2",
        "Vo",
        "K",
    ]
    for col in numeric_cols:
        if col in roughness.columns:
            roughness[col] = pd.to_numeric(roughness[col], errors="coerce")

    roughness = roughness[roughness["Feed (mm)"].notna()].copy()
    roughness = roughness.rename(
        columns={
            "Feed (mm)": "feed_mm",
            "Speed (rpm)": "speed_rpm",
            "DoC (mm)": "depth_mm",
            "Ra": "roughness_ra",
        }
    )
    roughness["feed_mm"] = roughness["feed_mm"].astype(float)
    roughness["speed_rpm"] = roughness["speed_rpm"].astype(int)
    roughness["depth_mm"] = roughness["depth_mm"].astype(float)
    roughness["condition_id"] = roughness.apply(make_condition_id, axis=1)

    cols = ["condition_id", "feed_mm", "speed_rpm", "depth_mm", "roughness_ra"]
    other_cols = [col for col in roughness.columns if col not in cols]
    return roughness[cols + other_cols].reset_index(drop=True)


def make_condition_id(row: pd.Series) -> str:
    return f"f{float(row['feed_mm']):g}_s{int(row['speed_rpm'])}_d{float(row['depth_mm']):g}"


def parse_condition_from_path(path: Path) -> dict[str, float | int]:
    feed = speed = depth = None

    for part in path.parts:
        if feed is None:
            match = re.fullmatch(r"feed-(\d+(?:\.\d+)?)mm", part, flags=re.IGNORECASE)
            if match:
                feed = float(match.group(1))
        if speed is None:
            match = re.fullmatch(r"speed-(\d+(?:\.\d+)?)rpm", part, flags=re.IGNORECASE)
            if match:
                speed = int(float(match.group(1)))
        if depth is None:
            match = re.fullmatch(r"doc-(\d+(?:\.\d+)?)mm", part, flags=re.IGNORECASE)
            if match:
                depth = float(match.group(1))

    if feed is None or speed is None or depth is None:
        raise ValueError(f"Could not parse machining condition from path: {path}")

    return {"feed_mm": feed, "speed_rpm": speed, "depth_mm": depth}


def parse_run_number(aup_path: Path) -> int:
    match = re.match(r"(\d+)", aup_path.stem)
    if not match:
        raise ValueError(f"Could not parse run number from .aup file name: {aup_path}")
    return int(match.group(1))


def read_aup_summary(aup_path: Path) -> dict[str, float | int]:
    root = ET.parse(aup_path).getroot()
    sample_rate = float(root.attrib.get("rate", 0))
    tracks = root.findall(".//audacity:wavetrack", AUDACITY_NS)
    blocks = root.findall(".//audacity:simpleblockfile", AUDACITY_NS)
    sequences = root.findall(".//audacity:sequence", AUDACITY_NS)

    samples = 0
    if sequences:
        samples = max(int(sequence.attrib.get("numsamples", 0)) for sequence in sequences)

    duration = samples / sample_rate if sample_rate else 0.0
    return {
        "sample_rate_hz": sample_rate,
        "channels": len(tracks),
        "samples_per_channel": samples,
        "duration_sec": duration,
        "block_files_count": len(blocks),
    }


def build_audio_metadata(data_raw: Path, project_root: Path | None = None) -> pd.DataFrame:
    project_root = project_root or data_raw.parent.parent
    rows = []

    for aup_path in sorted(data_raw.rglob("*.aup")):
        condition = parse_condition_from_path(aup_path)
        row = {
            **condition,
            "run_id": parse_run_number(aup_path),
            "aup_path": aup_path.relative_to(project_root).as_posix(),
            "audio_data_dir": aup_path.with_name(f"{aup_path.stem}_data")
            .relative_to(project_root)
            .as_posix(),
        }
        row.update(read_aup_summary(aup_path))
        row["condition_id"] = make_condition_id(pd.Series(row))
        rows.append(row)

    metadata = pd.DataFrame(rows)
    sort_cols = ["feed_mm", "speed_rpm", "depth_mm", "run_id"]
    return metadata.sort_values(sort_cols).reset_index(drop=True)


def build_dataset_metadata(data_raw: Path, data_processed: Path, project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    excel_files = sorted(data_raw.rglob("*.xlsx"))
    if not excel_files:
        raise FileNotFoundError(f"No .xlsx roughness table found in {data_raw}")

    data_processed.mkdir(parents=True, exist_ok=True)
    roughness = clean_roughness_table(excel_files[0])
    audio_metadata = build_audio_metadata(data_raw, project_root=project_root)

    metadata = audio_metadata.merge(
        roughness,
        on=["condition_id", "feed_mm", "speed_rpm", "depth_mm"],
        how="left",
        validate="many_to_one",
    )

    if metadata["roughness_ra"].isna().any():
        missing = metadata.loc[
            metadata["roughness_ra"].isna(), ["condition_id", "feed_mm", "speed_rpm", "depth_mm"]
        ].drop_duplicates()
        raise ValueError(f"Some audio projects do not have roughness labels:\n{missing}")

    roughness.to_csv(data_processed / "roughness_table.csv", index=False, encoding="utf-8")
    metadata.to_csv(data_processed / "metadata.csv", index=False, encoding="utf-8")

    condition_summary = (
        metadata.groupby(["condition_id", "feed_mm", "speed_rpm", "depth_mm"], as_index=False)
        .agg(
            runs_count=("run_id", "count"),
            roughness_ra=("roughness_ra", "first"),
            duration_sec_mean=("duration_sec", "mean"),
            block_files_mean=("block_files_count", "mean"),
        )
        .sort_values(["feed_mm", "speed_rpm", "depth_mm"])
    )
    condition_summary.to_csv(data_processed / "condition_summary.csv", index=False, encoding="utf-8")

    return roughness, metadata
