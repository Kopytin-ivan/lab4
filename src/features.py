from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import welch


AUDACITY_NS = {"audacity": "http://audacity.sourceforge.net/xml/"}


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _parse_aup_tracks(aup_path: Path) -> tuple[float, list[list[dict[str, str]]]]:
    root = ET.parse(aup_path).getroot()
    sample_rate = float(root.attrib.get("rate", 44100.0))
    tracks = []

    for track in root.findall(".//audacity:wavetrack", AUDACITY_NS):
        blocks = track.findall(".//audacity:simpleblockfile", AUDACITY_NS)
        tracks.append([block.attrib for block in blocks])

    return sample_rate, tracks


def _block_stats(tracks: list[list[dict[str, str]]]) -> dict[str, float]:
    blocks = [block for track in tracks for block in track]
    mins = np.array([float(block.get("min", 0.0)) for block in blocks], dtype=float)
    maxs = np.array([float(block.get("max", 0.0)) for block in blocks], dtype=float)
    rms = np.array([float(block.get("rms", 0.0)) for block in blocks], dtype=float)
    lens = np.array([int(block.get("len", 0)) for block in blocks], dtype=float)

    peak = float(np.maximum(np.abs(mins), np.abs(maxs)).max()) if len(blocks) else 0.0
    rms_mean = float(rms.mean()) if len(blocks) else 0.0
    clipping_share = float(((mins <= -0.999) | (maxs >= 0.999)).mean()) if len(blocks) else 0.0

    return {
        "xml_block_count": float(len(blocks)),
        "xml_block_len_mean": float(lens.mean()) if len(blocks) else 0.0,
        "xml_min_mean": float(mins.mean()) if len(blocks) else 0.0,
        "xml_min_abs_max": float(abs(mins.min())) if len(blocks) else 0.0,
        "xml_max_mean": float(maxs.mean()) if len(blocks) else 0.0,
        "xml_max_abs_max": float(abs(maxs.max())) if len(blocks) else 0.0,
        "xml_rms_mean": rms_mean,
        "xml_rms_std": float(rms.std(ddof=0)) if len(blocks) else 0.0,
        "xml_rms_min": float(rms.min()) if len(blocks) else 0.0,
        "xml_rms_max": float(rms.max()) if len(blocks) else 0.0,
        "xml_peak_to_rms": _safe_divide(peak, rms_mean),
        "xml_dynamic_range": float(maxs.max() - mins.min()) if len(blocks) else 0.0,
        "xml_clipping_share": clipping_share,
    }


def _find_block_file(audio_data_dir: Path, filename: str, cache: dict[str, Path]) -> Path | None:
    if not cache:
        cache.update({path.name: path for path in audio_data_dir.rglob("*.au")})
    return cache.get(filename)


def _selected_block_indices(block_count: int, max_blocks: int) -> list[int]:
    if block_count <= 0:
        return []
    if block_count <= max_blocks:
        return list(range(block_count))
    return sorted(set(np.linspace(0, block_count - 1, max_blocks).round().astype(int).tolist()))


def _read_representative_signal(
    tracks: list[list[dict[str, str]]],
    audio_data_dir: Path,
    max_blocks_per_track: int = 6,
) -> np.ndarray:
    if not tracks:
        return np.array([], dtype=np.float32)

    min_track_blocks = min(len(track) for track in tracks if track)
    indices = _selected_block_indices(min_track_blocks, max_blocks_per_track)
    file_cache: dict[str, Path] = {}
    segments = []

    for index in indices:
        channel_segments = []
        for track in tracks:
            if index >= len(track):
                continue
            block_path = _find_block_file(audio_data_dir, track[index]["filename"], file_cache)
            if block_path is None:
                continue
            data, _ = sf.read(str(block_path), dtype="float32", always_2d=False)
            if data.ndim > 1:
                data = data.mean(axis=1)
            channel_segments.append(data.astype(np.float32, copy=False))

        if channel_segments:
            min_len = min(len(segment) for segment in channel_segments)
            stacked = np.vstack([segment[:min_len] for segment in channel_segments])
            segments.append(stacked.mean(axis=0))

    if not segments:
        return np.array([], dtype=np.float32)

    return np.concatenate(segments).astype(np.float32, copy=False)


def _time_features(signal: np.ndarray) -> dict[str, float]:
    if signal.size == 0:
        return {
            "sig_mean": 0.0,
            "sig_std": 0.0,
            "sig_rms": 0.0,
            "sig_abs_mean": 0.0,
            "sig_peak_abs": 0.0,
            "sig_peak_to_peak": 0.0,
            "sig_crest_factor": 0.0,
            "sig_zero_crossing_rate": 0.0,
        }

    rms = float(np.sqrt(np.mean(signal**2)))
    peak_abs = float(np.max(np.abs(signal)))
    signs = np.signbit(signal)
    zero_crossing_rate = float(np.mean(signs[1:] != signs[:-1])) if signal.size > 1 else 0.0

    return {
        "sig_mean": float(np.mean(signal)),
        "sig_std": float(np.std(signal)),
        "sig_rms": rms,
        "sig_abs_mean": float(np.mean(np.abs(signal))),
        "sig_peak_abs": peak_abs,
        "sig_peak_to_peak": float(np.max(signal) - np.min(signal)),
        "sig_crest_factor": _safe_divide(peak_abs, rms),
        "sig_zero_crossing_rate": zero_crossing_rate,
    }


def _band_power(freqs: np.ndarray, power: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs < high)
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(power[mask], freqs[mask]))


def _spectral_features(signal: np.ndarray, sample_rate: float) -> dict[str, float]:
    if signal.size < 2048:
        return {
            "spec_centroid_hz": 0.0,
            "spec_bandwidth_hz": 0.0,
            "spec_rolloff85_hz": 0.0,
            "spec_flatness": 0.0,
            "spec_dominant_hz": 0.0,
            "spec_power_total": 0.0,
            "spec_power_0_500": 0.0,
            "spec_power_500_2000": 0.0,
            "spec_power_2000_8000": 0.0,
            "spec_power_8000_plus": 0.0,
            "spec_high_to_low_ratio": 0.0,
        }

    nperseg = min(8192, signal.size)
    freqs, power = welch(signal, fs=sample_rate, nperseg=nperseg)
    power = np.maximum(power, 1e-20)
    total_power = float(np.trapezoid(power, freqs))
    power_sum = float(power.sum())

    centroid = _safe_divide(float(np.sum(freqs * power)), power_sum)
    bandwidth = math.sqrt(_safe_divide(float(np.sum(((freqs - centroid) ** 2) * power)), power_sum))
    cumulative = np.cumsum(power)
    rolloff_idx = int(np.searchsorted(cumulative, 0.85 * cumulative[-1]))
    rolloff_idx = min(rolloff_idx, len(freqs) - 1)
    dominant_idx = int(np.argmax(power))
    flatness = float(np.exp(np.mean(np.log(power))) / np.mean(power))

    low_power = _band_power(freqs, power, 0, 500)
    low_mid_power = _band_power(freqs, power, 500, 2000)
    mid_high_power = _band_power(freqs, power, 2000, 8000)
    high_power = _band_power(freqs, power, 8000, sample_rate / 2 + 1)

    return {
        "spec_centroid_hz": float(centroid),
        "spec_bandwidth_hz": float(bandwidth),
        "spec_rolloff85_hz": float(freqs[rolloff_idx]),
        "spec_flatness": flatness,
        "spec_dominant_hz": float(freqs[dominant_idx]),
        "spec_power_total": total_power,
        "spec_power_0_500": low_power,
        "spec_power_500_2000": low_mid_power,
        "spec_power_2000_8000": mid_high_power,
        "spec_power_8000_plus": high_power,
        "spec_high_to_low_ratio": _safe_divide(mid_high_power + high_power, low_power + low_mid_power),
    }


def extract_features_for_row(row: pd.Series, project_root: Path, max_blocks_per_track: int = 6) -> dict[str, float]:
    aup_path = project_root / str(row["aup_path"])
    audio_data_dir = project_root / str(row["audio_data_dir"])

    sample_rate, tracks = _parse_aup_tracks(aup_path)
    signal = _read_representative_signal(
        tracks=tracks,
        audio_data_dir=audio_data_dir,
        max_blocks_per_track=max_blocks_per_track,
    )

    features = {
        "condition_id": row["condition_id"],
        "feed_mm": float(row["feed_mm"]),
        "speed_rpm": int(row["speed_rpm"]),
        "depth_mm": float(row["depth_mm"]),
        "run_id": int(row["run_id"]),
        "roughness_ra": float(row["roughness_ra"]),
        "sample_rate_hz": float(sample_rate),
        "channels": float(len(tracks)),
        "duration_sec": float(row["duration_sec"]),
        "selected_samples": float(signal.size),
    }
    features.update(_block_stats(tracks))
    features.update(_time_features(signal))
    features.update(_spectral_features(signal, sample_rate))
    return features


def build_feature_table(
    metadata: pd.DataFrame,
    project_root: Path,
    output_path: Path | None = None,
    max_blocks_per_track: int = 6,
    progress_every: int = 25,
) -> pd.DataFrame:
    rows = []
    total = len(metadata)

    for index, (_, row) in enumerate(metadata.iterrows(), start=1):
        rows.append(
            extract_features_for_row(
                row=row,
                project_root=project_root,
                max_blocks_per_track=max_blocks_per_track,
            )
        )
        if progress_every and (index % progress_every == 0 or index == total):
            print(f"Processed {index}/{total} audio projects")

    features = pd.DataFrame(rows)
    sort_cols = ["feed_mm", "speed_rpm", "depth_mm", "run_id"]
    features = features.sort_values(sort_cols).reset_index(drop=True)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(output_path, index=False, encoding="utf-8")

    return features
