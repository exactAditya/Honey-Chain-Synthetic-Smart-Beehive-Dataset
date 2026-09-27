"""Create Honey Chain's synthetic smart-beehive monitoring dataset.

The source temperature and weight observations are used as empirical baselines;
class-conditioned offsets and sensor noise make the measurements reflect the
requested hive conditions. Re-run with the same seed for identical output.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE_PATH = ROOT / "honey_bee_dataset_filtered.csv"
OUTPUT_PATH = ROOT / "honey_bee_dataset_final.csv"
SEED = 42

CLASS_COUNTS = {
    "Normal": 1500,
    "Weak_Colony": 375,
    "Disease_Risk": 375,
    "Swarming_Risk": 375,
    "Queenless": 375,
}


def main() -> None:
    rng = np.random.default_rng(SEED)
    source = pd.read_csv(SOURCE_PATH)
    expected_columns = ["Temperature_C", "Hive_Weight_kg"]
    if list(source.columns) != expected_columns:
        raise ValueError(f"Expected source columns {expected_columns}, got {list(source.columns)}")

    row_count = len(source)
    if row_count != sum(CLASS_COUNTS.values()):
        raise ValueError(
            f"Expected {sum(CLASS_COUNTS.values())} source rows, found {row_count}"
        )

    # Shuffle source observations without replacement so each input row
    # contributes one empirical temperature/weight baseline.
    baseline = source.iloc[rng.permutation(row_count)].reset_index(drop=True)
    base_temperature = baseline["Temperature_C"].to_numpy(dtype=float)
    base_weight = baseline["Hive_Weight_kg"].to_numpy(dtype=float)
    temp_center = float(source["Temperature_C"].median())
    weight_center = float(source["Hive_Weight_kg"].median())

    labels = np.concatenate(
        [np.repeat(status, count) for status, count in CLASS_COUNTS.items()]
    )
    rng.shuffle(labels)

    temperature = np.empty(row_count)
    weight = np.empty(row_count)
    sound = np.empty(row_count)

    for status, count in CLASS_COUNTS.items():
        indices = np.flatnonzero(labels == status)
        rank = np.linspace(0.0, 1.0, count)
        rng.shuffle(rank)

        if status == "Normal":
            temperature[indices] = (
                33.55
                + 0.22 * (base_temperature[indices] - temp_center)
                + rng.normal(0, 0.58, count)
            )
            weight[indices] = (
                30.0
                + 0.55 * (base_weight[indices] - weight_center)
                + 0.7 * rank
                + rng.normal(0, 0.45, count)
            )
            sound[indices] = rng.normal(50.0, 2.2, count)
            temperature[indices] = np.clip(temperature[indices], 31.8, 35.7)
            sound[indices] = np.clip(sound[indices], 43.0, 57.0)

        elif status == "Weak_Colony":
            temperature[indices] = (
                31.25
                + 0.16 * (base_temperature[indices] - temp_center)
                + rng.normal(0, 0.58, count)
            )
            weight[indices] = (
                27.5
                + 0.45 * (base_weight[indices] - weight_center)
                - 0.5 * rank
                + rng.normal(0, 0.55, count)
            )
            sound[indices] = rng.normal(38.5, 2.2, count)
            temperature[indices] = np.clip(temperature[indices], 29.7, 33.0)
            sound[indices] = np.clip(sound[indices], 33.0, 45.0)

        elif status == "Disease_Risk":
            temperature[indices] = (
                34.0
                + 0.42 * (base_temperature[indices] - temp_center)
                + rng.normal(0, 1.05, count)
            )
            weight[indices] = (
                27.0
                + 0.68 * (base_weight[indices] - weight_center)
                - 0.8 * rank
                + rng.normal(0, 0.65, count)
            )
            sound[indices] = rng.normal(49.0, 4.5, count)
            temperature[indices] = np.clip(temperature[indices], 29.0, 40.5)
            sound[indices] = np.clip(sound[indices], 38.0, 61.0)

        elif status == "Swarming_Risk":
            temperature[indices] = (
                33.9
                + 0.25 * (base_temperature[indices] - temp_center)
                + rng.normal(0, 0.6, count)
            )
            # Lower the empirical weight baseline to represent a sudden drop.
            weight[indices] = (
                25.0
                + 0.65 * (base_weight[indices] - weight_center)
                + rng.normal(0, 0.55, count)
            )
            sound[indices] = rng.normal(65.0, 2.4, count)
            temperature[indices] = np.clip(temperature[indices], 31.8, 35.8)
            sound[indices] = np.clip(sound[indices], 58.0, 72.0)

        else:  # Queenless
            temperature[indices] = (
                33.7
                + 0.55 * (base_temperature[indices] - temp_center)
                + rng.normal(0, 0.95, count)
            )
            weight[indices] = (
                29.5
                + 0.30 * (base_weight[indices] - weight_center)
                + rng.normal(0, 0.5, count)
            )
            sound[indices] = rng.normal(59.5, 3.2, count)
            temperature[indices] = np.clip(temperature[indices], 28.5, 39.0)
            sound[indices] = np.clip(sound[indices], 51.0, 68.0)

    result = pd.DataFrame(
        {
            "Temperature_C": np.round(temperature, 2),
            "Hive_Weight_kg": np.round(weight, 2),
            "Sound_Level_dB": np.round(sound, 2),
            "Hive_Status": labels,
        }
    )
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(result)} rows to {OUTPUT_PATH.name}")
    print(result["Hive_Status"].value_counts().reindex(CLASS_COUNTS).to_string())


if __name__ == "__main__":
    main()