"""Runnable example for read_pv_history(). Requires a filled-in config.yaml."""

import yaml

from epics_archiver_reader import read_pv_history


def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    df = read_pv_history(
        config,
        pv_name="CMA01-CRYO:TT001:TempMes",
        time_start="2026-01-01 00:00:00",
        time_end="2026-01-01 04:00:00",
    )
    print(f"{len(df)} samples")
    print(df.head())


if __name__ == "__main__":
    main()
