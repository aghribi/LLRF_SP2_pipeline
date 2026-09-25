"""Read process-variable (PV) history from an EPICS "Channel Archiver" MySQL backend.

Adapted from Charly Lassalle's ``GetEPICSdata`` package
(https://gitlab.in2p3.fr/charly-lassalle-phd-thesis/programmes, MIT License,
Copyright (c) 2025 Charly Lassalle), written for his SPIRAL2 heat-load-observer PhD
thesis work at GANIL. This version keeps only the core archiver-read function,
translated to English, with parameterized SQL (the original built queries by string
concatenation) and a pandas DataFrame return instead of an astropy Table, to keep the
dependency footprint small. See LICENSE in this folder for the original license text.

Targets the standard EPICS Channel Archiver RDB schema: a ``channel`` table mapping PV
names to IDs, and a ``sample`` table of ``(channel_id, smpl_time, nanosecs, float_val)``
rows -- the same schema used by many EPICS-based facilities, not just GANIL's.
"""

from __future__ import annotations

import pandas as pd
import pymysql


def read_pv_history(config: dict, pv_name: str, time_start: str, time_end: str) -> pd.DataFrame:
    """Fetch the archived history of one PV between two timestamps.

    Parameters
    ----------
    config : dict
        Connection parameters: ``host``, ``port``, ``user``, ``passwd``, ``db``. See
        ``config.example.yaml`` -- copy it to ``config.yaml`` (gitignored) and fill in
        your own archiver's credentials.
    pv_name : str
        Exact PV name as recorded in the archiver's ``channel`` table.
    time_start, time_end : str
        Time bounds, e.g. ``"2026-01-01 00:00:00"``, in whatever format your MySQL
        server accepts for a DATETIME comparison. Half-open interval: samples with
        ``time_start <= smpl_time < time_end`` are returned.

    Returns
    -------
    pandas.DataFrame
        Columns ``time`` (nanosecond-resolution timestamp) and ``value`` (float),
        sorted by time. Empty if the PV has no samples in the requested window (e.g.
        the window predates the archiver's retention, or the PV name doesn't exist).
    """
    conn = pymysql.connect(
        host=config["host"],
        port=config["port"],
        user=config["user"],
        passwd=config["passwd"],
        db=config["db"],
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT s.smpl_time, s.nanosecs, s.float_val
                FROM sample s
                JOIN channel c ON c.channel_id = s.channel_id
                WHERE c.name = %s
                  AND s.smpl_time >= %s
                  AND s.smpl_time < %s
                ORDER BY s.smpl_time, s.nanosecs
                """,
                (pv_name, time_start, time_end),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return pd.DataFrame(columns=["time", "value"])

    df = pd.DataFrame(rows, columns=["smpl_time", "nanosecs", "value"])
    df["time"] = pd.to_datetime(df["smpl_time"]) + pd.to_timedelta(df["nanosecs"], unit="ns")
    return df[["time", "value"]]
