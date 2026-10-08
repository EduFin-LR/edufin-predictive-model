"""
Exporta interacciones reales EDUFIN desde PostgreSQL al formato mínimo que
consume dkt_forget_final.py.

Salida CSV:
    user_id,skill_id,correct,timestamp

Interacciones válidas para DKT-Forget:
    PRE_TEST, QUIZ, REINFORCEMENT, FINAL

Uso local (PowerShell):
    $env:DATABASE_URL="postgresql://usuario:password@host:5432/db"
    python export_real_interactions.py

Opcional:
    python export_real_interactions.py --output datasets/pilot_real_v1.csv
    python export_real_interactions.py --after 2026-10-01T00:00:00Z
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import psycopg


VALID_INTERACTION_TYPES = (
    "PRE_TEST",
    "QUIZ",
    "REINFORCEMENT",
    "FINAL",
)

DEFAULT_OUTPUT = Path("datasets/pilot_real_v1.csv")


SQL_BASE = """
SELECT
    si.id::text AS interaction_id,
    si.user_id::text AS user_id,
    si.dkt_skill_id AS skill_id,
    si.is_correct AS correct,
    si.interacted_at AS timestamp,
    si.interaction_type AS interaction_type
FROM student_interactions si
WHERE si.interaction_type = ANY(%s)
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exporta interacciones reales EDUFIN para DKT-Forget."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"CSV de salida (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--after",
        type=str,
        default=None,
        help=(
            "Exporta solo interacciones posteriores a este instante ISO-8601, "
            "por ejemplo 2026-10-01T00:00:00Z."
        ),
    )
    parser.add_argument(
        "--before",
        type=str,
        default=None,
        help=(
            "Exporta solo interacciones anteriores o iguales a este instante "
            "ISO-8601."
        ),
    )
    return parser.parse_args()


def _parse_optional_timestamp(value: Optional[str], name: str) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    try:
        ts = pd.to_datetime(value, utc=True, errors="raise")
    except Exception as exc:
        raise ValueError(f"{name} no es un timestamp ISO-8601 válido: {value}") from exc
    return ts


def fetch_interactions(
    database_url: str,
    *,
    after: Optional[pd.Timestamp] = None,
    before: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    query = SQL_BASE
    params: list[object] = [list(VALID_INTERACTION_TYPES)]

    if after is not None:
        query += " AND si.interacted_at > %s\n"
        params.append(after.to_pydatetime())

    if before is not None:
        query += " AND si.interacted_at <= %s\n"
        params.append(before.to_pydatetime())

    # Orden determinista. interaction_id rompe empates si dos eventos tienen
    # exactamente el mismo timestamp.
    query += " ORDER BY si.user_id, si.interacted_at, si.id\n"

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
            columns = [desc.name for desc in cur.description]

    return pd.DataFrame(rows, columns=columns)


def validate_interactions(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("No se encontraron interacciones DKT válidas en el rango solicitado.")

    required = {
        "interaction_id",
        "user_id",
        "skill_id",
        "correct",
        "timestamp",
        "interaction_type",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en la consulta: {sorted(missing)}")

    if df[list(required)].isnull().any().any():
        null_counts = df[list(required)].isnull().sum()
        null_counts = null_counts[null_counts > 0].to_dict()
        raise ValueError(f"Se encontraron valores nulos: {null_counts}")

    if df["interaction_id"].duplicated().any():
        duplicates = int(df["interaction_id"].duplicated().sum())
        raise ValueError(f"Se encontraron {duplicates} interaction_id duplicados.")

    df = df.copy()
    df["skill_id"] = pd.to_numeric(df["skill_id"], errors="raise").astype(int)
    invalid_skills = sorted(df.loc[~df["skill_id"].between(1, 30), "skill_id"].unique().tolist())
    if invalid_skills:
        raise ValueError(f"skill_id fuera del contrato 1..30: {invalid_skills}")

    df["correct"] = pd.to_numeric(df["correct"], errors="raise").astype(int)
    invalid_correct = sorted(df.loc[~df["correct"].isin([0, 1]), "correct"].unique().tolist())
    if invalid_correct:
        raise ValueError(f"correct debe contener solo 0/1. Valores inválidos: {invalid_correct}")

    invalid_types = sorted(set(df["interaction_type"]) - set(VALID_INTERACTION_TYPES))
    if invalid_types:
        raise ValueError(f"interaction_type no válido para DKT: {invalid_types}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="raise")

    # Conservamos todas las interacciones reales. No eliminamos filas por tener
    # exactamente el mismo contenido, ya que podrían representar respuestas
    # distintas. La identidad real del evento está dada por interaction_id.
    df = df.sort_values(
        ["user_id", "timestamp", "interaction_id"],
        kind="stable",
    ).reset_index(drop=True)

    return df


def build_training_csv(df: pd.DataFrame) -> pd.DataFrame:
    result = df[["user_id", "skill_id", "correct", "timestamp"]].copy()
    result["timestamp"] = result["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return result


def print_summary(df: pd.DataFrame, output: Path) -> None:
    counts = df["interaction_type"].value_counts().to_dict()
    user_counts = df.groupby("user_id").size()

    print("\nExportación EDUFIN completada")
    print("-" * 40)
    print(f"Usuarios encontrados:       {df['user_id'].nunique()}")
    print(f"Interacciones DKT:          {len(df)}")
    print(f"Usuarios con >= 8 eventos:  {(user_counts >= 8).sum()}")

    for interaction_type in VALID_INTERACTION_TYPES:
        print(f"{interaction_type:<24}{counts.get(interaction_type, 0)}")

    print(f"Skills distintas:           {df['skill_id'].nunique()}/30")
    print(f"Primer timestamp:           {df['timestamp'].min().isoformat()}")
    print(f"Último timestamp:           {df['timestamp'].max().isoformat()}")
    print(f"Archivo:                    {output}")


def main() -> None:
    args = parse_args()

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "Falta la variable de entorno DATABASE_URL. "
            "No guardes credenciales de PostgreSQL dentro del código."
        )

    after = _parse_optional_timestamp(args.after, "--after")
    before = _parse_optional_timestamp(args.before, "--before")

    if after is not None and before is not None and after >= before:
        raise ValueError("--after debe ser anterior a --before.")

    raw = fetch_interactions(
        database_url,
        after=after,
        before=before,
    )
    validated = validate_interactions(raw)
    training_df = build_training_csv(validated)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    training_df.to_csv(args.output, index=False)

    print_summary(validated, args.output)


if __name__ == "__main__":
    main()
