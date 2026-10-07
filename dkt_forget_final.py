"""DKT-Forget final base for EDUFIN.

This module implements a DKT-Forget style model aligned with the temporal
feature preprocessing used by pyKT/Nagatani-style DKT-Forget:

- rgap: log2-binned time (minutes) since previous interaction with same skill
- sgap: log2-binned time (minutes) since previous interaction overall
- pcount: log2-binned number of previous interactions with same skill

The module is intended for EDUFIN production training/inference with the
financial skill taxonomy (for example 30 skills). The ASSISTments 2012
benchmark should remain in the separate pyKT experiment notebook.

Expected CSV columns (aliases supported):
    user_id, skill_id, correct, timestamp

Important:
- Timestamps are mandatory. They are never synthesized.
- DKT output after interaction t estimates probability of correctness on the
  next interaction for every skill.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, random_split

EDUFIN_NUM_SKILLS = 30

EDUFIN_SKILL2IDX = {
    skill_id: skill_id - 1
    for skill_id in range(1, EDUFIN_NUM_SKILLS + 1)
}

EDUFIN_IDX2SKILL = {
    idx: skill_id
    for skill_id, idx in EDUFIN_SKILL2IDX.items()
}

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Temporal features: pyKT-compatible DKT-Forget transformation
# -----------------------------------------------------------------------------


def log2_bin(value: float) -> int:
    """Return round(log2(value + 1)), clamped at 0.

    pyKT's DKT-Forget preprocessing uses this transformation for elapsed
    minutes and previous practice counts.
    """
    value = max(0.0, float(value))
    return int(round(math.log(value + 1.0, 2)))


def compute_forgetting_features(
    skill_ids: Sequence[int],
    timestamps: Sequence[Union[pd.Timestamp, datetime, str, int, float]],
) -> Tuple[List[int], List[int], List[int]]:
    """Compute rgap, sgap and pcount for an ordered interaction sequence.

    Timestamps may be datetime-like values or numeric epoch values. Numeric
    timestamps are interpreted as milliseconds when their magnitude looks like
    epoch milliseconds; otherwise as seconds.

    Returns:
        (rgaps, sgaps, pcounts), all already discretized with log2_bin.
    """
    if len(skill_ids) != len(timestamps):
        raise ValueError("skill_ids and timestamps must have the same length")

    parsed = [_to_timestamp_ms(t) for t in timestamps]

    rgaps: List[int] = []
    sgaps: List[int] = []
    pcounts: List[int] = []

    last_skill_ms: Dict[int, int] = {}
    skill_counts: Dict[int, int] = {}
    previous_ms: Optional[int] = None

    for skill, current_ms in zip(skill_ids, parsed):
        skill = int(skill)

        if skill in last_skill_ms:
            repeated_minutes = max(0.0, (current_ms - last_skill_ms[skill]) / 1000.0 / 60.0)
            rgap = log2_bin(repeated_minutes) + 1
        else:
            rgap = 0
        last_skill_ms[skill] = current_ms

        if previous_ms is None:
            sgap = 0
        else:
            sequence_minutes = max(0.0, (current_ms - previous_ms) / 1000.0 / 60.0)
            sgap = log2_bin(sequence_minutes) + 1
        previous_ms = current_ms

        previous_count = skill_counts.get(skill, 0)
        pcount = log2_bin(previous_count)
        skill_counts[skill] = previous_count + 1

        rgaps.append(rgap)
        sgaps.append(sgap)
        pcounts.append(pcount)

    return rgaps, sgaps, pcounts


def _to_timestamp_ms(value: Union[pd.Timestamp, datetime, str, int, float]) -> int:
    if isinstance(value, pd.Timestamp):
        ts = value
    elif isinstance(value, datetime):
        ts = pd.Timestamp(value)
    elif isinstance(value, str):
        ts = pd.to_datetime(value, utc=True)
    elif isinstance(value, (int, float, np.integer, np.floating)):
        v = float(value)
        # 1e11 is safely above current epoch seconds and below epoch ms.
        return int(v if abs(v) >= 1e11 else v * 1000.0)
    else:
        raise TypeError(f"Unsupported timestamp type: {type(value)!r}")

    if pd.isna(ts):
        raise ValueError("Timestamp cannot be null")
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return int(ts.timestamp() * 1000.0)


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


COLUMN_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "user_id": ("user_id", "user", "uid", "student_id"),
    "skill_id": ("skill_id", "sequence_id", "concept", "skill", "concept_id"),
    "correct": ("correct", "response", "is_correct", "result"),
    "timestamp": ("timestamp", "start_time", "time", "created_at", "answered_at"),
}


@dataclass(frozen=True)
class DKTForgetSample:
    skills: torch.Tensor
    corrects: torch.Tensor
    rgaps: torch.Tensor
    sgaps: torch.Tensor
    pcounts: torch.Tensor


class DKTForgetDataset(Dataset):
    """One sequence per student, ordered by timestamp."""

    def __init__(
        self,
        csv_file: Union[str, os.PathLike],
        *,
        min_interactions: int = 3,
        skill2idx: Optional[Mapping[Union[str, int], int]] = None,
        expected_num_skills: Optional[int] = None,
    ) -> None:
        self.csv_file = str(csv_file)
        self.df = pd.read_csv(self.csv_file)
        self.df = self._normalize_columns(self.df)

        missing = [c for c in COLUMN_ALIASES if c not in self.df.columns]
        if missing:
            raise ValueError(
                "Dataset missing required columns: " + ", ".join(missing) +
                ". DKT-Forget requires real timestamps; timestamps will not be synthesized."
            )

        self.df = self.df[["user_id", "skill_id", "correct", "timestamp"]].copy()
        self.df = self.df.dropna(
            subset=["user_id", "skill_id", "correct", "timestamp"]
        )

        # Normalize EDUFIN skill IDs and enforce the fixed 1..30 contract.
        self.df["skill_id"] = pd.to_numeric(
            self.df["skill_id"],
            errors="raise",
        ).astype(int)

        invalid_skill_ids = sorted(
            set(self.df["skill_id"]) - set(EDUFIN_SKILL2IDX.keys())
        )

        if invalid_skill_ids:
            raise ValueError(
                "Se encontraron skill_id fuera del rango 1..30: "
                f"{invalid_skill_ids}"
            )

        self.df["timestamp"] = pd.to_datetime(
            self.df["timestamp"],
            format="ISO8601",
            utc=True,
            errors="raise",
        )

        self.df["correct"] = self.df["correct"].astype(int)
        if not self.df["correct"].isin([0, 1]).all():
            raise ValueError("Column 'correct' must contain only 0/1 values")

        if skill2idx is None:
            unique_skills = sorted(self.df["skill_id"].unique().tolist(), key=lambda x: str(x))
            self.skill2idx = {s: i for i, s in enumerate(unique_skills)}
        else:
            self.skill2idx = dict(skill2idx)

        unknown = sorted(
            {s for s in self.df["skill_id"].unique().tolist() if s not in self.skill2idx},
            key=lambda x: str(x),
        )
        if unknown:
            raise ValueError(f"Dataset contains skills absent from skill2idx: {unknown[:10]}")

        self.idx2skill = {v: k for k, v in self.skill2idx.items()}
        self.num_skills = len(self.skill2idx)

        if expected_num_skills is not None and self.num_skills != expected_num_skills:
            raise ValueError(
                f"Expected {expected_num_skills} skills but mapping contains {self.num_skills}."
            )

        self.df["skill_idx"] = self.df["skill_id"].map(self.skill2idx).astype(int)
        self.samples: List[DKTForgetSample] = []

        for _, group in self.df.groupby("user_id", sort=False):
            group = group.sort_values("timestamp", kind="stable")
            if len(group) < min_interactions:
                continue

            skills = group["skill_idx"].astype(int).tolist()
            corrects = group["correct"].astype(int).tolist()
            timestamps = group["timestamp"].tolist()
            rgaps, sgaps, pcounts = compute_forgetting_features(skills, timestamps)

            self.samples.append(
                DKTForgetSample(
                    skills=torch.tensor(skills, dtype=torch.long),
                    corrects=torch.tensor(corrects, dtype=torch.long),
                    rgaps=torch.tensor(rgaps, dtype=torch.long),
                    sgaps=torch.tensor(sgaps, dtype=torch.long),
                    pcounts=torch.tensor(pcounts, dtype=torch.long),
                )
            )

        if not self.samples:
            raise ValueError("No student sequences satisfy min_interactions")

        self.num_rgap = max(int(s.rgaps.max()) for s in self.samples) + 1
        self.num_sgap = max(int(s.sgaps.max()) for s in self.samples) + 1
        self.num_pcount = max(int(s.pcounts.max()) for s in self.samples) + 1

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        lower_to_original = {str(c).lower(): c for c in df.columns}
        rename = {}
        for canonical, aliases in COLUMN_ALIASES.items():
            for alias in aliases:
                if alias in lower_to_original:
                    rename[lower_to_original[alias]] = canonical
                    break
        return df.rename(columns=rename)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> DKTForgetSample:
        return self.samples[idx]


def collate_batch(samples: Sequence[DKTForgetSample]) -> Dict[str, torch.Tensor]:
    """Pad variable-length student sequences and return a valid-position mask."""
    batch_size = len(samples)
    max_len = max(len(s.skills) for s in samples)

    skills = torch.zeros((batch_size, max_len), dtype=torch.long)
    corrects = torch.zeros((batch_size, max_len), dtype=torch.long)
    rgaps = torch.zeros((batch_size, max_len), dtype=torch.long)
    sgaps = torch.zeros((batch_size, max_len), dtype=torch.long)
    pcounts = torch.zeros((batch_size, max_len), dtype=torch.long)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for i, sample in enumerate(samples):
        n = len(sample.skills)
        skills[i, :n] = sample.skills
        corrects[i, :n] = sample.corrects
        rgaps[i, :n] = sample.rgaps
        sgaps[i, :n] = sample.sgaps
        pcounts[i, :n] = sample.pcounts
        mask[i, :n] = True

    return {
        "skills": skills,
        "corrects": corrects,
        "rgaps": rgaps,
        "sgaps": sgaps,
        "pcounts": pcounts,
        "mask": mask,
    }


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


class DKTForgetModel(nn.Module):
    """DKT with explicit forgetting features (rgap, sgap, pcount)."""

    def __init__(
        self,
        num_skills: int,
        num_rgap: int,
        num_sgap: int,
        num_pcount: int,
        *,
        emb_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_skills = int(num_skills)
        self.num_rgap = int(num_rgap)
        self.num_sgap = int(num_sgap)
        self.num_pcount = int(num_pcount)
        self.emb_dim = int(emb_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout_p = float(dropout)

        self.interaction_emb = nn.Embedding(self.num_skills * 2, self.emb_dim)
        self.register_buffer("rgap_eye", torch.eye(self.num_rgap))
        self.register_buffer("sgap_eye", torch.eye(self.num_sgap))
        self.register_buffer("pcount_eye", torch.eye(self.num_pcount))

        total_input_dim = self.emb_dim + self.num_rgap + self.num_sgap + self.num_pcount
        self.lstm = nn.LSTM(total_input_dim, self.hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(self.dropout_p)
        self.fc = nn.Linear(self.hidden_dim, self.num_skills)

    def forward(
        self,
        skills: torch.Tensor,
        corrects: torch.Tensor,
        rgaps: torch.Tensor,
        sgaps: torch.Tensor,
        pcounts: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(skills, corrects, rgaps, sgaps, pcounts)

        interaction_idx = skills + corrects.long() * self.num_skills
        interaction_vec = self.interaction_emb(interaction_idx)

        rgap_vec = self.rgap_eye[rgaps]
        sgap_vec = self.sgap_eye[sgaps]
        pcount_vec = self.pcount_eye[pcounts]

        features = torch.cat([interaction_vec, rgap_vec, sgap_vec, pcount_vec], dim=-1)
        hidden, _ = self.lstm(features)
        logits = self.fc(self.dropout(hidden))
        return torch.sigmoid(logits)

    def _validate_inputs(self, skills, corrects, rgaps, sgaps, pcounts) -> None:
        if torch.any(skills < 0) or torch.any(skills >= self.num_skills):
            raise ValueError("skill index outside model range")
        if torch.any(corrects < 0) or torch.any(corrects > 1):
            raise ValueError("corrects must be 0/1")
        if torch.any(rgaps < 0) or torch.any(rgaps >= self.num_rgap):
            raise ValueError("rgap index outside model range")
        if torch.any(sgaps < 0) or torch.any(sgaps >= self.num_sgap):
            raise ValueError("sgap index outside model range")
        if torch.any(pcounts < 0) or torch.any(pcounts >= self.num_pcount):
            raise ValueError("pcount index outside model range")


# -----------------------------------------------------------------------------
# Training / evaluation
# -----------------------------------------------------------------------------


def _next_step_vectors(
    preds: torch.Tensor,
    skills: torch.Tensor,
    corrects: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather P(correct at t+1) from predictions emitted after t."""
    valid = mask[:, 1:] & mask[:, :-1]
    target_skills = skills[:, 1:]
    target_corrects = corrects[:, 1:].float()
    previous_preds = preds[:, :-1, :]
    selected = torch.gather(previous_preds, 2, target_skills.unsqueeze(-1)).squeeze(-1)
    return selected[valid], target_corrects[valid]


def evaluate_model(model: DKTForgetModel, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    y_true: List[float] = []
    y_pred: List[float] = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            preds = model(
                batch["skills"], batch["corrects"], batch["rgaps"], batch["sgaps"], batch["pcounts"]
            )
            selected, targets = _next_step_vectors(
                preds, batch["skills"], batch["corrects"], batch["mask"]
            )
            y_pred.extend(selected.detach().cpu().tolist())
            y_true.extend(targets.detach().cpu().tolist())

    if not y_true:
        raise ValueError("Validation set has no valid next-step targets")

    result = {
        "acc": float(accuracy_score(y_true, np.asarray(y_pred) >= 0.5)),
    }
    # AUC requires both classes.
    result["auc"] = float(roc_auc_score(y_true, y_pred)) if len(set(y_true)) > 1 else float("nan")
    return result


def train_model(
    csv_path: Union[str, os.PathLike],
    output_path: Union[str, os.PathLike] = "edu_fin_dkt_forget.pth",
    *,
    expected_num_skills=EDUFIN_NUM_SKILLS,
    skill2idx=EDUFIN_SKILL2IDX,
    min_interactions: int = 3,
    emb_dim: int = 64,
    hidden_dim: int = 128,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
    epochs: int = 30,
    batch_size: int = 32,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> Dict[str, object]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DKTForgetDataset(
        csv_path,
        min_interactions=min_interactions,
        skill2idx=skill2idx,
        expected_num_skills=expected_num_skills,
    )

    if len(dataset) < 2:
        raise ValueError("At least two student sequences are required for train/validation split")

    val_size = max(1, int(round(len(dataset) * validation_fraction)))
    train_size = len(dataset) - val_size
    if train_size < 1:
        train_size, val_size = len(dataset) - 1, 1

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch,
        generator=generator,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    model = DKTForgetModel(
        num_skills=dataset.num_skills,
        num_rgap=dataset.num_rgap,
        num_sgap=dataset.num_sgap,
        num_pcount=dataset.num_pcount,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    best_auc = -float("inf")
    best_state = None
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        batches = 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            preds = model(
                batch["skills"], batch["corrects"], batch["rgaps"], batch["sgaps"], batch["pcounts"]
            )
            selected, targets = _next_step_vectors(
                preds, batch["skills"], batch["corrects"], batch["mask"]
            )
            if selected.numel() == 0:
                continue

            loss = criterion(selected, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += float(loss.item())
            batches += 1

        if batches == 0:
            raise ValueError("Training set produced no valid next-step targets")

        metrics = evaluate_model(model, val_loader, device)
        row = {
            "epoch": epoch,
            "loss": running_loss / batches,
            "val_auc": metrics["auc"],
            "val_acc": metrics["acc"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{epochs} | loss={row['loss']:.4f} | "
            f"val_auc={row['val_auc']:.4f} | val_acc={row['val_acc']:.4f}"
        )

        score = metrics["auc"] if not math.isnan(metrics["auc"]) else metrics["acc"]
        if score > best_auc:
            best_auc = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    checkpoint = {
        "format_version": 1,
        "model_name": "dkt_forget",
        "state_dict": best_state,
        "num_skills": dataset.num_skills,
        "skill2idx": dataset.skill2idx,
        "idx2skill": dataset.idx2skill,
        "num_rgap": dataset.num_rgap,
        "num_sgap": dataset.num_sgap,
        "num_pcount": dataset.num_pcount,
        "emb_dim": emb_dim,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "temporal_encoding": "pykt_log2_minutes_v1",
        "training": {
            "seed": seed,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "min_interactions": min_interactions,
            "validation_fraction": validation_fraction,
        },
        "history": history,
    }
    torch.save(checkpoint, output_path)
    return checkpoint


# -----------------------------------------------------------------------------
# Checkpoint loading and online inference
# -----------------------------------------------------------------------------


def load_checkpoint(
    checkpoint_path: Union[str, os.PathLike],
    *,
    device: Optional[torch.device] = None,
) -> Tuple[DKTForgetModel, Dict[str, object]]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = DKTForgetModel(
        num_skills=int(checkpoint["num_skills"]),
        num_rgap=int(checkpoint["num_rgap"]),
        num_sgap=int(checkpoint["num_sgap"]),
        num_pcount=int(checkpoint["num_pcount"]),
        emb_dim=int(checkpoint["emb_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint.get("dropout", 0.2)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def predict_mastery(
    model: DKTForgetModel,
    checkpoint: Mapping[str, object],
    interactions: Sequence[Mapping[str, object]],
    *,
    device: Optional[torch.device] = None,
) -> Dict[Union[str, int], float]:
    """Return next-response probability for every EDUFIN skill.

    Each interaction requires: skill_id, correct, timestamp.
    The returned values are DKT next-response probabilities, which can be used
    by the adaptive layer as the model's mastery proxy.
    """
    if not interactions:
        raise ValueError("At least one interaction is required")

    device = device or next(model.parameters()).device
    skill2idx = checkpoint["skill2idx"]
    idx2skill = checkpoint["idx2skill"]

    skill_ids = [x["skill_id"] for x in interactions]
    unknown = [s for s in skill_ids if s not in skill2idx]
    if unknown:
        raise ValueError(f"Unknown skill_id(s): {unknown}")

    skills = [int(skill2idx[s]) for s in skill_ids]
    corrects = [int(bool(x["correct"])) for x in interactions]
    timestamps = [x["timestamp"] for x in interactions]
    rgaps, sgaps, pcounts = compute_forgetting_features(skills, timestamps)

    # A future online sequence can produce a larger temporal bin than any seen
    # in training. Clamp to the last learned bucket rather than crashing.
    rgaps = [min(v, model.num_rgap - 1) for v in rgaps]
    sgaps = [min(v, model.num_sgap - 1) for v in sgaps]
    pcounts = [min(v, model.num_pcount - 1) for v in pcounts]

    def t(values: Sequence[int]) -> torch.Tensor:
        return torch.tensor([values], dtype=torch.long, device=device)

    with torch.no_grad():
        probs = model(t(skills), t(corrects), t(rgaps), t(sgaps), t(pcounts))[0, -1]

    result: Dict[Union[str, int], float] = {}
    for idx, probability in enumerate(probs.detach().cpu().tolist()):
        # torch serialization can preserve int keys; JSON may convert them to str.
        original_skill = idx2skill.get(idx, idx2skill.get(str(idx), idx))
        result[original_skill] = float(probability)
    return result


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train EDUFIN DKT-Forget")
    parser.add_argument("csv", help="CSV with user_id, skill_id, correct, timestamp")
    parser.add_argument("--output", default="edu_fin_dkt_forget.pth")
    parser.add_argument("--num-skills", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    checkpoint = train_model(
        args.csv,
        args.output,
        expected_num_skills=args.num_skills,
        epochs=args.epochs,
        batch_size=args.batch_size,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        learning_rate=args.lr,
        seed=args.seed,
    )
    print(f"\nSaved: {args.output}")
    print(
        json.dumps(
            {
                "num_skills": checkpoint["num_skills"],
                "num_rgap": checkpoint["num_rgap"],
                "num_sgap": checkpoint["num_sgap"],
                "num_pcount": checkpoint["num_pcount"],
                "temporal_encoding": checkpoint["temporal_encoding"],
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
