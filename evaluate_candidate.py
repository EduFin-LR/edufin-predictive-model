"""Evaluación y promoción local de candidatos DKT-Forget para EDUFIN.

Este script NO entrena el modelo. Lee el metadata generado por fine_tune.py,
aplica reglas mínimas de aceptación y genera una decisión reproducible.

Por defecto SOLO evalúa:
    python evaluate_candidate.py

Si el candidato cumple todos los criterios y quieres promoverlo al área local
de modelos aprobados:
    python evaluate_candidate.py --promote

IMPORTANTE:
- "Promover" aquí NO actualiza FastAPI ni Railway.
- Solo copia el checkpoint aprobado a checkpoints/approved/ y actualiza
  checkpoints/approved/active_model.json.
- La publicación hacia FastAPI/Object Storage se implementará después.

Criterios por defecto:
- >= 10 usuarios elegibles
- >= 500 interacciones elegibles
- >= 3 usuarios de validation
- AUC del candidato disponible
- Delta AUC >= 0.0
- Accuracy no puede caer más de 0.03

Los umbrales se pueden cambiar por CLI.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_METADATA = Path("checkpoints/candidates/dkt_forget_candidate_v2.json")
DEFAULT_EVALUATION = Path("checkpoints/candidates/dkt_forget_candidate_v2_evaluation.json")
DEFAULT_APPROVED_DIR = Path("checkpoints/approved")
DEFAULT_ACTIVE_MANIFEST = DEFAULT_APPROVED_DIR / "active_model.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evalúa si un checkpoint candidato DKT-Forget puede ser aprobado."
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--evaluation-output", type=Path, default=DEFAULT_EVALUATION)

    parser.add_argument("--min-users", type=int, default=10)
    parser.add_argument("--min-interactions", type=int, default=500)
    parser.add_argument("--min-validation-users", type=int, default=3)
    parser.add_argument("--min-auc-delta", type=float, default=0.0)
    parser.add_argument("--max-accuracy-drop", type=float, default=0.03)

    parser.add_argument(
        "--promote",
        action="store_true",
        help="Si el candidato es aprobado, copiarlo a checkpoints/approved/ y actualizar active_model.json.",
    )
    parser.add_argument("--approved-dir", type=Path, default=DEFAULT_APPROVED_DIR)
    parser.add_argument("--active-manifest", type=Path, default=DEFAULT_ACTIVE_MANIFEST)
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"No existe el metadata del candidato: {path}")

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, dict):
        raise ValueError("El metadata del candidato debe ser un objeto JSON.")
    return data


def as_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Campo inválido '{field}': {value!r}") from exc
    return result


def as_int(value: Any, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Campo inválido '{field}': {value!r}") from exc


def evaluate(metadata: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    data = metadata.get("data") or {}
    base_metrics = metadata.get("base_metrics") or {}
    candidate_metrics = metadata.get("candidate_metrics") or {}
    delta = metadata.get("delta") or {}

    users = as_int(data.get("users"), "data.users")
    interactions = as_int(data.get("interactions"), "data.interactions")
    validation_users = as_int(data.get("validation_users"), "data.validation_users")

    base_auc = as_float(base_metrics.get("auc"), "base_metrics.auc")
    base_acc = as_float(base_metrics.get("accuracy"), "base_metrics.accuracy")
    candidate_auc = as_float(candidate_metrics.get("auc"), "candidate_metrics.auc")
    candidate_acc = as_float(candidate_metrics.get("accuracy"), "candidate_metrics.accuracy")

    delta_auc = as_float(delta.get("auc"), "delta.auc")
    delta_acc = as_float(delta.get("accuracy"), "delta.accuracy")

    reasons: List[str] = []
    checks: Dict[str, Any] = {}

    checks["min_users"] = {
        "passed": users >= args.min_users,
        "actual": users,
        "required": args.min_users,
    }
    if not checks["min_users"]["passed"]:
        reasons.append(f"usuarios insuficientes: {users}/{args.min_users}")

    checks["min_interactions"] = {
        "passed": interactions >= args.min_interactions,
        "actual": interactions,
        "required": args.min_interactions,
    }
    if not checks["min_interactions"]["passed"]:
        reasons.append(f"interacciones insuficientes: {interactions}/{args.min_interactions}")

    checks["min_validation_users"] = {
        "passed": validation_users >= args.min_validation_users,
        "actual": validation_users,
        "required": args.min_validation_users,
    }
    if not checks["min_validation_users"]["passed"]:
        reasons.append(
            f"usuarios de validation insuficientes: {validation_users}/{args.min_validation_users}"
        )

    auc_available = not (
        math.isnan(base_auc)
        or math.isnan(candidate_auc)
        or math.isnan(delta_auc)
    )
    checks["auc_available"] = {
        "passed": auc_available,
        "base_auc": base_auc,
        "candidate_auc": candidate_auc,
        "delta_auc": delta_auc,
    }
    if not auc_available:
        reasons.append("AUC no disponible o NaN; no se puede aprobar automáticamente")

    auc_improved = auc_available and delta_auc >= args.min_auc_delta
    checks["auc_delta"] = {
        "passed": auc_improved,
        "actual": delta_auc,
        "required_min": args.min_auc_delta,
    }
    if auc_available and not auc_improved:
        reasons.append(
            f"delta AUC insuficiente: {delta_auc:+.4f} < {args.min_auc_delta:+.4f}"
        )

    min_allowed_acc_delta = -abs(args.max_accuracy_drop)
    accuracy_ok = not math.isnan(delta_acc) and delta_acc >= min_allowed_acc_delta
    checks["accuracy_drop"] = {
        "passed": accuracy_ok,
        "actual_delta": delta_acc,
        "minimum_allowed_delta": min_allowed_acc_delta,
        "max_allowed_drop": abs(args.max_accuracy_drop),
    }
    if not accuracy_ok:
        reasons.append(
            f"caída de accuracy excesiva: {delta_acc:+.4f}; "
            f"mínimo permitido {min_allowed_acc_delta:+.4f}"
        )

    approved = len(reasons) == 0

    return {
        "decision": "APPROVED" if approved else "REJECTED",
        "approved": approved,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_checkpoint": metadata.get("candidate_checkpoint"),
        "base_checkpoint": metadata.get("base_checkpoint"),
        "metrics": {
            "base": {
                "auc": base_auc,
                "accuracy": base_acc,
            },
            "candidate": {
                "auc": candidate_auc,
                "accuracy": candidate_acc,
            },
            "delta": {
                "auc": delta_auc,
                "accuracy": delta_acc,
            },
        },
        "data": {
            "users": users,
            "interactions": interactions,
            "validation_users": validation_users,
        },
        "criteria": {
            "min_users": args.min_users,
            "min_interactions": args.min_interactions,
            "min_validation_users": args.min_validation_users,
            "min_auc_delta": args.min_auc_delta,
            "max_accuracy_drop": abs(args.max_accuracy_drop),
        },
        "checks": checks,
        "reasons": reasons,
        "note": (
            "La aprobación significa que el candidato superó las reglas automáticas "
            "configuradas. No sustituye evaluación científica independiente."
        ),
    }


def infer_approved_filename(candidate_path: Path) -> str:
    name = candidate_path.name
    if "candidate" in name:
        return name.replace("candidate", "approved")
    return f"approved_{name}"


def promote(
    evaluation: Dict[str, Any],
    metadata: Dict[str, Any],
    approved_dir: Path,
    active_manifest: Path,
) -> Path:
    if not evaluation["approved"]:
        raise RuntimeError("No se puede promover un candidato rechazado.")

    candidate_raw = metadata.get("candidate_checkpoint")
    if not candidate_raw:
        raise ValueError("El metadata no contiene candidate_checkpoint.")

    candidate = Path(candidate_raw)
    if not candidate.exists():
        raise FileNotFoundError(f"No existe el checkpoint candidato: {candidate}")

    approved_dir.mkdir(parents=True, exist_ok=True)
    approved_path = approved_dir / infer_approved_filename(candidate)
    shutil.copy2(candidate, approved_path)

    manifest = {
        "status": "approved_local",
        "active_checkpoint": str(approved_path),
        "source_candidate": str(candidate),
        "base_checkpoint": metadata.get("base_checkpoint"),
        "approved_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation": {
            "auc": evaluation["metrics"]["candidate"]["auc"],
            "accuracy": evaluation["metrics"]["candidate"]["accuracy"],
            "delta_auc": evaluation["metrics"]["delta"]["auc"],
            "delta_accuracy": evaluation["metrics"]["delta"]["accuracy"],
            "users": evaluation["data"]["users"],
            "interactions": evaluation["data"]["interactions"],
            "validation_users": evaluation["data"]["validation_users"],
        },
        "note": (
            "Este manifest solo marca el modelo activo dentro del proyecto de training. "
            "Todavía no publica el checkpoint a FastAPI/Railway."
        ),
    }

    active_manifest.parent.mkdir(parents=True, exist_ok=True)
    with active_manifest.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    return approved_path


def main() -> None:
    args = parse_args()

    if args.min_users < 2:
        raise ValueError("min-users debe ser al menos 2.")
    if args.min_interactions < 1:
        raise ValueError("min-interactions debe ser positivo.")
    if args.min_validation_users < 1:
        raise ValueError("min-validation-users debe ser positivo.")
    if args.max_accuracy_drop < 0:
        raise ValueError("max-accuracy-drop debe ser >= 0.")

    metadata = load_json(args.metadata)
    evaluation = evaluate(metadata, args)

    args.evaluation_output.parent.mkdir(parents=True, exist_ok=True)
    with args.evaluation_output.open("w", encoding="utf-8") as fh:
        json.dump(evaluation, fh, ensure_ascii=False, indent=2)

    print("Evaluación de candidato EDUFIN DKT-Forget")
    print("----------------------------------------")
    print(f"Candidato:          {evaluation['candidate_checkpoint']}")
    print(f"Usuarios:           {evaluation['data']['users']}")
    print(f"Interacciones:      {evaluation['data']['interactions']}")
    print(f"Validation users:   {evaluation['data']['validation_users']}")

    print("\nMétricas")
    print(
        f"AUC:      {evaluation['metrics']['base']['auc']:.4f} "
        f"-> {evaluation['metrics']['candidate']['auc']:.4f} "
        f"({evaluation['metrics']['delta']['auc']:+.4f})"
    )
    print(
        f"Accuracy: {evaluation['metrics']['base']['accuracy']:.4f} "
        f"-> {evaluation['metrics']['candidate']['accuracy']:.4f} "
        f"({evaluation['metrics']['delta']['accuracy']:+.4f})"
    )

    print(f"\nDecisión: {evaluation['decision']}")

    if evaluation["reasons"]:
        print("Motivos:")
        for reason in evaluation["reasons"]:
            print(f"- {reason}")

    print(f"\nReporte: {args.evaluation_output}")

    if args.promote:
        if evaluation["approved"]:
            approved_path = promote(
                evaluation,
                metadata,
                approved_dir=args.approved_dir,
                active_manifest=args.active_manifest,
            )
            print("\nPromoción local completada.")
            print(f"Checkpoint aprobado: {approved_path}")
            print(f"Manifest activo:      {args.active_manifest}")
            print("IMPORTANTE: FastAPI/Railway todavía NO fue actualizado.")
        else:
            print("\nNo se promovió el checkpoint porque fue rechazado.")
    else:
        print("\nNo se realizó promoción. Usa --promote solo cuando quieras aprobar localmente.")


if __name__ == "__main__":
    main()
