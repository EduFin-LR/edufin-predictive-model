from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parent

EXPORT_SCRIPT = ROOT / "export_real_interactions.py"
FINE_TUNE_SCRIPT = ROOT / "fine_tune.py"
EVALUATE_SCRIPT = ROOT / "evaluate_candidate.py"
PUBLISH_SCRIPT = ROOT / "publish_model.py"

CANDIDATE_METADATA = ROOT / "checkpoints" / "candidates" / "dkt_forget_candidate_v2.json"
EVALUATION_REPORT = ROOT / "checkpoints" / "candidates" / "dkt_forget_candidate_v2_evaluation.json"
APPROVED_MANIFEST = ROOT / "checkpoints" / "approved" / "active_model.json"


def run_step(name: str, command: Sequence[str]) -> None:
    print()
    print("=" * 64)
    print(name)
    print("=" * 64)
    print("Comando:", " ".join(command))
    print()

    result = subprocess.run(list(command), cwd=ROOT, check=False)

    if result.returncode != 0:
        raise RuntimeError(f"{name} falló con exit code {result.returncode}.")


def require_script(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo requerido: {path.name}")


def mtime(path: Path):
    return path.stat().st_mtime if path.exists() else None


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON inválido en {path}")
    return data


def main() -> None:
    print("Pipeline automático EDUFIN DKT-Forget")
    print("=" * 64)

    for script in (EXPORT_SCRIPT, FINE_TUNE_SCRIPT, EVALUATE_SCRIPT, PUBLISH_SCRIPT):
        require_script(script)

    python = sys.executable

    run_step("1/4 - Exportar interacciones reales", [python, str(EXPORT_SCRIPT)])

    candidate_before = mtime(CANDIDATE_METADATA)

    run_step("2/4 - Fine-tuning", [python, str(FINE_TUNE_SCRIPT)])

    candidate_after = mtime(CANDIDATE_METADATA)

    if candidate_after is None or candidate_after == candidate_before:
        print()
        print("=" * 64)
        print("Pipeline finalizado sin candidato nuevo")
        print("=" * 64)
        print("No se generó un candidato nuevo.")
        print("Probablemente aún no se cumplen los mínimos de usuarios/interacciones.")
        print("No se evaluó ni publicó ningún modelo.")
        return

    evaluation_before = mtime(EVALUATION_REPORT)

    run_step(
        "3/4 - Evaluar candidato",
        [python, str(EVALUATE_SCRIPT), "--promote"],
    )

    evaluation_after = mtime(EVALUATION_REPORT)

    if evaluation_after is None or evaluation_after == evaluation_before:
        raise RuntimeError(
            "evaluate_candidate.py no generó un reporte de evaluación nuevo."
        )

    evaluation = load_json(EVALUATION_REPORT)

    if not bool(evaluation.get("approved")):
        print()
        print("=" * 64)
        print("Candidato rechazado")
        print("=" * 64)
        for reason in evaluation.get("reasons") or []:
            print(f"- {reason}")
        print()
        print("El modelo activo del bucket NO fue modificado.")
        return

    if not APPROVED_MANIFEST.exists():
        raise RuntimeError(
            "El candidato fue aprobado, pero falta checkpoints/approved/active_model.json."
        )

    run_step("4/4 - Publicar modelo aprobado", [python, str(PUBLISH_SCRIPT)])

    print()
    print("=" * 64)
    print("Pipeline EDUFIN completado")
    print("=" * 64)
    print("El checkpoint aprobado fue publicado y active_model.json fue actualizado.")
    print("FastAPI podrá cargar el nuevo modelo remoto en su siguiente arranque/redeploy.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 64)
        print("PIPELINE FALLIDO")
        print("=" * 64)
        print(str(exc))
        sys.exit(1)
