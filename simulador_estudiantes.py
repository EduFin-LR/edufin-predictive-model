"""
Simulador EDUFIN alineado con el flujo actual del producto.

Objetivo:
- Generar datos sintéticos para preentrenar DKT-Forget.
- Mantener la simulación del aprendizaje/olvido del estudiante separada
  del modelo DKT-Forget que luego se entrena con esas interacciones.

IMPORTANTE:
- Las actividades LESSON pueden modificar el conocimiento latente del estudiante
  en el simulador, pero NO se escriben como interacciones DKT.
- Solo se exportan:
    QUIZ
    REINFORCEMENT
    FINAL

Contrato mínimo del modelo:
    user_id, skill_id, correct, timestamp

Metadata adicional:
    interaction_type, selection_reason
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


PARAMS = {
    # ---------------- Población ----------------
    "habilidad_media": 0.0,
    "habilidad_de": 0.8,
    "ruido_skill_de": 0.3,

    # Positivo = más fácil para la población simulada.
    "facilidad_modulo": {
        1: 0.3,
        2: 0.0,
        3: 0.0,
        4: -0.4,
        5: -0.4,
        6: -0.2,
        7: 0.0,
    },

    # ---------------- Preguntas ----------------
    "b_por_dificultad": {
        1: -0.8,
        2: 0.2,
        3: 1.0,
        None: 0.0,
    },
    "ruido_item_de": 0.25,
    "adivinar_opcion_multiple": 0.25,
    "adivinar_arrastrar": 0.05,
    "discriminacion": 1.7,

    # ---------------- Aprendizaje latente ----------------
    "tasa_aprendizaje_de_log": 0.3,
    "ganancia_lectura_leccion": 0.20,
    "ganancia_acierto": 0.10,
    "ganancia_error_con_feedback": 0.06,
    "transferencia_mismo_modulo": 0.10,

    # ---------------- Olvido del simulador ----------------
    # Esto solo genera comportamiento sintético plausible.
    # NO se añade esta curva al modelo DKT-Forget en producción.
    "estabilidad_inicial_dias": 4.0,
    "retencion_minima": 0.25,
    "estabilidad_de_log": 0.3,
    "aumento_estabilidad_espaciado": 1.0,
    "aumento_estabilidad_masivo": 0.02,

    # ---------------- Uso de EDUFIN ----------------
    "dias_entre_sesiones_media": 1.5,
    "segundos_por_pregunta": (20, 90),

    # QUIZ actual de EDUFIN:
    # hasta 10 preguntas = estándar + refuerzo.
    "quiz_total": 10,
    "quiz_refuerzo_objetivo": 2,

    # FINAL dinámico EDUFIN:
    # 10 preguntas base + 3 a 5 adicionales.
    # Las adicionales priorizan LOW_MASTERY y,
    # si faltan, se completan con STANDARD.
    "final_base": 10,
    "final_extra_min": 3,
    "final_extra_max": 5,
}


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class Banco:
    def __init__(self, ruta_json: str | Path, rng: random.Random, params: Mapping[str, object]):
        data = json.load(open(ruta_json, encoding="utf-8"))

        self.items = {row["id"]: row for row in data}
        self.skills: List[int] = sorted({
            int(row["competencia_dkt"])
            for row in data
            if row.get("competencia_dkt") is not None
        })

        if self.skills != list(range(1, 31)):
            raise ValueError(
                f"Se esperaban skills EDUFIN 1..30 y se obtuvo: {self.skills}"
            )

        self.lesson_items: Dict[int, List[str]] = defaultdict(list)
        self.quiz_items: Dict[int, List[str]] = defaultdict(list)

        self.modulo: Dict[int, int] = {}
        self.concepto_id: Dict[int, str] = {}

        for row in data:
            skill = int(row["competencia_dkt"])
            self.modulo[skill] = int(row["modulo_orden"])
            self.concepto_id[skill] = str(row["concepto_id"])

            if row["tipo_leccion"] == "LESSON":
                self.lesson_items[skill].append(row["id"])
            elif row["tipo_leccion"] == "QUIZ":
                self.quiz_items[skill].append(row["id"])

        self.skills_por_modulo: Dict[int, List[int]] = defaultdict(list)
        for skill in self.skills:
            self.skills_por_modulo[self.modulo[skill]].append(skill)

        self.modulos = sorted(self.skills_por_modulo)

        b_por_dificultad = params["b_por_dificultad"]
        ruido_item_de = float(params["ruido_item_de"])

        self.b: Dict[str, float] = {}
        for item_id, row in self.items.items():
            dificultad = row.get("dificultad")
            base = b_por_dificultad.get(dificultad, b_por_dificultad.get(None, 0.0))
            self.b[item_id] = float(base) + rng.gauss(0, ruido_item_de)


class Estudiante:
    def __init__(
        self,
        sid: str,
        banco: Banco,
        rng: random.Random,
        params: Mapping[str, object],
        start_at: datetime,
    ):
        self.sid = sid
        self.B = banco
        self.rng = rng
        self.P = params
        self.start_at = start_at

        habilidad_general = rng.gauss(
            float(params["habilidad_media"]),
            float(params["habilidad_de"]),
        )

        self.eta = math.exp(
            rng.gauss(0, float(params["tasa_aprendizaje_de_log"]))
        )

        self.theta0: Dict[int, float] = {}
        self.G: Dict[int, float] = {}
        self.S: Dict[int, float] = {}
        self.t_ref: Dict[int, float] = {}

        facilidad_modulo = params["facilidad_modulo"]

        for skill in banco.skills:
            modulo = banco.modulo[skill]
            self.theta0[skill] = (
                habilidad_general
                + float(facilidad_modulo[modulo])
                + rng.gauss(0, float(params["ruido_skill_de"]))
            )
            self.G[skill] = 0.0
            self.S[skill] = (
                float(params["estabilidad_inicial_dias"])
                * math.exp(
                    rng.gauss(0, float(params["estabilidad_de_log"]))
                )
            )
            self.t_ref[skill] = 0.0

        self.t = 0.0  # días desde start_at
        self.vistos: set[str] = set()

        self.registro: List[dict] = []

        self.aciertos: Dict[int, List[int]] = defaultdict(lambda: [0, 0])
        self.ultima_practica: Dict[int, float] = {}
        self.skills_observadas: set[int] = set()

    # ------------------------------------------------------------------
    # Conocimiento latente
    # ------------------------------------------------------------------

    def theta(self, skill: int) -> float:
        return self.theta0[skill] + self.G[skill] * self._ret(skill)

    def _ret(self, skill: int) -> float:
        rho = float(self.P["retencion_minima"])
        dt = self.t - self.t_ref[skill]
        return rho + (1 - rho) * math.exp(-dt / self.S[skill])

    def _consolidar(self, skill: int) -> None:
        dt = self.t - self.t_ref[skill]
        self.G[skill] *= self._ret(skill)

        if dt > 0.5:
            aumento = float(self.P["aumento_estabilidad_espaciado"])
        else:
            aumento = float(self.P["aumento_estabilidad_masivo"])

        self.S[skill] *= 1 + aumento
        self.t_ref[skill] = self.t

    def _aprender(self, skill: int, ganancia: float) -> None:
        self._consolidar(skill)
        self.G[skill] += ganancia * self.eta

        transferencia = float(self.P["transferencia_mismo_modulo"])

        for other in self.B.skills:
            if (
                other != skill
                and self.B.modulo[other] == self.B.modulo[skill]
            ):
                self.G[other] = (
                    self.G[other] * self._ret(other)
                    + transferencia * ganancia * self.eta
                )
                self.t_ref[other] = self.t

    # ------------------------------------------------------------------
    # Tiempo
    # ------------------------------------------------------------------

    def _timestamp_iso(self) -> str:
        current = self.start_at + timedelta(days=self.t)
        return current.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    def _avanzar_pregunta(self) -> None:
        segundos_min, segundos_max = self.P["segundos_por_pregunta"]
        self.t += self.rng.uniform(segundos_min, segundos_max) / 86400.0

    def _nueva_sesion(self) -> None:
        media = float(self.P["dias_entre_sesiones_media"])
        self.t += max(
            0.3,
            self.rng.expovariate(1 / media),
        )

    # ------------------------------------------------------------------
    # LESSON: modifica conocimiento latente, pero NO crea fila DKT
    # ------------------------------------------------------------------

    def estudiar_leccion(self, skill: int) -> None:
        lesson_items = self.B.lesson_items.get(skill, [])

        # La app real muestra contenido educativo antes del quiz.
        # Conservamos ese efecto en el simulador, pero no exportamos
        # ninguna interacción LESSON hacia DKT.
        repeticiones = max(1, len(lesson_items))

        ganancia_total = (
            float(self.P["ganancia_lectura_leccion"])
            * min(repeticiones, 3)
            / 3.0
        )

        self._aprender(skill, ganancia_total)

        # Tiempo de lectura aproximado, sin crear interacción DKT.
        self.t += self.rng.uniform(120, 420) / 86400.0

    # ------------------------------------------------------------------
    # Respuesta evaluativa
    # ------------------------------------------------------------------

    def responder(
        self,
        item_id: str,
        interaction_type: str,
        selection_reason: str,
        activity_module_id: Optional[int] = None,
    ) -> None:
        item = self.B.items[item_id]
        skill = int(item["competencia_dkt"])

        tipo = item["tipo_pregunta"]

        if tipo == "MULTIPLE_CHOICE":
            c = float(self.P["adivinar_opcion_multiple"])
        else:
            c = float(self.P["adivinar_arrastrar"])

        theta = self.theta(skill)

        p = c + (1 - c) * sigmoid(
            float(self.P["discriminacion"])
            * (theta - self.B.b[item_id])
        )

        correcto = int(self.rng.random() < p)

        self.registro.append({
            # Contrato usado por dkt_forget_final.py:
            "user_id": self.sid,
            "skill_id": skill,
            "correct": correcto,
            "timestamp": self._timestamp_iso(),

            # Metadata EDUFIN:
            "interaction_type": interaction_type,
            "selection_reason": selection_reason,

            # Columnas auxiliares para análisis del simulador:
            "item_id": item_id,
            # Módulo de origen de la pregunta.
            "module_id": self.B.modulo[skill],

            # Módulo/actividad en que fue presentada.
            # En FINAL una pregunta LOW_MASTERY puede venir de un módulo previo.
            "activity_module_id": (
                activity_module_id
                if activity_module_id is not None
                else self.B.modulo[skill]
            ),

            "concept_id": self.B.concepto_id[skill],
            "question_type": tipo,
            "difficulty": item.get("dificultad") or "",
            "true_probability": round(p, 6),
            "latent_theta": round(theta, 6),
        })

        if correcto:
            ganancia = float(self.P["ganancia_acierto"])
        else:
            ganancia = float(self.P["ganancia_error_con_feedback"])

        self._aprender(skill, ganancia)

        self.aciertos[skill][0] += correcto
        self.aciertos[skill][1] += 1
        self.ultima_practica[skill] = self.t
        self.skills_observadas.add(skill)
        self.vistos.add(item_id)

        self._avanzar_pregunta()

    # ------------------------------------------------------------------
    # Selección
    # ------------------------------------------------------------------

    def _elegir(self, pool: Sequence[str], n: int) -> List[str]:
        if n <= 0 or not pool:
            return []

        nuevos = [item for item in pool if item not in self.vistos]
        self.rng.shuffle(nuevos)
        seleccion = nuevos[:n]

        if len(seleccion) < n:
            resto = [item for item in pool if item not in seleccion]
            self.rng.shuffle(resto)
            seleccion += resto[: n - len(seleccion)]

        return seleccion[:n]

    def skills_low_mastery(self, candidates: Iterable[int], n: int) -> List[int]:
        """
        Heurística SOLO del simulador para decidir qué skill reforzar.
        La app real usa el mastery de DKT-Forget.
        """
        candidates = list(dict.fromkeys(int(x) for x in candidates))

        def score(skill: int) -> float:
            aciertos, intentos = self.aciertos[skill]
            accuracy_suavizada = (aciertos + 1) / (intentos + 2)
            dias = self.t - self.ultima_practica.get(skill, 0.0)

            return (
                (1 - accuracy_suavizada)
                + 0.1 * math.log1p(max(0.0, dias))
                + self.rng.random() * 0.05
            )

        return sorted(
            candidates,
            key=score,
            reverse=True,
        )[:n]

    def seleccionar_estandar_modulo(self, modulo: int, n: int) -> List[str]:
        if n <= 0:
            return []

        skills = self.B.skills_por_modulo[modulo][:]
        if not skills:
            return []

        items: List[str] = []

        # Reparte preguntas entre las skills del módulo.
        cursor = 0
        intentos = 0

        while len(items) < n and intentos < n * max(3, len(skills)):
            skill = skills[cursor % len(skills)]
            cursor += 1
            intentos += 1

            elegidos = self._elegir(
                self.B.quiz_items.get(skill, []),
                1,
            )

            for item_id in elegidos:
                if item_id not in items:
                    items.append(item_id)
                    if len(items) >= n:
                        break

        return items

    # ------------------------------------------------------------------
    # Ruta EDUFIN
    # ------------------------------------------------------------------

    def ejecutar_quiz_skill(self, skill: int) -> None:
        total = int(self.P["quiz_total"])
        objetivo_refuerzo = int(self.P["quiz_refuerzo_objetivo"])

        previas = sorted(
            s
            for s in self.skills_observadas
            if s != skill
        )

        n_refuerzo = min(
            objetivo_refuerzo,
            len(previas),
            total,
        )

        n_standard = total - n_refuerzo

        items: List[tuple[str, str, str]] = []

        for item_id in self._elegir(
            self.B.quiz_items.get(skill, []),
            n_standard,
        ):
            items.append((
                item_id,
                "QUIZ",
                "STANDARD",
            ))

        for weak_skill in self.skills_low_mastery(
            previas,
            n_refuerzo,
        ):
            chosen = self._elegir(
                self.B.quiz_items.get(weak_skill, []),
                1,
            )

            for item_id in chosen:
                items.append((
                    item_id,
                    "REINFORCEMENT",
                    "LOW_MASTERY",
                ))

        self.rng.shuffle(items)

        for item_id, interaction_type, reason in items[:total]:
            self.responder(
                item_id,
                interaction_type,
                reason,
                activity_module_id=self.B.modulo[skill],
            )

    def ejecutar_final_modulo(self, modulo: int) -> None:
        """
        Replica la política actual del backend:

        - 10 preguntas BASE equilibradas entre skills del módulo.
        - 3 a 5 preguntas adicionales.
        - Las adicionales priorizan LOW_MASTERY.
        - Si faltan adaptativas, se completa con STANDARD.
        - Todas usan interaction_type = FINAL.
        """
        base_total = int(self.P["final_base"])
        extra_min = int(self.P["final_extra_min"])
        extra_max = int(self.P["final_extra_max"])

        skills_modulo = list(self.B.skills_por_modulo[modulo])

        if not skills_modulo:
            return

        final_items: List[tuple[str, str, str]] = []
        usados: set[str] = set()

        # --------------------------------------------------------------
        # 1. 10 preguntas base equilibradas entre skills del módulo
        # --------------------------------------------------------------
        base_per_skill = base_total // len(skills_modulo)
        remainder = base_total % len(skills_modulo)

        for idx, skill in enumerate(skills_modulo):
            quota = base_per_skill + (1 if idx < remainder else 0)

            if quota <= 0:
                continue

            chosen = self._elegir(
                self.B.quiz_items.get(skill, []),
                quota,
            )

            for item_id in chosen:
                if item_id in usados:
                    continue

                usados.add(item_id)

                final_items.append((
                    item_id,
                    "FINAL",
                    "STANDARD",
                ))

        # Rellenar la base si alguna skill no tenía suficientes preguntas
        if len(final_items) < base_total:
            extras_base = self.seleccionar_estandar_modulo(
                modulo,
                base_total - len(final_items) + 10,
            )

            for item_id in extras_base:
                if item_id in usados:
                    continue

                usados.add(item_id)
                final_items.append((
                    item_id,
                    "FINAL",
                    "STANDARD",
                ))

                if len(final_items) >= base_total:
                    break

        # --------------------------------------------------------------
        # 2. Hasta 5 preguntas extra LOW_MASTERY
        # --------------------------------------------------------------
        candidatas = sorted(self.skills_observadas)

        weak_skills = self.skills_low_mastery(
            candidatas,
            min(extra_max, len(candidatas)),
        )

        adaptive_added = 0

        for weak_skill in weak_skills:
            if adaptive_added >= extra_max:
                break

            # Una pregunta por skill débil; el backend hace la selección
            # por skill de forma análoga.
            chosen = self._elegir(
                self.B.quiz_items.get(weak_skill, []),
                1,
            )

            for item_id in chosen:
                if item_id in usados:
                    continue

                usados.add(item_id)

                final_items.append((
                    item_id,
                    "FINAL",
                    "LOW_MASTERY",
                ))

                adaptive_added += 1
                break

        # --------------------------------------------------------------
        # 3. Target final 13..15
        # --------------------------------------------------------------
        desired_extra_count = max(
            extra_min,
            min(adaptive_added, extra_max),
        )

        target_total = base_total + desired_extra_count

        # --------------------------------------------------------------
        # 4. Completar STANDARD si faltan preguntas
        # --------------------------------------------------------------
        if len(final_items) < target_total:
            standard_fill = self.seleccionar_estandar_modulo(
                modulo,
                target_total - len(final_items) + 10,
            )

            for item_id in standard_fill:
                if item_id in usados:
                    continue

                usados.add(item_id)

                final_items.append((
                    item_id,
                    "FINAL",
                    "STANDARD",
                ))

                if len(final_items) >= target_total:
                    break

        # Seguridad: máximo 15.
        final_items = final_items[: base_total + extra_max]

        self.rng.shuffle(final_items)

        for item_id, interaction_type, reason in final_items:
            self.responder(
                item_id,
                interaction_type,
                reason,
                activity_module_id=modulo,
            )

    def recorrer_ruta(self) -> List[dict]:
        for modulo in self.B.modulos:

            for skill in self.B.skills_por_modulo[modulo]:
                self.estudiar_leccion(skill)
                self.ejecutar_quiz_skill(skill)
                self._nueva_sesion()

            # FINAL dinámico de 10 preguntas al terminar el módulo.
            self.ejecutar_final_modulo(modulo)
            self._nueva_sesion()

        return self.registro


def simular(
    n_estudiantes: int,
    ruta_banco: str | Path,
    *,
    semilla: int = 42,
    params: Optional[Mapping[str, object]] = None,
    prefijo: str = "sim",
) -> List[dict]:
    if n_estudiantes <= 0:
        raise ValueError("n_estudiantes debe ser mayor que 0")

    P = dict(PARAMS)

    if params:
        P.update(params)

    rng = random.Random(semilla)
    banco = Banco(ruta_banco, rng, P)

    filas: List[dict] = []

    base_start = datetime(
        2026,
        1,
        1,
        8,
        0,
        tzinfo=timezone.utc,
    )

    for idx in range(n_estudiantes):
        # Pequeña variación en el inicio de cada estudiante.
        offset = timedelta(
            days=rng.uniform(0, 30),
            hours=rng.uniform(0, 12),
        )

        estudiante = Estudiante(
            f"{prefijo}_{idx:05d}",
            banco,
            rng,
            P,
            base_start + offset,
        )

        filas.extend(
            estudiante.recorrer_ruta()
        )

    return filas


def guardar_csv(
    filas: Sequence[Mapping[str, object]],
    ruta: str | Path,
) -> None:
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    if not filas:
        raise ValueError("No hay filas para guardar")

    columnas = [
        "user_id",
        "skill_id",
        "correct",
        "timestamp",
        "interaction_type",
        "selection_reason",
        "item_id",
        "module_id",
        "activity_module_id",
        "concept_id",
        "question_type",
        "difficulty",
        "true_probability",
        "latent_theta",
    ]

    with ruta.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columnas,
        )

        writer.writeheader()

        for row in filas:
            writer.writerow({
                key: row.get(key, "")
                for key in columnas
            })


def validar_dataset(filas: Sequence[Mapping[str, object]]) -> None:
    permitidos = {
        "QUIZ",
        "REINFORCEMENT",
        "FINAL",
    }

    razones = {
        "STANDARD",
        "LOW_MASTERY",
    }

    for row in filas:
        skill = int(row["skill_id"])

        if skill < 1 or skill > 30:
            raise ValueError(
                f"skill_id inválido: {skill}"
            )

        if row["interaction_type"] not in permitidos:
            raise ValueError(
                f"interaction_type inválido: {row['interaction_type']}"
            )

        if row["selection_reason"] not in razones:
            raise ValueError(
                f"selection_reason inválido: {row['selection_reason']}"
            )

        if row["interaction_type"] == "REINFORCEMENT":
            if row["selection_reason"] != "LOW_MASTERY":
                raise ValueError(
                    "REINFORCEMENT debe usar LOW_MASTERY"
                )

        # Las adaptativas del FINAL siguen siendo FINAL.
        if (
            row["interaction_type"] == "FINAL"
            and row["selection_reason"] == "LOW_MASTERY"
        ):
            pass


def resumen(filas: Sequence[Mapping[str, object]]) -> dict:
    por_tipo = defaultdict(int)
    por_razon = defaultdict(int)

    usuarios = set()
    skills = set()

    for row in filas:
        usuarios.add(row["user_id"])
        skills.add(int(row["skill_id"]))
        por_tipo[row["interaction_type"]] += 1
        por_razon[row["selection_reason"]] += 1

    return {
        "users": len(usuarios),
        "interactions": len(filas),
        "skills": sorted(skills),
        "interaction_types": dict(por_tipo),
        "selection_reasons": dict(por_razon),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera interacciones sintéticas EDUFIN para pretraining DKT-Forget"
    )

    parser.add_argument(
        "--students",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--bank",
        default="banco_preguntas_estructurado.json",
    )

    parser.add_argument(
        "--output",
        default="datasets/simulated_v1.csv",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    filas = simular(
        args.students,
        args.bank,
        semilla=args.seed,
    )

    validar_dataset(filas)
    guardar_csv(filas, args.output)

    print(
        json.dumps(
            resumen(filas),
            indent=2,
            ensure_ascii=False,
        )
    )

    print(f"\nDataset guardado en: {args.output}")


if __name__ == "__main__":
    main()
