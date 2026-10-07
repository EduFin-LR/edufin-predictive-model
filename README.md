# EDUFIN Training

Entorno separado para simulación, preentrenamiento y posteriormente fine-tuning de DKT-Forget.

## Decisión de arquitectura

Este directorio **no es FastAPI** y **no es Spring Boot**.

Su función es:

1. Generar interacciones sintéticas alineadas con EDUFIN.
2. Preentrenar `dkt_forget_final.py`.
3. Crear un checkpoint `.pth`.
4. Posteriormente hacer fine-tuning con interacciones reales del piloto.
5. Copiar el checkpoint aprobado al servicio FastAPI para inferencia.

## Qué se cambió respecto al ZIP original

El simulador original incluía respuestas de `LESSON` dentro del dataset DKT.

En EDUFIN actual:

- `LESSON` → NO entra a DKT.
- `VIDEO` → NO entra a DKT.
- `PRE-TEST` → NO entra a DKT.
- `POST-TEST` → NO entra a DKT.
- `QUIZ` → sí entra.
- `REINFORCEMENT` → sí entra.
- `FINAL` → sí entra.

Las lecciones siguen afectando el conocimiento latente del estudiante simulado, pero no se exportan como interacciones.

## Quiz simulado

Cada skill genera hasta 10 preguntas:

- preguntas normales:
  - `interaction_type=QUIZ`
  - `selection_reason=STANDARD`
- hasta 2 preguntas provenientes de skills previas débiles:
  - `interaction_type=REINFORCEMENT`
  - `selection_reason=LOW_MASTERY`

Para la primera skill puede no haber refuerzo porque todavía no existen skills previas observadas.

## Final simulado

Al terminar cada módulo se genera un FINAL de **13 a 15 preguntas**:

- 10 preguntas BASE equilibradas entre las skills del módulo.
- 3 a 5 preguntas adicionales.
- Las adicionales priorizan `LOW_MASTERY`.
- Si no alcanzan las adaptativas, se completa con `STANDARD`.
- El máximo es 15.

Las preguntas base o de relleno:

- `interaction_type=FINAL`
- `selection_reason=STANDARD`

Las preguntas añadidas por debilidad:

- `interaction_type=FINAL`
- `selection_reason=LOW_MASTERY`

Importante: dentro de un FINAL, una pregunta adaptativa sigue siendo `FINAL`, no `REINFORCEMENT`.

El CSV incluye además `activity_module_id`. Esto distingue el módulo del FINAL
en que la pregunta fue presentada del `module_id` original de la pregunta.
Una pregunta LOW_MASTERY puede venir de un módulo previo y aun así pertenecer
al FINAL actual.

## Dataset

El CSV incluye:

```text
user_id
skill_id
correct
timestamp
interaction_type
selection_reason
item_id
module_id
activity_module_id
concept_id
question_type
difficulty
true_probability
latent_theta
```

`dkt_forget_final.py` utiliza únicamente:

```text
user_id
skill_id
correct
timestamp
```

El resto queda como metadata para análisis.

## Paso 1: crear entorno

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Si vas a entrenar con GPU NVIDIA, instala la build de PyTorch compatible con tu versión de CUDA siguiendo las instrucciones oficiales de PyTorch.

## Paso 2: generar una muestra pequeña

```powershell
python simulador_estudiantes.py --students 10 --output datasets/simulated_test.csv
```

Revisa que solo existan:

```text
QUIZ
REINFORCEMENT
FINAL
```

## Paso 3: generar dataset de pretraining

Ejemplo:

```powershell
python simulador_estudiantes.py --students 2000 --output datasets/simulated_v1.csv
```

El número de estudiantes es un parámetro experimental, no una afirmación de que 2000 sea el tamaño óptimo.

## Paso 4: preentrenar

```powershell
python train_pretrained.py
```

Resultado:

```text
checkpoints/dkt_forget_pretrained_v1.pth
```

## Paso 5: FastAPI

Todavía no copies el checkpoint a FastAPI hasta verificar:

- que el entrenamiento termine;
- que AUC/accuracy no sean inválidos;
- que el checkpoint cargue correctamente;
- que `skill_id` sea 1..30;
- que el dataset no contenga `LESSON`.

Cuando esté validado, se puede copiar como checkpoint inicial del motor de inferencia.

## Fine-tuning

El fine-tuning con `pilot_real_v1.csv` se implementará después de validar este pretraining.

No se debe mezclar todavía el dataset real con el sintético sin definir el protocolo de validación.
