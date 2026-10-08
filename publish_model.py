from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.client import Config
from botocore.exceptions import BotoCoreError, ClientError

DEFAULT_MANIFEST = Path("checkpoints/approved/active_model.json")
DEFAULT_REMOTE_MANIFEST = "models/active_model.json"


def env(name: str) -> str:
    value = os.getenv(name)

    if not value:
        raise RuntimeError(
            f"Falta la variable de entorno obligatoria: {name}"
        )

    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def create_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=env("S3_ENDPOINT"),
        region_name=env("S3_REGION"),
        aws_access_key_id=env("S3_ACCESS_KEY_ID"),
        aws_secret_access_key=env("S3_SECRET_ACCESS_KEY"),
        config=Config(
            signature_version="s3v4",
            s3={
                "addressing_style": "path"
            },
            retries={
                "max_attempts": 3,
                "mode": "standard"
            }
        )
    )


def main():

    parser = argparse.ArgumentParser(
        description="Publica un modelo DKT-Forget aprobado al bucket EDUFIN."
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST
    )

    parser.add_argument(
        "--remote-manifest-key",
        default=DEFAULT_REMOTE_MANIFEST
    )

    parser.add_argument(
        "--dry-run",
        action="store_true"
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # Leer manifest aprobado
    # --------------------------------------------------

    if not args.manifest.exists():
        raise FileNotFoundError(
            f"No existe el manifest local: {args.manifest}"
        )

    with args.manifest.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    if manifest.get("status") != "approved_local":
        raise RuntimeError(
            "El modelo no está marcado como approved_local. "
            f"Estado actual: {manifest.get('status')}"
        )

    checkpoint_raw = manifest.get("active_checkpoint")

    if not checkpoint_raw:
        raise RuntimeError(
            "El manifest no contiene active_checkpoint."
        )

    checkpoint = Path(checkpoint_raw)

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"No existe el checkpoint aprobado: {checkpoint}"
        )

    # --------------------------------------------------
    # Configuración bucket
    # --------------------------------------------------

    bucket = env("S3_BUCKET")

    model_key = f"models/{checkpoint.name}"

    checksum = sha256_file(checkpoint)

    size_bytes = checkpoint.stat().st_size

    # --------------------------------------------------
    # Manifest remoto
    # --------------------------------------------------

    remote_manifest = {

        "status": "active",

        "model_key": model_key,

        "model_filename": checkpoint.name,

        "sha256": checksum,

        "size_bytes": size_bytes,

        "published_at_utc":
            datetime.now(timezone.utc).isoformat(),

        "base_checkpoint":
            manifest.get("base_checkpoint"),

        "evaluation":
            manifest.get("evaluation")
    }

    # --------------------------------------------------
    # Resumen
    # --------------------------------------------------

    print("Publicación de modelo EDUFIN")
    print("----------------------------------------")

    print(f"Checkpoint: {checkpoint}")
    print(f"Bucket:     {bucket}")
    print(f"Model key:  {model_key}")
    print(f"SHA256:     {checksum}")
    print(f"Tamaño:     {size_bytes} bytes")

    # --------------------------------------------------
    # Dry run
    # --------------------------------------------------

    if args.dry_run:

        print()
        print("DRY RUN")
        print("No se subió ningún archivo.")

        return

    # --------------------------------------------------
    # Cliente S3
    # --------------------------------------------------

    client = create_s3_client()

    try:

        # Verificar acceso al bucket
        client.head_bucket(
            Bucket=bucket
        )

        # --------------------------------------------------
        # Subir checkpoint
        # --------------------------------------------------

        content_type, _ = mimetypes.guess_type(
            str(checkpoint)
        )

        extra_args = {}

        if content_type:
            extra_args["ContentType"] = content_type

        client.upload_file(
            Filename=str(checkpoint),
            Bucket=bucket,
            Key=model_key,
            ExtraArgs=extra_args
        )

        # Verificar que existe
        client.head_object(
            Bucket=bucket,
            Key=model_key
        )

        # --------------------------------------------------
        # Actualizar active_model.json
        # --------------------------------------------------

        manifest_bytes = json.dumps(
            remote_manifest,
            ensure_ascii=False,
            indent=2
        ).encode("utf-8")

        client.put_object(
            Bucket=bucket,
            Key=args.remote_manifest_key,
            Body=manifest_bytes,
            ContentType="application/json"
        )

    except (ClientError, BotoCoreError) as error:

        raise RuntimeError(
            f"Error publicando modelo en bucket: {error}"
        ) from error

    # --------------------------------------------------
    # Resultado
    # --------------------------------------------------

    print()
    print("Publicación completada.")

    print(
        f"Modelo: "
        f"s3://{bucket}/{model_key}"
    )

    print(
        f"Manifest: "
        f"s3://{bucket}/{args.remote_manifest_key}"
    )


if __name__ == "__main__":
    main()