"""
Preentrenamiento del DKT-Forget de producción EDUFIN.

Usa exactamente dkt_forget_final.py.
No usa el modelo alternativo del ZIP original.
"""

from pathlib import Path

from dkt_forget_final import train_model


DATASET = Path("datasets/simulated_v1.csv")
OUTPUT = Path("checkpoints/dkt_forget_pretrained_v1.pth")


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = train_model(
        DATASET,
        OUTPUT,
        expected_num_skills=30,
        min_interactions=8,
        emb_dim=64,
        hidden_dim=128,
        dropout=0.2,
        learning_rate=1e-3,
        epochs=30,
        batch_size=32,
        validation_fraction=0.2,
        seed=42,
    )

    print("\nPretraining completado")
    print(f"Checkpoint: {OUTPUT}")
    print(f"Skills: {checkpoint['num_skills']}")


if __name__ == "__main__":
    main()
