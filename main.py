from src.train import main as entrenar


def main() -> None:
    print("=" * 60)
    print("ESCENARIO 1 — CON Credit_Score")
    print("=" * 60)
    entrenar(drop_credit_score=False)

    print()
    print("=" * 60)
    print("ESCENARIO 2 — SIN Credit_Score  (ablation study)")
    print("=" * 60)
    entrenar(drop_credit_score=True)

    print()
    print("Pipeline completo. Resultados disponibles en reports/")


if __name__ == "__main__":
    main()
