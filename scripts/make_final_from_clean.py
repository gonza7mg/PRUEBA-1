# scripts/make_final_from_clean.py

from __future__ import annotations
from pathlib import Path
import pandas as pd

from utils.harmonize_columns import harmonize_full, profile

# Mapear nombres lógicos de dataset a CSV CLEAN
INPUTS = {
    "anual_datos_generales": "data/clean/anual_datos_generales_clean.csv",
    "anual_mercados":       "data/clean/anual_mercados_clean.csv",
    "mensual":              "data/clean/mensual_clean.csv",
    "provinciales":         "data/clean/provinciales_clean.csv",
    "trimestrales":         "data/clean/trimestrales_clean.csv",
    "infraestructuras":     "data/clean/infraestructuras_clean.csv",
}

OUT_DIR = Path("data/final")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    report_rows: list[dict] = []

    for name, rel_path in INPUTS.items():
        p = Path(rel_path)
        print(f"\n=== {name} ===")
        print(f"Leyendo CLEAN: {p}")
        df = pd.read_csv(p)

        before = profile(df).iloc[0].to_dict()

        # Pipeline completo de armonización para la capa FINAL
        df2 = harmonize_full(df, dataset=name)

        after = profile(df2).iloc[0].to_dict()

        outp = OUT_DIR / f"{name}_final.csv"
        df2.to_csv(outp, index=False, encoding="utf-8")
        print(f"    → {outp} [{len(df2):,} filas, {len(df2.columns)} cols]")

        report_rows.append({
            "dataset": name,
            "input": str(p),
            "output": str(outp),
            **{f"before_{k}": v for k, v in before.items()},
            **{f"after_{k}": v for k, v in after.items()},
        })

    rep = pd.DataFrame(report_rows)
    rep.to_csv(OUT_DIR / "final_report.csv", index=False, encoding="utf-8")
    print(f"\nReporte: {OUT_DIR / 'final_report.csv'}")


if __name__ == "__main__":
    main()
