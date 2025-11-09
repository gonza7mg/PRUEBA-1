# scripts/make_final_from_clean.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
from utils.harmonize_columns import harmonize, profile

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

report_rows = []

for name, path in INPUTS.items():
    p = Path(path)
    if not p.exists():
        print(f"[!] No se encuentra {p}")
        continue
    print(f"[+] Procesando {name} …")
    df = pd.read_csv(p)
    before = profile(df)
    df2 = harmonize(df.copy())
    after = profile(df2)

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

# Guardar reporte
rep = pd.DataFrame(report_rows)
rep.to_csv(OUT_DIR / "final_report.csv", index=False, encoding="utf-8")
print(f"\nReporte: {OUT_DIR / 'final_report.csv'}")
