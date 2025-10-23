import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from pathlib import Path
from utils.data_cleaners import CLEANERS

RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/clean")
LOG_DIR = Path("logs")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

FILE_KEY = {
    "anual_datos_generales.csv": "anual_datos_generales",
    "anual_mercados.csv": "anual_mercados",
    "mensual.csv": "mensual",
    "provinciales.csv": "provinciales",
    "trimestrales.csv": "trimestrales",
    "infraestructuras.csv": "infraestructuras",
}

def run():
    logs = []
    for fname, key in FILE_KEY.items():
        src = RAW_DIR / fname
        if not src.exists():
            logs.append({"file": fname, "status": "MISSING", "rows_in": 0, "rows_out": 0, "notes": "no encontrado"})
            continue
        df = pd.read_csv(src)
        cleaner = CLEANERS.get(key)
        rows_in = len(df)
        try:
            dfc = cleaner(df)
        except Exception as e:
            logs.append({"file": fname, "status": "ERROR", "rows_in": rows_in, "rows_out": 0, "notes": str(e)})
            continue
        rows_out = len(dfc)
        dst = CLEAN_DIR / f"{key}_clean.csv"
        dfc.to_csv(dst, index=False, encoding="utf-8")
        logs.append({"file": fname, "status": "OK", "rows_in": rows_in, "rows_out": rows_out, "notes": f"→ {dst}"})
    rep = pd.DataFrame(logs)
    rep.to_csv(LOG_DIR / "clean_report.csv", index=False, encoding="utf-8")
    print(rep)

if __name__ == "__main__":
    run()
