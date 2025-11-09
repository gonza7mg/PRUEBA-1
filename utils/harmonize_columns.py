import pandas as pd
import re

# --- NUEVA FUNCIÓN: normaliza y hace únicos los nombres de columnas ---
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza los nombres de columnas: minúsculas, sin acentos, guiones bajos, 
    y asegura unicidad (añade sufijos __1, __2 si hay duplicados).
    """
    REVERSE = {}  # puedes mantener o ampliar tu diccionario de mapeos personalizados

    new_cols = {}
    for c in df.columns:
        c1 = (c.strip().lower()
              .replace("á", "a").replace("é", "e").replace("í", "i")
              .replace("ó", "o").replace("ú", "u").replace("ñ", "n"))
        c1 = re.sub(r"[^\w]+", "_", c1).strip("_")
        new_cols[c] = REVERSE.get(c1, c1)

    df = df.rename(columns=new_cols)

    # Hacer nombres únicos (evitar colisiones tras normalización)
    seen: dict[str, int] = {}
    uniq = []
    for c in df.columns:
        if c not in seen:
            seen[c] = 0
            uniq.append(c)
        else:
            seen[c] += 1
            uniq.append(f"{c}__{seen[c]}")  # p.ej. 'ingresos', 'ingresos__1'
    df.columns = uniq
    return df


# --- NUEVA FUNCIÓN: conversión robusta a numérico ---
def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte a numérico las columnas relevantes, manejando duplicadas.
    Sustituye errores por NaN (más trazabilidad) y elimina warnings futuros.
    """
    numeric_like = {
        "ingresos", "gastos", "inversiones", "ebitda",
        "empleados_por_operador", "lineas", "penetracion", "cuota"
    }

    def to_num_series(s: pd.Series) -> pd.Series:
        """Conversión robusta a numérico (quita puntos de miles, cambia coma por punto)."""
        if s.dtype.kind in "biufc":  # ya es numérico
            return s
        s2 = (s.astype(str)
                .str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False))
        return pd.to_numeric(s2, errors="coerce")

    for c in df.columns:
        obj = df[c]
        if isinstance(obj, pd.DataFrame):  # caso columnas duplicadas
            for sub in obj.columns:
                df[sub] = to_num_series(obj[sub])
        else:
            if obj.dtype == "object" or c in numeric_like:
                df[c] = to_num_series(obj)

    return df
