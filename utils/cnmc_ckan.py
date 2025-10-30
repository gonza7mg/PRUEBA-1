
# utils/cnmc_ckan.py
from __future__ import annotations
import io
import time
import csv
import requests
import pandas as pd

CKAN_BASE = "https://datos.cnmc.gob.es/api/3/action"
# Si usas otro portal CKAN, cambia el dominio arriba.

HEADERS = {
    "User-Agent": "CNMC-DSS-TFM/1.0 (+streamlit)"},
TIMEOUT = 30

def _get(url: str, params=None, stream=False):
    """GET con cabeceras, timeout y reintentos exponenciales."""
    backoff = [0, 1, 2, 4]
    last_exc = None
    for t in backoff:
        if t:
            time.sleep(t)
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT, stream=stream, allow_redirects=True)
            # Si hay 429/5xx, reintentar
            if r.status_code in (429, 500, 502, 503, 504):
                last_exc = requests.HTTPError(f"{r.status_code} {r.reason}")
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            last_exc = e
            continue
    # Tras reintentos, propaga con detalle
    raise requests.HTTPError(f"GET failed for {url} params={params} :: {last_exc}")

def _try_datastore(resource_id: str) -> pd.DataFrame | None:
    """Intenta leer por la API de Datastore (rápido si está activo)."""
    url = f"{CKAN_BASE}/datastore_search"
    # limit alto; si fuese muy grande, paginar.
    params = {"resource_id": resource_id, "limit": 500000}
    r = _get(url, params=params)
    js = r.json()
    if not js.get("success"):
        return None
    result = js.get("result", {})
    records = result.get("records", [])
    if not records:
        # Datastore activo pero vacío: devuelve DF vacío con schema si existe
        fields = result.get("fields", [])
        cols = [f["id"] for f in fields] if fields else []
        return pd.DataFrame(columns=cols)
    return pd.DataFrame.from_records(records)

def _resource_download_url(resource_id: str) -> str:
    """Obtiene la URL de descarga de la 'resource' si no hay Datastore."""
    url = f"{CKAN_BASE}/resource_show"
    r = _get(url, params={"id": resource_id})
    js = r.json()
    if not js.get("success"):
        raise requests.HTTPError(f"CKAN resource_show not success for {resource_id}")
    res = js.get("result", {})
    # CKAN guarda el enlace directo en 'url'
    dl = res.get("url")
    if not dl:
        raise requests.HTTPError(f"No download URL in resource_show for {resource_id}")
    return dl

def _read_csv_bytes(content: bytes) -> pd.DataFrame:
    """Lee CSV detectando BOM/encoding; separador por coma/puntoycoma."""
    # Intenta UTF-8-sig primero
    try:
        buf = io.StringIO(content.decode("utf-8-sig"))
        # Detecta separador básico
        sample = buf.getvalue()[:2000]
        delimiter = ";" if sample.count(";") > sample.count(",") else ","
        buf.seek(0)
        return pd.read_csv(buf, sep=delimiter)
    except UnicodeDecodeError:
        # Fallback latin-1
        buf = io.StringIO(content.decode("latin-1"))
        sample = buf.getvalue()[:2000]
        delimiter = ";" if sample.count(";") > sample.count(",") else ","
        buf.seek(0)
        return pd.read_csv(buf, sep=delimiter)

def fetch_resource(resource_id: str) -> pd.DataFrame:
    """
    Intenta:
      1) Datastore (datastore_search)
      2) Descarga directa del recurso (resource_show -> url)
    Devuelve un DataFrame o levanta HTTPError con info diagnosticable.
    """
    # 1) Datastore
    try:
        df = _try_datastore(resource_id)
        if df is not None:
            return df
    except requests.HTTPError as e:
        # Continuamos a método directo, pero guardamos el contexto
        ds_err = str(e)
    else:
        ds_err = None

    # 2) Descarga directa
    dl_url = _resource_download_url(resource_id)
    r = _get(dl_url, stream=True)
    content = r.content
    try:
        df = _read_csv_bytes(content)
        if df.empty and len(content) == 0:
            raise requests.HTTPError("Downloaded file is empty.")
        return df
    except Exception as e:
        raise requests.HTTPError(
            f"Failed to parse CSV from {dl_url} (res_id={resource_id}). "
            f"DatastoreError={ds_err}. ParseError={e}"
        )
