# utils/cnmc_ckan.py
import requests
import pandas as pd
from io import StringIO

BASE_URL = "https://datos.cnmc.gob.es/api/3/action/datastore_search?resource_id={}"

def fetch_resource(resource_id: str, limit: int = None) -> pd.DataFrame:
    """
    Descarga un recurso de la CNMC Open Data API (CKAN) y lo devuelve como DataFrame.
    - resource_id: identificador del recurso en CKAN
    - limit: número máximo de registros (por defecto, todo el dataset)
    """
    url = BASE_URL.format(resource_id)
    if limit:
        url += f"&limit={limit}"

    response = requests.get(url)
    response.raise_for_status()  # Lanza error si la API devuelve un código != 200

    data = response.json()
    records = data.get("result", {}).get("records", [])

    if not records:
        raise ValueError(f"No se obtuvieron registros del recurso {resource_id}")

    df = pd.DataFrame.from_records(records)
    return df

def download_csv(resource_id: str, path: str) -> str:
    """
    Descarga el CSV bruto desde la API de la CNMC y lo guarda en disco.
    """
    url = f"https://datos.cnmc.gob.es/datastore/dump/{resource_id}?bom=True"
    response = requests.get(url)
    response.raise_for_status()

    with open(path, "wb") as f:
        f.write(response.content)

    return path
