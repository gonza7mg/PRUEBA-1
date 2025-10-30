import time, requests, pandas as pd
CKAN_BASE = "https://catalogodatos.cnmc.es/api/3/action/datastore_search"
UA = {"User-Agent": "DSS-Telecom/1.0"}

def fetch_resource(resource_id: str, limit: int = 50000, sleep_sec: float = 0.25) -> pd.DataFrame:
    params = {"resource_id": resource_id, "limit": limit, "offset": 0}
    r = requests.get(CKAN_BASE, params=params, headers=UA, timeout=60)
    r.raise_for_status()
    res = r.json()["result"]
    total = res.get("total", len(res.get("records", [])))
    records = res.get("records", [])
    while len(records) < total:
        params["offset"] += limit
        time.sleep(sleep_sec)
        r = requests.get(CKAN_BASE, params=params, headers=UA, timeout=60)
        r.raise_for_status()
        recs = r.json()["result"].get("records", [])
        if not recs:
            break
        records.extend(recs)
    return pd.DataFrame.from_records(records)
