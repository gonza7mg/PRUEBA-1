from .clean_anual_general import clean_anual_general
from .clean_anual_mercados import clean_anual_mercados
from .clean_mensual import clean_mensual
from .clean_provinciales import clean_provinciales
from .clean_trimestrales import clean_trimestrales
from .clean_infraestructuras import clean_infraestructuras

CLEANERS = {
    "anual_datos_generales": clean_anual_general,
    "anual_mercados": clean_anual_mercados,
    "mensual": clean_mensual,
    "provinciales": clean_provinciales,
    "trimestrales": clean_trimestrales,
    "infraestructuras": clean_infraestructuras,
}
