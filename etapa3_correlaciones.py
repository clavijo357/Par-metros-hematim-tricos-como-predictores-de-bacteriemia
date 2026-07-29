# -*- coding: utf-8 -*-
"""
============================================================================
ETAPA 3 - Correlaciones entre variables (detección de colinealidad)
Proyecto: parámetros hematimétricos como predictores de bacteriemia
(confirmada por hemocultivo)
----------------------------------------------------------------------------
OBJETIVO (distinto al de la Etapa 2):
    La Etapa 2 preguntaba si cada variable se relaciona con LA BACTERIEMIA.
    La Etapa 3 pregunta si las variables se relacionan ENTRE SÍ.
    El fin es detectar COLINEALIDAD -- variables que miden casi lo mismo --
    ANTES de armar la regresión logística (Etapa 4). Dos predictoras muy
    correlacionadas entre sí desestabilizan el modelo: inflan los odds ratios,
    les cambian el signo y agrandan los intervalos de confianza.

MÉTODO:
    Correlación de Spearman (rho), coherente con la no-normalidad detectada en
    la Etapa 1. Spearman trabaja con rangos, por lo que no le afectan las colas
    largas ni los valores extremos de los ratios.

    El cálculo se hace sobre TODA la muestra (negativos + positivos juntos, n=190),
    que es lo correcto para este fin: el modelo de la Etapa 4 también verá a
    todos los pacientes juntos.

INTERPRETACIÓN DE rho (rango -1 a +1):
     0         -> sin relación monótona (ideal para el modelo)
    cerca +1   -> suben juntas | cerca -1 -> una sube cuando la otra baja
    |rho| > ~0.7-0.8 -> colinealidad preocupante (habría que descartar/combinar)
    |rho| 0.5-0.7    -> moderada (tolerable, vigilar)
    |rho| < 0.5      -> baja (sin problema)

Requisitos (versiones probadas):
    python >= 3.10
    pandas 3.0 | scipy 1.17 | matplotlib 3.10 | numpy 2.4 | openpyxl
    Instalar:  pip install pandas scipy matplotlib numpy openpyxl
    (No requiere scikit-learn.)
============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 0) CONFIGURACIÓN  (busca las planillas en la misma carpeta que este script)
# ---------------------------------------------------------------------------
CARPETA = Path(__file__).parent
ARCHIVO_NEGATIVOS = CARPETA / "negativos_para_IA__1_.xlsx"
ARCHIVO_POSITIVOS = CARPETA / "positivos_para_IA.xlsx"

VARIABLES = [
    "RATIO_NEUTRÓFILOS/LINFOCITOS",
    "RATIO_PLAQUETAS/LINFOCITOS",
    "PDW",
    "PLAQUETOCRITO",
]
ETIQUETAS = ["Ratio N/L", "Ratio P/L", "PDW (%)", "Plaquetocrito (%)"]

UMBRAL_ALTO = 0.7   # colinealidad preocupante
UMBRAL_MOD  = 0.5   # correlación moderada

# ---------------------------------------------------------------------------
# 1) CARGA (toda la muestra junta)
# ---------------------------------------------------------------------------
def cargar():
    neg = pd.read_excel(ARCHIVO_NEGATIVOS)
    pos = pd.read_excel(ARCHIVO_POSITIVOS)
    df = pd.concat([neg, pos], ignore_index=True)
    return df

# ---------------------------------------------------------------------------
# 2) MATRIZ DE CORRELACIÓN DE SPEARMAN
# ---------------------------------------------------------------------------
def matriz_spearman(df):
    X = df[VARIABLES].astype(float)
    rho_arr, p_arr = spearmanr(X)
    rho = pd.DataFrame(rho_arr, index=ETIQUETAS, columns=ETIQUETAS)
    pval = pd.DataFrame(p_arr, index=ETIQUETAS, columns=ETIQUETAS)
    return rho, pval

# ---------------------------------------------------------------------------
# 3) IMPRESIÓN + LISTA DE PARES ORDENADA POR |rho|
# ---------------------------------------------------------------------------
def informar(rho, pval):
    print("=" * 70)
    print(f"ETAPA 3 — Matriz de correlación de Spearman (rho)  [n = toda la muestra]")
    print("=" * 70)
    print(rho.round(3).to_string())
    print("\np-valores:")
    print(pval.map(lambda p: "<0.001" if p < 0.001 else f"{p:.3f}").to_string())

    print("\n" + "-" * 70)
    print("PARES ORDENADOS POR |rho| (de mayor a menor):")
    print("-" * 70)
    pares = []
    for i in range(len(ETIQUETAS)):
        for j in range(i + 1, len(ETIQUETAS)):
            pares.append((ETIQUETAS[i], ETIQUETAS[j], rho.iloc[i, j], pval.iloc[i, j]))
    pares.sort(key=lambda x: -abs(x[2]))

    hay_colinealidad = False
    for a, b, r, p in pares:
        if abs(r) > UMBRAL_ALTO:
            flag = "  <-- COLINEALIDAD ALTA (descartar o combinar una)"
            hay_colinealidad = True
        elif abs(r) > UMBRAL_MOD:
            flag = "  <- moderada (tolerable)"
        else:
            flag = ""
        pp = "<0.001" if p < 0.001 else f"{p:.3f}"
        print(f"  {a:<16} <-> {b:<18} rho = {r:+.3f}   p = {pp}{flag}")

    print("-" * 70)
    if hay_colinealidad:
        print("CONCLUSIÓN: hay al menos un par con colinealidad alta. Antes de la "
              "Etapa 4,\nconviene conservar UNA de las dos variables del par "
              "(la más interpretable o\nla de mejor AUC en la Etapa 2) y descartar "
              "la otra, o combinarlas.")
    else:
        print("CONCLUSIÓN: ningún par supera |rho| = 0.7. No hay colinealidad grave; "
              "las\ncuatro variables pueden entrar juntas a la regresión logística "
              "(Etapa 4)\nsin riesgo de inestabilidad por redundancia.")
    print("-" * 70 + "\n")

# ---------------------------------------------------------------------------
# 4) HEATMAP
# ---------------------------------------------------------------------------
def heatmap(rho, salida=None):
    if salida is None:
        salida = CARPETA / "etapa3_heatmap_spearman.png"
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(rho.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(len(ETIQUETAS))); ax.set_yticks(range(len(ETIQUETAS)))
    ax.set_xticklabels(ETIQUETAS, rotation=30, ha="right")
    ax.set_yticklabels(ETIQUETAS)
    for i in range(len(ETIQUETAS)):
        for j in range(len(ETIQUETAS)):
            val = rho.values[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    color=color, fontsize=11,
                    fontweight="bold" if i != j else "normal")
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("rho de Spearman", fontsize=10)
    ax.set_title("Etapa 3 — Matriz de correlación de Spearman\nentre variables",
                 fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.savefig(salida, dpi=140, bbox_inches="tight")
    print(f"Figura guardada en: {salida}\n")

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    df = cargar()
    rho, pval = matriz_spearman(df)
    informar(rho, pval)
    heatmap(rho)

    # Exportar la matriz de rho y la de p-valores
    salida_csv = CARPETA / "etapa3_spearman_rho.csv"
    rho.to_csv(salida_csv, encoding="utf-8-sig")
    pval.to_csv(CARPETA / "etapa3_spearman_pvalores.csv", encoding="utf-8-sig")
    print(f"Tablas exportadas: {salida_csv.name} | etapa3_spearman_pvalores.csv")

if __name__ == "__main__":
    main()
