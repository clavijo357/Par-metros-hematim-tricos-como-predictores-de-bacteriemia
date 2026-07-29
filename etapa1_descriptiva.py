# -*- coding: utf-8 -*-
"""
============================================================================
ETAPA 1 - Estadística descriptiva y evaluación de normalidad
Proyecto: parámetros hematimétricos como predictores de bacteriemia
(confirmada por hemocultivo)
----------------------------------------------------------------------------
Grupos: hemocultivo NEGATIVO vs POSITIVO (pacientes independientes)
Variables:
    - Ratio neutrófilos/linfocitos (N/L)   [adimensional]
    - Ratio plaquetas/linfocitos   (P/L)   [adimensional]
    - PDW  (ancho de distribución plaquetaria)  [%]
    - Plaquetocrito                             [%]

Qué hace este script:
    1. Carga las dos planillas y las combina en un único dataset con etiqueta de grupo.
    2. Vuelve a validar la calidad (tipos, faltantes, duplicados) -- por seguridad.
    3. Calcula estadística descriptiva por grupo (n, mediana, RIC, media, DE, min, max).
    4. Evalúa normalidad por variable y grupo con Shapiro-Wilk.
    5. Genera histogramas y box plots por grupo.
    6. Exporta la tabla descriptiva a CSV y la figura a PNG.

Requisitos (versiones con las que fue probado):
    python >= 3.10
    pandas 3.0 | scipy 1.17 | matplotlib 3.10 | numpy 2.4 | openpyxl
    Instalar:  pip install pandas scipy matplotlib numpy openpyxl
============================================================================
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")          # backend sin ventana; genera archivos de imagen
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 0) CONFIGURACIÓN  --  AJUSTAR ESTAS RUTAS A TU COMPUTADORA
# ---------------------------------------------------------------------------
from pathlib import Path
CARPETA = Path(__file__).parent
ARCHIVO_NEGATIVOS = CARPETA / "negativos_para_IA__1_.xlsx"
ARCHIVO_POSITIVOS = CARPETA / "positivos_para_IA.xlsx"

COL_ID = "número_muestra"
VARIABLES = [
    "RATIO_NEUTRÓFILOS/LINFOCITOS",
    "RATIO_PLAQUETAS/LINFOCITOS",
    "PDW",
    "PLAQUETOCRITO",
]
# Etiquetas legibles (con unidades) para tablas y gráficos
ETIQUETAS = {
    "RATIO_NEUTRÓFILOS/LINFOCITOS": "Ratio N/L",
    "RATIO_PLAQUETAS/LINFOCITOS":   "Ratio P/L",
    "PDW":                          "PDW (%)",
    "PLAQUETOCRITO":                "Plaquetocrito (%)",
}
COLORES = {"Negativo": "#4C9BD1", "Positivo": "#D96459"}
ALFA = 0.05   # umbral para las pruebas de normalidad

# ---------------------------------------------------------------------------
# 1) CARGA Y COMBINACIÓN
# ---------------------------------------------------------------------------
def cargar():
    neg = pd.read_excel(ARCHIVO_NEGATIVOS)
    pos = pd.read_excel(ARCHIVO_POSITIVOS)
    neg["grupo"] = "Negativo"
    pos["grupo"] = "Positivo"
    df = pd.concat([neg, pos], ignore_index=True)
    return df

# ---------------------------------------------------------------------------
# 2) CONTROL DE CALIDAD (se vuelve a correr por seguridad)
# ---------------------------------------------------------------------------
def control_calidad(df):
    print("=" * 70)
    print("CONTROL DE CALIDAD")
    print("=" * 70)
    print("N por grupo:", df["grupo"].value_counts().to_dict())

    # tipos numéricos
    for v in VARIABLES:
        no_num = pd.to_numeric(df[v], errors="coerce").isna().sum()
        estado = "OK" if no_num == 0 else f"¡{no_num} valores NO numéricos!"
        print(f"  {ETIQUETAS[v]:<20} tipo -> {estado}")

    # faltantes
    faltantes = df[VARIABLES].isna().sum().sum()
    print(f"  Valores faltantes totales: {faltantes}")

    # duplicados de ID (dentro de cada grupo y global)
    dup = df[COL_ID].duplicated().sum()
    print(f"  IDs duplicados: {dup}")

    # solapamiento entre grupos (los pacientes deben ser distintos)
    ids_neg = set(df.loc[df.grupo == 'Negativo', COL_ID])
    ids_pos = set(df.loc[df.grupo == 'Positivo', COL_ID])
    print(f"  IDs presentes en AMBOS grupos: {ids_neg & ids_pos or 'ninguno'}")
    print()

# ---------------------------------------------------------------------------
# 3) ESTADÍSTICA DESCRIPTIVA POR GRUPO
# ---------------------------------------------------------------------------
def descriptiva(df):
    filas = []
    for v in VARIABLES:
        for g in ["Negativo", "Positivo"]:
            s = df.loc[df.grupo == g, v].astype(float)
            q1, q3 = s.quantile(.25), s.quantile(.75)
            filas.append({
                "Variable": ETIQUETAS[v],
                "Grupo": g,
                "n": s.size,
                "Mediana": round(s.median(), 3),
                "Q1": round(q1, 3),
                "Q3": round(q3, 3),
                "RIC": round(q3 - q1, 3),
                "Media": round(s.mean(), 3),
                "DE": round(s.std(), 3),
                "Min": round(s.min(), 3),
                "Max": round(s.max(), 3),
            })
    tabla = pd.DataFrame(filas)
    print("=" * 70)
    print("ESTADÍSTICA DESCRIPTIVA POR GRUPO")
    print("=" * 70)
    print(tabla.to_string(index=False))
    print()
    return tabla

# ---------------------------------------------------------------------------
# 4) NORMALIDAD  (Shapiro-Wilk por variable y grupo)
# ---------------------------------------------------------------------------
def normalidad(df):
    filas = []
    print("=" * 70)
    print("PRUEBA DE NORMALIDAD (Shapiro-Wilk)")
    print("=" * 70)
    for v in VARIABLES:
        for g in ["Negativo", "Positivo"]:
            s = df.loc[df.grupo == g, v].astype(float)
            W, p = stats.shapiro(s)
            normal = p >= ALFA
            filas.append({"Variable": ETIQUETAS[v], "Grupo": g,
                          "W": round(W, 3), "p": p,
                          "Normal": "Sí" if normal else "No"})
            print(f"  {ETIQUETAS[v]:<20} {g:<9} W={W:.3f}  "
                  f"p={'<0.001' if p < 0.001 else f'{p:.3f}':<7}  "
                  f"-> {'NORMAL' if normal else 'NO normal'}")
    print()
    res = pd.DataFrame(filas)

    # Regla de decisión para la Etapa 2
    todas_normales = (res["Normal"] == "Sí").all()
    print("-" * 70)
    if todas_normales:
        print("DECISIÓN: todas las variables son normales -> Etapa 2 con tests PARAMÉTRICOS "
              "(t de Student, Pearson).")
    else:
        print("DECISIÓN: al menos una variable NO es normal -> Etapa 2 con tests NO "
              "PARAMÉTRICOS (Mann-Whitney U, Spearman) y reporte con mediana + RIC.")
    print("-" * 70 + "\n")
    return res

# ---------------------------------------------------------------------------
# 5) GRÁFICOS  (histogramas + box plots)
# ---------------------------------------------------------------------------
def graficos(df, salida="etapa1_distribuciones.png"):
    fig, axes = plt.subplots(2, len(VARIABLES), figsize=(4.5 * len(VARIABLES), 8))
    for j, v in enumerate(VARIABLES):
        # ---- histograma (fila 0) ----
        ax = axes[0, j]
        for g in ["Negativo", "Positivo"]:
            s = df.loc[df.grupo == g, v].astype(float)
            ax.hist(s, bins=20, alpha=0.55, label=g,
                    color=COLORES[g], edgecolor="white", linewidth=0.4)
        ax.set_title(ETIQUETAS[v], fontweight="bold")
        ax.set_xlabel(ETIQUETAS[v]); ax.set_ylabel("Frecuencia")
        if j == 0:
            ax.legend()
        # ---- box plot (fila 1) ----
        ax = axes[1, j]
        data = [df.loc[df.grupo == g, v].astype(float) for g in ["Negativo", "Positivo"]]
        bp = ax.boxplot(data, tick_labels=["Neg", "Pos"], patch_artist=True,
                        widths=0.6, showmeans=True,
                        meanprops=dict(marker="D", markerfacecolor="black",
                                       markersize=5, markeredgecolor="black"))
        for patch, g in zip(bp["boxes"], ["Negativo", "Positivo"]):
            patch.set_facecolor(COLORES[g]); patch.set_alpha(0.65)
        for med in bp["medians"]:
            med.set_color("black"); med.set_linewidth(2)
        ax.set_title(f"{ETIQUETAS[v]} — box plot", fontweight="bold")
        ax.set_ylabel(ETIQUETAS[v])

    fig.suptitle("Etapa 1 — Distribuciones por grupo "
                 "(arriba: histogramas · abajo: box plots, ◇ = media)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(salida, dpi=140, bbox_inches="tight")
    print(f"Figura guardada en: {salida}\n")

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    df = cargar()
    control_calidad(df)
    tabla = descriptiva(df)
    res_norm = normalidad(df)
    graficos(df)

    # Exportar resultados
    tabla.to_csv("etapa1_descriptiva.csv", index=False, encoding="utf-8-sig")
    res_norm.to_csv("etapa1_normalidad.csv", index=False, encoding="utf-8-sig")
    print("Tablas exportadas: etapa1_descriptiva.csv | etapa1_normalidad.csv")

if __name__ == "__main__":
    main()

