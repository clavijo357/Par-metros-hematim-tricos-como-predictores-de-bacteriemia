# -*- coding: utf-8 -*-
"""
============================================================================
ETAPA 2 - Análisis bivariado
Proyecto: parámetros hematimétricos como predictores de bacteriemia
(confirmada por hemocultivo)
----------------------------------------------------------------------------
Grupos: hemocultivo NEGATIVO vs POSITIVO (pacientes independientes)
Variables:
    - Ratio neutrófilos/linfocitos (N/L)   [adimensional]
    - Ratio plaquetas/linfocitos   (P/L)   [adimensional]
    - PDW  (ancho de distribución plaquetaria)  [%]
    - Plaquetocrito                             [%]


Justificación de los métodos elegidos:
    En la Etapa 1, Shapiro-Wilk rechazó la normalidad en la mayoría de las
    variables -> se usan pruebas NO paramétricas.

Qué hace este script:
    1. Carga las dos planillas y las combina con etiqueta de grupo.
    2. Para cada variable:
        a. Prueba U de Mann-Whitney (comparación entre grupos, dos colas).
        b. Área bajo la curva ROC (AUC), como medida de discriminación.
    3. Grafica las curvas ROC de las 4 variables en un mismo panel.
    4. Exporta la tabla de resultados a CSV y la figura a PNG.

NOTA IMPORTANTE SOBRE EL PLAQUETOCRITO (dirección invertida):
    Para las demás variables, valores MÁS ALTOS se asocian a bacteriemia.
    Para el plaquetocrito ocurre lo contrario: valores MÁS BAJOS se asocian
    a bacteriemia. Por eso, SOLO para el cálculo del AUC de esa variable, se
    invierte la dirección del score (se usa el valor negativo). Si no se
    hiciera, su AUC daría < 0.5 y parecería un mal marcador cuando en realidad
    discrimina en sentido inverso. La prueba de Mann-Whitney NO necesita esta
    inversión: es de dos colas y su interpretación ya contempla el signo.

Requisitos (versiones probadas):
    python >= 3.10
    pandas 3.0 | scipy 1.17 | scikit-learn | matplotlib 3.10 | numpy 2.4 | openpyxl
    Instalar:  pip install pandas scipy scikit-learn matplotlib numpy openpyxl
============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_curve, auc as sk_auc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 0) CONFIGURACIÓN
#    El script busca las planillas en su MISMA carpeta. Mantené los 3 archivos
#    juntos (este .py + las 2 planillas) y no necesitás tocar nada.
# ---------------------------------------------------------------------------
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
ETIQUETAS = {
    "RATIO_NEUTRÓFILOS/LINFOCITOS": "Ratio N/L",
    "RATIO_PLAQUETAS/LINFOCITOS":   "Ratio P/L",
    "PDW":                          "PDW (%)",
    "PLAQUETOCRITO":                "Plaquetocrito (%)",
}
# Dirección clínica esperada:
#   True  = valores MÁS ALTOS predicen bacteriemia
#   False = valores MÁS BAJOS predicen bacteriemia (plaquetocrito)
MAYOR_PREDICE = {
    "RATIO_NEUTRÓFILOS/LINFOCITOS": True,
    "RATIO_PLAQUETAS/LINFOCITOS":   True,
    "PDW":                          True,
    "PLAQUETOCRITO":                False,
}
COLORES = {
    "RATIO_NEUTRÓFILOS/LINFOCITOS": "#E4A11B",
    "RATIO_PLAQUETAS/LINFOCITOS":   "#3B7A57",
    "PDW":                          "#D96459",
    "PLAQUETOCRITO":                "#4C9BD1",
}
ALFA = 0.05

# ---------------------------------------------------------------------------
# 1) CARGA
# ---------------------------------------------------------------------------
def cargar():
    neg = pd.read_excel(ARCHIVO_NEGATIVOS)
    pos = pd.read_excel(ARCHIVO_POSITIVOS)
    neg["grupo"] = 0   # 0 = hemocultivo negativo
    pos["grupo"] = 1   # 1 = hemocultivo positivo
    return pd.concat([neg, pos], ignore_index=True)

# ---------------------------------------------------------------------------
# 2) ANÁLISIS BIVARIADO POR VARIABLE
# ---------------------------------------------------------------------------
def bivariado(df):
    filas = []
    for v in VARIABLES:
        neg_vals = df.loc[df.grupo == 0, v].astype(float).values
        pos_vals = df.loc[df.grupo == 1, v].astype(float).values
        n_neg, n_pos = len(neg_vals), len(pos_vals)

        # a) Mann-Whitney U (dos colas). Se pasa (positivos, negativos) para
        #    que el signo del efecto quede referido al grupo positivo.
        U, p = mannwhitneyu(pos_vals, neg_vals, alternative="two-sided")

        # b) AUC puntual. AUC = U/(n1*n2) = P(positivo > negativo).
        #    Si la dirección clínica es "menor predice", se invierte (1 - AUC).
        auc = U / (n_pos * n_neg)
        if not MAYOR_PREDICE[v]:
            auc = 1 - auc

        filas.append({
            "Variable": ETIQUETAS[v],
            "n_neg": n_neg, "n_pos": n_pos,
            "U": U,
            "p_crudo": p,
            "signif": "Sí" if p < ALFA else "No",
            "AUC": round(auc, 3),
            "direccion_invertida": not MAYOR_PREDICE[v],
        })

    res = pd.DataFrame(filas)
    res = res[["Variable", "n_neg", "n_pos", "U",
               "p_crudo", "signif", "AUC", "direccion_invertida"]]
    return res

# ---------------------------------------------------------------------------
# 3) IMPRESIÓN LEGIBLE
# ---------------------------------------------------------------------------
def imprimir(res):
    print("=" * 78)
    print("ETAPA 2 — Análisis bivariado: Mann-Whitney U + AUC")
    print("=" * 78)
    print
    for _, r in res.iterrows():
        pc = "<0.001" if r.p_crudo < 0.001 else f"{r.p_crudo:.4f}"
        inv = "  [dirección invertida: menor -> bacteriemia]" if r.direccion_invertida else ""
        print(f"\n{r.Variable}{inv}")
        print(f"   U = {r.U:.0f}")
        print(f"   p crudo = {pc}    -> significativa (p<{ALFA}): {r.signif}")
        print(f"   AUC = {r.AUC:.3f}")
    print("\n" + "-" * 78)
    print("Guía de interpretación:")
    print("  AUC: 0.5 = azar | 0.7 = aceptable | 0.8 = bueno | 0.9 = excelente")
    print("-" * 78 + "\n")

# ---------------------------------------------------------------------------
# 4) CURVAS ROC
# ---------------------------------------------------------------------------
def curvas_roc(df, salida=None):
    if salida is None:
        salida = CARPETA / "etapa2_curvas_ROC.png"
    y = df["grupo"].values
    plt.figure(figsize=(7.5, 7.5))
    for v in VARIABLES:
        score = df[v].astype(float).values
        if not MAYOR_PREDICE[v]:
            score = -score   # invertir score para plaquetocrito
        fpr, tpr, _ = roc_curve(y, score)
        a = sk_auc(fpr, tpr)
        etiqueta = ETIQUETAS[v] + (" [invertido]" if not MAYOR_PREDICE[v] else "")
        plt.plot(fpr, tpr, lw=2.2, color=COLORES[v], label=f"{etiqueta} — AUC={a:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Azar (AUC=0.500)")
    plt.xlabel("1 − Especificidad (Tasa de falsos positivos)", fontsize=11)
    plt.ylabel("Sensibilidad (Tasa de verdaderos positivos)", fontsize=11)
    plt.title("Etapa 2 — Curvas ROC por variable\n"
              "(predicción de bacteriemia confirmada por hemocultivo)",
              fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10, framealpha=0.95)
    plt.grid(alpha=0.25)
    plt.xlim(-0.01, 1.01); plt.ylim(-0.01, 1.01)
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.savefig(salida, dpi=140, bbox_inches="tight")
    print(f"Figura guardada en: {salida}\n")

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    df = cargar()
    res = bivariado(df)
    imprimir(res)
    curvas_roc(df)

    salida_csv = CARPETA / "etapa2_bivariado_preliminar.csv"
    res.to_csv(salida_csv, index=False, encoding="utf-8-sig")
    print(f"Tabla exportada: {salida_csv.name}")

if __name__ == "__main__":
    main()
