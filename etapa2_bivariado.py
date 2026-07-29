# -*- coding: utf-8 -*-
"""
============================================================================
ETAPA 2 - Análisis bivariado y capacidad discriminatoria
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
        b. Tamaño de efecto: r de rango-biserial (rango -1 a +1).
        c. Área bajo la curva ROC (AUC) con su IC 95% por el método de DeLong,
           como medida de discriminación. El IC coincide con el que reporta SPSS.
    3. Corrige los p-valores por comparaciones múltiples con
       Benjamini-Hochberg (control de la tasa de falsos descubrimientos, FDR).
    4. Grafica las curvas ROC de las 4 variables en un mismo panel.
    5. Exporta la tabla de resultados a CSV y la figura a PNG.

NOTA IMPORTANTE SOBRE EL PLAQUETOCRITO (dirección invertida):
    Para las demás variables, valores MÁS ALTOS se asocian a bacteriemia.
    Para el plaquetocrito ocurre lo contrario: valores MÁS BAJOS se asocian
    a bacteriemia. Por eso, SOLO para el cálculo del AUC de esa variable, se
    invierte la dirección del score (se usa el valor negativo). Si no se
    hiciera, su AUC daría < 0.5 y parecería un mal marcador cuando en realidad
    discrimina en sentido inverso. La prueba de Mann-Whitney y el r de
    rango-biserial NO necesitan esta inversión: son de dos colas y su
    interpretación ya contempla el signo.

Requisitos (versiones probadas):
    python >= 3.10
    pandas 3.0 | scipy 1.17 | scikit-learn | matplotlib 3.10 | numpy 2.4 | openpyxl
    Instalar:  pip install pandas scipy scikit-learn matplotlib numpy openpyxl
============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu, false_discovery_control
from scipy import stats
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
# 1b) IC 95% del AUC por el método de DeLong
#     DeLong estima la varianza del AUC a partir de sus "componentes
#     estructurales" (cuánto aporta cada paciente al AUC) y construye el IC de
#     forma analítica -- sin remuestreo, por lo que es determinístico (mismo
#     resultado siempre). Es el método que usa SPSS por defecto, de modo que
#     los IC coinciden con los de esa herramienta.
#     Entrada: y_true (0/1) y scores (predictor continuo, ya orientado de modo
#     que valores más altos predigan el evento=1).
# ---------------------------------------------------------------------------
def delong_auc_ci(y_true, scores, alpha=0.95):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    P = scores[y_true == 1]      # scores de los positivos
    N = scores[y_true == 0]      # scores de los negativos
    m, n = len(P), len(N)

    def midrank(x):
        # rango medio (promedia empates), necesario para datos con valores repetidos
        J = np.argsort(x); Z = x[J]; L = len(x); T = np.zeros(L); i = 0
        while i < L:
            j = i
            while j < L and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5 * (i + j - 1) + 1
            i = j
        out = np.empty(L); out[J] = T
        return out

    tx = midrank(P)
    ty = midrank(N)
    tz = midrank(np.concatenate([P, N]))
    auc = (np.sum(tz[:m]) - m * (m + 1) / 2) / (m * n)
    v10 = (tz[:m] - tx) / n                 # componente de los positivos
    v01 = 1 - (tz[m:] - ty) / m             # componente de los negativos
    var = np.var(v10, ddof=1) / m + np.var(v01, ddof=1) / n
    se = np.sqrt(var)
    z = stats.norm.ppf(1 - (1 - alpha) / 2)
    lo, hi = auc - z * se, auc + z * se
    p = 2 * (1 - stats.norm.cdf(abs((auc - 0.5) / se)))   # H0: AUC = 0.5 (azar)
    return auc, max(0.0, lo), min(1.0, hi), se, p

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

        # b) r de rango-biserial = 2*U/(n1*n2) - 1
        #    +1: positivos siempre mayores | -1: positivos siempre menores | 0: sin separación
        r_rb = 2 * U / (n_pos * n_neg) - 1

        # c) AUC + IC 95% (DeLong). Se orienta el score según la dirección
        #    clínica: para las variables donde "menor predice" (plaquetocrito)
        #    se usa el score negado, de modo que el AUC quede > 0.5 y el IC se
        #    calcule sobre la dirección correcta.
        score = df[v].astype(float).values
        if not MAYOR_PREDICE[v]:
            score = -score
        auc, auc_lo, auc_hi, auc_se, auc_p = delong_auc_ci(df["grupo"].values, score)

        filas.append({
            "Variable": ETIQUETAS[v],
            "n_neg": n_neg, "n_pos": n_pos,
            "U": U,
            "p_crudo": p,
            "r_rango_biserial": round(r_rb, 3),
            "AUC": round(auc, 3),
            "AUC_IC95_inf": round(auc_lo, 3),
            "AUC_IC95_sup": round(auc_hi, 3),
            "AUC_p_vs_0.5": auc_p,
            "direccion_invertida": not MAYOR_PREDICE[v],
        })

    res = pd.DataFrame(filas)

    # 3) Corrección Benjamini-Hochberg (FDR) sobre los 4 p-valores
    res["p_BH"] = false_discovery_control(res["p_crudo"], method="bh")
    res["signif_BH"] = np.where(res["p_BH"] < ALFA, "Sí", "No")

    # Reordenar columnas para lectura
    res = res[["Variable", "n_neg", "n_pos", "U",
               "p_crudo", "p_BH", "signif_BH",
               "r_rango_biserial",
               "AUC", "AUC_IC95_inf", "AUC_IC95_sup", "AUC_p_vs_0.5",
               "direccion_invertida"]]
    return res

# ---------------------------------------------------------------------------
# 3) IMPRESIÓN LEGIBLE
# ---------------------------------------------------------------------------
def imprimir(res):
    print("=" * 78)
    print("ETAPA 2 — Análisis bivariado (Mann-Whitney U) + tamaño de efecto + AUC")
    print("=" * 78)
    for _, r in res.iterrows():
        pc = "<0.001" if r.p_crudo < 0.001 else f"{r.p_crudo:.4f}"
        pb = "<0.001" if r.p_BH < 0.001 else f"{r.p_BH:.4f}"
        inv = "  [dirección invertida: menor -> bacteriemia]" if r.direccion_invertida else ""
        print(f"\n{r.Variable}{inv}")
        print(f"   U = {r.U:.0f}")
        print(f"   p crudo = {pc}    p BH = {pb}    -> significativa tras BH: {r.signif_BH}")
        auc_p = "<0.001" if r["AUC_p_vs_0.5"] < 0.001 else f"{r['AUC_p_vs_0.5']:.4f}"
        print(f"   r rango-biserial = {r.r_rango_biserial:+.3f}")
        print(f"   AUC = {r.AUC:.3f}  IC95% ({r['AUC_IC95_inf']:.3f}-{r['AUC_IC95_sup']:.3f})  "
              f"p vs 0.5 = {auc_p}")
    print("\n" + "-" * 78)
    print("Guía de interpretación:")
    print("  r rango-biserial: ~0.1 pequeño | ~0.3 mediano | ~0.5 grande (referencias)")
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

    salida_csv = CARPETA / "etapa2_bivariado.csv"
    res.to_csv(salida_csv, index=False, encoding="utf-8-sig")
    print(f"Tabla exportada: {salida_csv}")

if __name__ == "__main__":
    main()
