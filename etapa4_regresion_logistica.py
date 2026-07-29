# -*- coding: utf-8 -*-
"""
============================================================================
ETAPA 4 - Análisis multivariado (regresión logística)
Proyecto: parámetros hematimétricos como predictores de bacteriemia
(confirmada por hemocultivo)
----------------------------------------------------------------------------
OBJETIVO:
    Estimar la asociación INDEPENDIENTE de cada parámetro con la bacteriemia
    (ajustando por los demás) y evaluar si COMBINAR variables mejora la
    discriminación respecto del mejor predictor individual (PDW, Etapa 2).

DISEÑO (decidido en el chat):
    - Modelo COMPLETO: las 4 variables juntas.
        Justificación: la Etapa 3 descartó colinealidad (ningún |rho|>0.7) y
        el tamaño de muestra lo permite. Eventos (positivos)=90; con 4
        variables el EPV = 90/4 = 22.5, muy por encima del mínimo recomendado
        de 10 eventos por variable. Incluir todas evita el sesgo de
        preselección univariada.
    - Modelos de SENSIBILIDAD (para mostrar que la conclusión es estable):
        * PDW + Ratio N/L  (las de p crudo < 0.10 en la Etapa 2)
        * PDW solo          (referencia mínima)

INTERPRETACIÓN DEL ODDS RATIO (OR):
    OR = 1  -> la variable no cambia las chances de bacteriemia
    OR > 1  -> a mayor valor de la variable, MÁS chances de bacteriemia
    OR < 1  -> a mayor valor, MENOS chances (efecto "protector")
    IC95% que CRUZA el 1  -> efecto NO significativo.

    PLAQUETOCRITO: por su dirección clínica invertida (menor -> más
    bacteriemia), es ESPERABLE que su OR sea < 1. No se invierte nada a mano:
    la regresión lo detecta sola y le asigna un coeficiente negativo.

    OR ESTANDARIZADO (OR_std): como las variables están en escalas muy
    distintas (PDW ~19 vs Ratio P/L ~200), se reporta también el OR por cada
    aumento de 1 desviación estándar, que SÍ es comparable entre variables
    para ver cuál "pesa" más.

MÉTRICAS DEL MODELO:
    - AUC: capacidad de discriminación (0.5 azar, 0.7 aceptable, 0.8 buena).
    - Pseudo-R2 de McFadden: proporción de "información" explicada
      (valores 0.2-0.4 ya indican muy buen ajuste en este índice).

Requisitos (versiones probadas):
    python >= 3.10
    pandas 3.0 | numpy 2.4 | scipy 1.17 | statsmodels 0.14 |
    scikit-learn | matplotlib 3.10 | openpyxl
    Instalar:
      pip install pandas numpy scipy statsmodels scikit-learn matplotlib openpyxl
============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 0) CONFIGURACIÓN  (busca las planillas en la misma carpeta que este script)
# ---------------------------------------------------------------------------
CARPETA = Path(__file__).parent
ARCHIVO_NEGATIVOS = CARPETA / "negativos_para_IA__1_.xlsx"
ARCHIVO_POSITIVOS = CARPETA / "positivos_para_IA.xlsx"

ETIQUETAS = {
    "RATIO_NEUTRÓFILOS/LINFOCITOS": "Ratio N/L",
    "RATIO_PLAQUETAS/LINFOCITOS":   "Ratio P/L",
    "PDW":                          "PDW (%)",
    "PLAQUETOCRITO":                "Plaquetocrito (%)",
}
TODAS = list(ETIQUETAS.keys())

# ---------------------------------------------------------------------------
# 1) CARGA
# ---------------------------------------------------------------------------
def cargar():
    neg = pd.read_excel(ARCHIVO_NEGATIVOS); neg["grupo"] = 0  # 0 = negativo
    pos = pd.read_excel(ARCHIVO_POSITIVOS); pos["grupo"] = 1  # 1 = positivo
    return pd.concat([neg, pos], ignore_index=True)

# ---------------------------------------------------------------------------
# 2) AJUSTE DE UN MODELO
#    Devuelve el modelo, el AUC y una tabla ordenada de OR / IC / p / OR_std.
# ---------------------------------------------------------------------------
def ajustar(df, cols, titulo):
    y = df["grupo"].values
    X = df[cols].astype(float).copy()

    # --- modelo en escala original (OR interpretables por unidad real) ---
    Xc = sm.add_constant(X)
    modelo = sm.Logit(y, Xc).fit(disp=0)

    # --- modelo estandarizado (OR comparables entre variables) ---
    Xz = (X - X.mean()) / X.std()
    modelo_z = sm.Logit(y, sm.add_constant(Xz)).fit(disp=0)

    auc = roc_auc_score(y, modelo.predict(Xc))

    filas = []
    for c in cols:
        OR = np.exp(modelo.params[c])
        lo, hi = np.exp(modelo.conf_int().loc[c])
        p = modelo.pvalues[c]
        OR_std = np.exp(modelo_z.params[c])
        filas.append({
            "Variable": ETIQUETAS[c],
            "OR": round(OR, 3),
            "IC95%_inf": round(lo, 3),
            "IC95%_sup": round(hi, 3),
            "p": p,
            "OR_estand": round(OR_std, 3),
            "signif": "Sí" if p < 0.05 else "No",
        })
    tabla = pd.DataFrame(filas)

    # --- impresión ---
    print("=" * 84)
    print(titulo)
    print("=" * 84)
    print(f"{'Variable':<20}{'OR':>9}{'IC95%':>22}{'p':>10}{'OR_std':>10}")
    for _, r in tabla.iterrows():
        pp = "<0.001" if r.p < 0.001 else f"{r.p:.3f}"
        sig = " *" if r.signif == "Sí" else ""
        ic = f"{r['IC95%_inf']:.3f}-{r['IC95%_sup']:.3f}"
        print(f"{r.Variable:<20}{r.OR:>9.3f}{ic:>22}{pp:>10}{r.OR_estand:>10.3f}{sig}")
    print(f"\n  AUC = {auc:.3f}   |   Pseudo-R2 (McFadden) = {modelo.prsquared:.3f}")
    print(f"  n = {int(modelo.nobs)}   eventos(positivos) = {int(y.sum())}   "
          f"EPV = {y.sum()/len(cols):.1f}\n")

    return modelo, auc, tabla

# ---------------------------------------------------------------------------
# 3) CURVA ROC COMPARATIVA DE LOS MODELOS
# ---------------------------------------------------------------------------
def roc_comparativa(df, definiciones, salida=None):
    if salida is None:
        salida = CARPETA / "etapa4_ROC_modelos.png"
    y = df["grupo"].values
    plt.figure(figsize=(7.5, 7.5))
    for etiqueta, (cols, color) in definiciones.items():
        Xc = sm.add_constant(df[cols].astype(float))
        pred = sm.Logit(y, Xc).fit(disp=0).predict(Xc)
        fpr, tpr, _ = roc_curve(y, pred)
        a = roc_auc_score(y, pred)
        plt.plot(fpr, tpr, lw=2.3, color=color, label=f"{etiqueta} (AUC={a:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Azar (0.500)")
    plt.xlabel("1 − Especificidad", fontsize=11)
    plt.ylabel("Sensibilidad", fontsize=11)
    plt.title("Etapa 4 — Curvas ROC de los modelos de regresión logística",
              fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10, framealpha=0.95)
    plt.grid(alpha=0.25); plt.xlim(-0.01, 1.01); plt.ylim(-0.01, 1.01)
    plt.gca().set_aspect("equal"); plt.tight_layout()
    plt.savefig(salida, dpi=140, bbox_inches="tight")
    print(f"Figura guardada en: {salida}\n")

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    df = cargar()

    # Modelo completo (principal)
    _, auc_full, tabla_full = ajustar(
        df, TODAS, "MODELO COMPLETO — las 4 variables (principal)")

    # Sensibilidad 1: PDW + Ratio N/L
    _, auc_red2, _ = ajustar(
        df, ["PDW", "RATIO_NEUTRÓFILOS/LINFOCITOS"],
        "MODELO DE SENSIBILIDAD — PDW + Ratio N/L")

    # Referencia mínima: PDW solo
    _, auc_min, _ = ajustar(
        df, ["PDW"], "MODELO MÍNIMO (referencia) — solo PDW")

    # Comparación de AUC
    print("=" * 84)
    print("COMPARACIÓN DE DISCRIMINACIÓN (AUC)")
    print("=" * 84)
    print(f"  PDW solo:                 {auc_min:.3f}")
    print(f"  PDW + Ratio N/L:          {auc_red2:.3f}")
    print(f"  Modelo completo (4 vars): {auc_full:.3f}")
    print("  -> Si el completo no supera claramente al mínimo, combinar variables")
    print("     no aporta discriminación relevante sobre el PDW solo.\n")

    # Gráfico comparativo
    roc_comparativa(df, {
        "PDW solo":           (["PDW"], "#4C9BD1"),
        "PDW + Ratio N/L":    (["PDW", "RATIO_NEUTRÓFILOS/LINFOCITOS"], "#3B7A57"),
        "Completo (4 vars)":  (TODAS, "#D96459"),
    })

    # Exportar la tabla del modelo principal
    salida_csv = CARPETA / "etapa4_modelo_completo.csv"
    tabla_full.to_csv(salida_csv, index=False, encoding="utf-8-sig")
    print(f"Tabla exportada: {salida_csv.name}")

if __name__ == "__main__":
    main()
