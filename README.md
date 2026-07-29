# Par-metros-hematim-tricos-como-predictores-de-bacteriemia
Pipeline de análisis estadístico.
# Parámetros hematimétricos como predictores de bacteriemia

Código de análisis estadístico del estudio retrospectivo que evalúa la asociación
entre parámetros hematimétricos y el desarrollo de bacteriemia confirmada por
hemocultivo, en pacientes del Hospital de Clínicas "Dr. Manuel Quintela"
(período 2025–2026).

## Objetivo

Evaluar si cuatro parámetros hematimétricos —ratio neutrófilos/linfocitos (N/L),
ratio plaquetas/linfocitos (P/L), ancho de distribución plaquetaria (PDW) y
plaquetocrito— se asocian con la bacteriemia y con qué capacidad discriminatoria,
de forma individual y combinada.

## Datos

El análisis compara dos grupos independientes de pacientes:

- **Negativos** (n = 100): hemocultivo sin desarrollo de bacteriemia.
- **Positivos** (n = 90): hemocultivo con desarrollo de un patógeno primario.

Los scripts esperan dos planillas de Excel con las columnas: `número_muestra`,
`RATIO_NEUTRÓFILOS/LINFOCITOS`, `RATIO_PLAQUETAS/LINFOCITOS`, `PDW`,
`PLAQUETOCRITO`.

> **Nota sobre los datos:** por razones de confidencialidad de los pacientes,
> las planillas originales no se incluyen en este repositorio. Para reproducir el
> análisis con datos propios, deben respetarse los nombres de columna indicados
> arriba.

## Estructura del análisis

El análisis se organiza en cuatro etapas, cada una en su propio script:

| Script | Etapa | Contenido |
|---|---|---|
| `etapa1_descriptiva.py` | 1 — Estadística descriptiva | Mediana, RIC, media y DE por grupo; prueba de normalidad de Shapiro-Wilk; histogramas y box plots. |
| `etapa2_bivariado.py` | 2 — Análisis bivariado | Prueba U de Mann-Whitney; tamaño de efecto (r de rango-biserial); AUC con IC 95% por DeLong; corrección de Benjamini-Hochberg; curvas ROC. |
| `etapa3_correlaciones.py` | 3 — Correlaciones | Matriz de correlación de Spearman entre variables (detección de colinealidad); heatmap. |
| `etapa4_regresion_logistica.py` | 4 — Análisis multivariado | Regresión logística (modelo completo y modelos de sensibilidad); odds ratios crudos y estandarizados con IC 95%; comparación de AUC entre modelos. |

Cada script incluye en su encabezado la justificación metodológica de las
decisiones tomadas (por qué pruebas no paramétricas, por qué se invierte la
dirección del plaquetocrito para el AUC, criterio de colinealidad, cálculo del
EPV, etc.).

## Cómo reproducir el análisis

1. Instalar Python 3.10 o superior.
2. Instalar las dependencias:
   ```
   pip install -r requirements.txt
   ```
3. Colocar las dos planillas de Excel en la misma carpeta que los scripts, con
   los nombres esperados (`negativos_para_IA__1_.xlsx` y `positivos_para_IA.xlsx`),
   o editar las rutas en la sección `CONFIGURACIÓN` de cada script.
4. Ejecutar cada etapa en orden:
   ```
   python etapa1_descriptiva.py
   python etapa2_bivariado.py
   python etapa3_correlaciones.py
   python etapa4_regresion_logistica.py
   ```

Cada script imprime los resultados en pantalla y guarda las tablas (CSV) y
figuras (PNG) en la misma carpeta.

## Entorno y versiones

Las versiones utilizadas se detallan en `requirements.txt`. El análisis fue
desarrollado en Python 3.14. Los métodos son determinísticos: con los mismos
datos producen siempre el mismo resultado (no se emplea remuestreo aleatorio).

## Validación

Los resultados obtenidos con este código fueron verificados de forma
independiente en **IBM SPSS Statistics**, con resultados concordantes en:
estadística descriptiva y prueba de normalidad (Etapa 1), prueba de Mann-Whitney
(Etapa 2), AUC e IC 95% de la curva ROC (Etapa 2) y regresión logística —odds
ratios, IC 95% y p-valores— (Etapa 4).

## Nota sobre el desarrollo

Los pipelines de análisis fueron construidos con asistencia de IA (Claude, de
Anthropic) y verificados de forma independiente en SPSS, según se indica arriba.

## Licencia

Publicado bajo licencia MIT (ver archivo `LICENSE`).
