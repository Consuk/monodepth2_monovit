import pandas as pd

# --- CONFIGURACIÓN ---
# Reemplaza por la ruta real de tu CSV
csv_path = "corruptions_summary.csv"

# Lee el archivo (asegúrate de que tiene encabezados como los que mostraste)
df = pd.read_csv(csv_path)

# Si tu CSV tiene columnas adicionales (como 'corruption' o 'severity'),
# asegúrate de que solo seleccionas las métricas numéricas:
metric_cols = ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]

# --- CÁLCULO DE PROMEDIOS ---
global_means = df[metric_cols].mean()

# --- RESULTADOS ---
print("=== Promedio Global de Todas las Corrupciones y Severidades ===")
for metric, value in global_means.items():
    print(f"{metric:10s}: {value:.3f}")

# --- OPCIONAL: Promedio por tipo de corrupción ---
if "corruption" in df.columns:
    avg_per_corr = df.groupby("corruption")[metric_cols].mean().reset_index()
    print("\n=== Promedio por tipo de corrupción ===")
    print(avg_per_corr)

    # Guarda tabla resumen
    avg_per_corr.to_csv("corruption_averages.csv", index=False)
    print("\nPromedios por corrupción guardados en 'corruption_averages.csv'")
