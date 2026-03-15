import os
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser("Average corruption metrics CSV")
    parser.add_argument("--input_csv", type=str, required=True,
                        help="CSV de entrada con métricas por corrupción/severidad")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directorio donde guardar resultados")
    parser.add_argument("--global_filename", type=str, default="global_average.csv",
                        help="Nombre del CSV de promedio global")
    parser.add_argument("--per_corruption_filename", type=str, default="corruption_averages.csv",
                        help="Nombre del CSV de promedio por corrupción")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input_csv)

    metric_cols = ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]

    global_means = df[metric_cols].mean()

    print("=== Promedio Global de Todas las Corrupciones y Severidades ===")
    for metric, value in global_means.items():
        print(f"{metric:10s}: {value:.6f}")

    global_df = pd.DataFrame([["global"] + list(global_means.values)],
                             columns=["label"] + metric_cols)

    global_out = os.path.join(args.output_dir, args.global_filename)
    global_df.to_csv(global_out, index=False)
    print(f"\nPromedio global guardado en: {global_out}")

    if "corruption" in df.columns:
        avg_per_corr = df.groupby("corruption")[metric_cols].mean().reset_index()

        print("\n=== Promedio por tipo de corrupción ===")
        print(avg_per_corr)

        corr_out = os.path.join(args.output_dir, args.per_corruption_filename)
        avg_per_corr.to_csv(corr_out, index=False)
        print(f"\nPromedios por corrupción guardados en: {corr_out}")


if __name__ == "__main__":
    main()