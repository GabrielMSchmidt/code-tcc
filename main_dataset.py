import numpy as np
from matplotlib import pyplot as plt
import random
import os
from utils import dataset_utils, lc_utils


def plot_light_curve(example):
    id, label, fg, fl = example['id'], example['label'], example['flux_global'], example['flux_local']
    bins_global = np.linspace(-0.5, 0.5, 2002)
    bins_local_width = (4 * example['duration']) / example['period']
    bins_local = np.linspace(-bins_local_width / 2, bins_local_width / 2, 202)
    xg, xl = 0.5 * (bins_global[:-1] + bins_global[1:]), 0.5 * (bins_local[:-1] + bins_local[1:])
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharey=False)
    fig.suptitle(f'Curva de Luz TIC {id} - {label}', fontsize=14)
    axs[0].plot(xg, fg, 'k.-', markersize=3)
    axs[0].set_title('Visão Global (Dobrada)')
    axs[0].set_xlabel('Fase')
    axs[0].set_ylabel('Fluxo Normalizado')
    axs[1].plot(xl, fl, 'r.-', markersize=3)
    axs[1].set_title('Visão Local (Trânsito)')
    axs[1].set_xlabel('Fase')
    axs[1].set_ylabel('Fluxo Normalizado')
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    plt.show()


def main():
    tces = dataset_utils.build_dataset()
    if tces is None or tces.empty:
        print("Não foi possível carregar o dataset de TCEs. Abortando.")
        return

    flux_columns_to_test = ['SAP_FLUX_LAG']
    nan_thresholds_to_test = [0.99]

    for flux_col in flux_columns_to_test:
        for nan_thresh in nan_thresholds_to_test:
            lcs = lc_utils.extraction(tces, flux_column=flux_col, nan_threshold=nan_thresh)

            if not lcs:
                print(f"Nenhuma curva de luz gerada para {flux_col} com threshold {nan_thresh}. Pulando salvamento.")
                continue

            flux_name = flux_col.replace('SAP_FLUX_', '').lower()
            thresh_percent = int(nan_thresh * 100)

            output_dir = '../TCC/datasets'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            output_filename = os.path.join(output_dir, f'lcs_{flux_name}_{thresh_percent}_2001.npz')

            lc_utils.save_lcs(lcs, output_filename)

    print("\nPlotando alguns exemplos do último dataset gerado...")
    for _ in range(20):
        k = random.randint(0, len(lcs) - 1)
        if len(lcs[k]['flux_local']) > 10:
            plot_light_curve(lcs[k])
        else:
            print(f"Pulando TIC {lcs[k]['id']} — poucos pontos na visão local.")

if __name__ == "__main__":
    main()