import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, average_precision_score
import os


MODELS_TO_PLOT = {
    'CNN (rw=true)': {
        'folder_name': 'CNN_lcs_lag_90_e100_20250622-002513',
        'color': 'blue'
    },
    'CNN (rw=false)': {
        'folder_name': 'CNN_lcs_lag_90_e100_20250622-002920',
        'color': 'purple'
    },
    'CNN Astronet(rw=true)': {
        'folder_name': 'CNN-Astronet_lcs_lag_90_e100_20250622-003743',
        'color': 'green'
    },
    'CNN Astronet (rw=false)': {
        'folder_name': 'CNN-Astronet_lcs_lag_90_e100_20250622-004858',
        'color': 'red'
    }
}
PLOT_NAME = 'CNN_Astronet-comparison.png'

project_root = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(project_root)
results_base_path = os.path.join(project_root, 'results', 'imgs')
plot_path = os.path.join(project_root, 'results', 'comparisons', PLOT_NAME)

# Criar a figura com dois subplots (PR e ROC)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# --- Loop para plotar cada modelo ---
for model_name, config in MODELS_TO_PLOT.items():
    predictions_path = os.path.join(results_base_path, config['folder_name'], 'predictions.npz')
    print(predictions_path)
    try:
        data = np.load(predictions_path)
        y_true = data['y_true']
        y_pred_proba = data['y_pred_proba']
    except FileNotFoundError:
        print(f"AVISO: Arquivo de previsões não encontrado para o modelo '{model_name}'. Pulando: {config['folder_name']}")
        continue

    # 1. Calcular e Plotar a Curva Precision-Recall (PR)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    ap_score = average_precision_score(y_true, y_pred_proba)  # Average Precision
    ax1.plot(recall, precision, color=config['color'], lw=2, label=f'{model_name} (AP = {ap_score:.3f})')

    # 2. Calcular e Plotar a Curva ROC
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)  # AUC
    ax2.plot(fpr, tpr, color=config['color'], lw=2, label=f'{model_name} (AUC = {auc_score:.3f})')

# --- Formatação dos Gráficos ---

# Gráfico da Curva PR
ax1.set_xlabel('Recall (Sensibilidade)', fontsize=12)
ax1.set_ylabel('Precision', fontsize=12)
ax1.set_title('Curva Precision-Recall Comparativa', fontsize=14)
ax1.set_xlim([0.2, 1.0])
ax1.set_ylim([0.2, 1.0])
ax1.grid(linestyle='--')
ax1.legend(loc="lower left")

# Gráfico da Curva ROC
ax2.set_xlabel('Taxa de Falsos Positivos (FPR)', fontsize=12)
ax2.set_ylabel('Taxa de Verdadeiros Positivos (TPR)', fontsize=12)
ax2.set_title('Curva ROC Comparativa', fontsize=14)
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.0])
ax2.grid(linestyle='--')
ax2.legend(loc="lower right")

plt.suptitle('Comparação de Desempenho dos Modelos no Conjunto de Teste', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(plot_path)
plt.show()