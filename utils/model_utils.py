import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score, \
    accuracy_score, precision_score, recall_score, f1_score, roc_curve, precision_recall_curve
import pandas as pd
import os
from datetime import datetime

def load_and_preprocess_data(filepath):
    """
    Carrega o dataset .npz, trata NaNs e normaliza as visões.
    """
    try:
        data = np.load(filepath, allow_pickle=True)
    except FileNotFoundError:
        print(f"ERRO: Arquivo de dataset '{filepath}' não encontrado.")
        return None, None, None, None

    flux_global = data['flux_global']
    flux_local = data['flux_local']
    labels = data['label'].astype(np.float32)
    splits = data['split']

    print(f"Dataset carregado de '{filepath}'.")
    print(f"Forma original: Global={flux_global.shape}, Local={flux_local.shape}")

    # Tratamento de NaNs/Infs que possam ter restado
    flux_global = np.nan_to_num(flux_global, nan=0.0, posinf=1.0, neginf=-1.0)
    flux_local = np.nan_to_num(flux_local, nan=0.0, posinf=1.0, neginf=-1.0)

    # Normalização padrão por visão (subtrair mediana, dividir por desvio padrão)
    epsilon = 1e-7
    std_dev_global = np.std(flux_global, axis=1, keepdims=True)
    std_dev_local = np.std(flux_local, axis=1, keepdims=True)

    flux_global_norm = (flux_global - np.median(flux_global, axis=1, keepdims=True)) / (std_dev_global + epsilon)
    flux_local_norm = (flux_local - np.median(flux_local, axis=1, keepdims=True)) / (std_dev_local + epsilon)

    print("Dados pré-processados (tratamento de NaN e normalização) com sucesso.")
    return flux_global_norm, flux_local_norm, labels, splits


def split_data_by_column(global_data, local_data, labels, splits):
    """
    Divide os dados em conjuntos de treino, validação e teste usando a coluna 'splits'.
    """

    print("\nRealizando a divisão dos dados com base na coluna 'split' pré-definida...")

    splits_cleaned = np.char.strip(np.char.upper(splits.astype(str)))

    train_mask = (splits_cleaned == 'TRAIN')
    val_mask = (splits_cleaned == 'VAL')
    test_mask = (splits_cleaned == 'TEST')
    val_test_mask = val_mask | test_mask
    val_train_mask = val_mask | train_mask

    data_dict = {
        'X_global_train': global_data[train_mask],
        'X_local_train': local_data[train_mask],
        'y_train': labels[train_mask],

        'X_global_test': global_data[val_test_mask],
        'X_local_test': local_data[val_test_mask],
        'y_test': labels[val_test_mask],
    }
    data_dict = {
        'X_global_train': global_data[train_mask],
        'X_local_train': local_data[train_mask],
        'y_train': labels[train_mask],

        'X_global_val': global_data[val_mask],
        'X_local_val': local_data[val_mask],
        'y_val': labels[val_mask],

        'X_global_test': global_data[test_mask],
        'X_local_test': local_data[test_mask],
        'y_test': labels[test_mask],
    }

    original_total_samples = len(labels)
    print("--- Verificação da Divisão ---")
    print(f"Total de amostras: {original_total_samples}")
    print(f"Amostras no conjunto de Treino: {len(data_dict['y_train'])} ({len(data_dict['y_train']) / original_total_samples:.2%})")
    print(f"Amostras no conjunto de Validação: {len(data_dict['y_val'])} ({len(data_dict['y_val']) / original_total_samples:.2%})")
    print(f"Amostras no conjunto de Teste: {len(data_dict['y_test'])} ({len(data_dict['y_test']) / original_total_samples:.2%})")

    print("\n--- Distribuição Geral das Classes ---")

    class_1_total = np.sum(labels)
    class_0_total = original_total_samples - class_1_total
    class_1_perc = (class_1_total / original_total_samples) * 100
    class_0_perc = (class_0_total / original_total_samples) * 100

    print(f"-> Classe 0 (Não Planeta): {int(class_0_total)} ({class_0_perc:.2f}%)")
    print(f"-> Classe 1 (Planeta):   {int(class_1_total)} ({class_1_perc:.2f}%)")
    return data_dict


def evaluate_model_performance(y_true, y_pred_proba, model_name, threshold=0.5, save_path=None):
    """
    Exibe o relatório de classificação e a matriz de confusão para um modelo.
    """

    print(f"\n--- Avaliação Detalhada no Conjunto de Teste ({model_name}) ---")

    # Converte probabilidades para classes com limiar pré-definido
    y_pred_classes = (y_pred_proba > threshold).astype(int)

    # 1. Relatório de Classificação
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred_classes, target_names=['Classe 0 (Não Planeta)', 'Classe 1 (Planeta)'])
    print(report)

    # 2. Matriz de Confusão
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred_classes)
    print(cm)

    # 3. Plot da Matriz de Confusão
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Previsto Não Planeta', 'Previsto Planeta'],
                yticklabels=['Real Não Planeta', 'Real Planeta'])
    plt.xlabel('Previsão')
    plt.ylabel('Real')
    plt.title(f'Matriz de Confusão - {model_name}')

    # 4. Salvar a Matriz de Confusão
    if save_path:
        plt.savefig(save_path)
        plt.show()
        print(f"Matriz de confusão salva em: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_learning_curves(history, model_name, save_path=None):
    """
    Plota as curvas de aprendizado (loss e acurácia/auc).
    """

    if hasattr(history, 'history'):  # Se for um objeto Keras History
        hist_dict = history.history
    elif isinstance(history, dict):  # Se já for um dicionário (PyTorch)
        hist_dict = history
    else:
        print("Formato de 'history' não suportado para plotagem.")
        return

    if 'auc_pr' in hist_dict:
        metric_key = 'auc_pr'
        name_metric = 'AUC-PR'
    else:
        metric_key = 'accuracy'
        name_metric = 'Accuracy'

    if 'loss' not in hist_dict or metric_key not in hist_dict:
        print(f"Métricas de treino ('loss' e/ou '{metric_key}') não encontradas no histórico.")
        return

    fig, ax1 = plt.subplots(figsize=(11, 6))
    # --- Plotagem do Loss ---
    color = 'tab:red'
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(hist_dict['loss'], color=color, linestyle='-', label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    # --- Plotagem da Métrica (AUC/Accuracy) ---
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel(name_metric, color=color)
    ax2.plot(hist_dict[metric_key], color=color, linestyle='-', label=f'Train {name_metric}')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    fig.tight_layout()
    plt.title(f'Curvas de Aprendizado: Loss & {name_metric} ({model_name})')

    if save_path:
        plt.savefig(save_path)
        plt.show()
        print(f"Gráfico de aprendizado salvo em: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_roc_curve(y_true, y_pred_proba, model_name, save_path):
    """
        Plota a curva ROC.
    """

    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    # --- Plot do Gráfico ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(8, 6))

    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Taxa de Falsos Positivos (FPR)', fontsize=12)
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)', fontsize=12)
    plt.title(f'Curva ROC: ({model_name}) - (AUC = {roc_auc:.2f})')
    plt.legend(loc="lower right")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        plt.show()
        print(f"Curva ROC salva em: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_pr_curve(y_true, y_pred_proba, model_name, save_path):
    """
        Plota a curva PR.
    """

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)

    # --- Plot do Gráfico ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(8, 6))

    plt.plot(recall, precision, color='darkorange', lw=2, label=f'Curva PR (AP = {pr_auc:.2f})')
    plt.xlim([0.2, 1.0])
    plt.ylim([0.2, 1.0])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Curva Precision-Recall: ({model_name}) - (AP = {pr_auc:.2f})')
    plt.legend(loc="lower right")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        plt.show()
        print(f"Curva Precision-Recall salva em: {save_path}")
    else:
        plt.show()
    plt.close()
    return pr_auc

def save_results(model_name, dataset_path, hyperparameters, history, y_true, y_pred_proba, training_time, threshold=0.5):
    """
    Calcula métricas, salva gráficos numa pasta com nome descritivo
    e anexa os resultados numéricos a um arquivo CSV específico do modelo.
    """
    print("\n--- Salvando Resultados do Experimento ---")

    # 1. Criar um diretório único para este experimento
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    epochs = hyperparameters.get('epochs', 'N_A')
    experiment_name = f"{model_name}_{dataset_name}_e{epochs}_{timestamp}"
    project_root = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(project_root)
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    exp_results_dir = os.path.join(results_dir, 'imgs', experiment_name)
    os.makedirs(exp_results_dir, exist_ok=True)
    print(f"Diretório de resultados criado em: {exp_results_dir}")

    # 2. Salvar os gráficos neste diretório
    evaluate_model_performance(y_true, y_pred_proba, model_name, threshold, save_path=os.path.join(exp_results_dir, 'confusion_matrix.png'))
    plot_learning_curves(history, model_name, save_path=os.path.join(exp_results_dir, 'learning_curves.png'))
    plot_roc_curve(y_true, y_pred_proba, model_name, os.path.join(exp_results_dir, 'roc_curve.png'))
    pr_auc = plot_pr_curve(y_true, y_pred_proba, model_name, os.path.join(exp_results_dir, 'pr_curve.png'))

    # 3. Salvar as predições para gráficos futuros
    y_true_flat = y_true.flatten()
    y_pred_proba_flat = y_pred_proba.flatten()
    predictions_path = os.path.join(exp_results_dir, 'predictions.npz')
    np.savez(predictions_path, y_true=y_true_flat, y_pred_proba=y_pred_proba_flat)
    print(f"Previsões do modelo {model_name} salvas em: {predictions_path}")

    # 4. Calcular todas as métricas numéricas
    y_pred_classes = (y_pred_proba > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred_classes)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        'timestamp': timestamp,
        'model_name': model_name,
        'dataset': dataset_name,
        'accuracy': accuracy_score(y_true, y_pred_classes),
        'auc-roc': roc_auc_score(y_true, y_pred_proba),
        'auc-pr': pr_auc,
        'precision_class1': precision_score(y_true, y_pred_classes, pos_label=1, zero_division=0),
        'recall_class1': recall_score(y_true, y_pred_classes, pos_label=1, zero_division=0),
        'f1_score_class1': f1_score(y_true, y_pred_classes, pos_label=1, zero_division=0),
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn,
        'training_time': training_time,
        'results_path': exp_results_dir
    }

    # 5. Combinar hiperparâmetros e métricas
    results_data = {**metrics, **hyperparameters}

    # 6. Anexar ao arquivo CSV específico do modelo
    df = pd.DataFrame([results_data])

    # Define o caminho do CSV dinamicamente com base no nome do modelo
    results_csv_path = os.path.join(results_dir, f"{model_name.lower()}_experiments.csv")

    # Escreve o cabeçalho se o arquivo não existir, senão anexa sem cabeçalho
    df.to_csv(results_csv_path, mode='a', header=not os.path.exists(results_csv_path), index=False)
    print(f"Resultados numéricos anexados com sucesso a: {results_csv_path}")