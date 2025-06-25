import numpy as np
from sktime.classification.deep_learning import InceptionTimeClassifier
import tensorflow as tf
import time
import os
from utils import model_utils

def create_model(epochs, early_stopping_patience, restore_best_weights):
    model = InceptionTimeClassifier(
        n_epochs=epochs,
        batch_size=64,
        use_residual=True,
        use_bottleneck=True,
        depth=6,
        verbose=True,
        metrics=['accuracy', 'f1_score', tf.keras.metrics.AUC(curve="PR", name='auc_pr')],
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='auc_pr',
                mode='max',
                patience=early_stopping_patience,
                min_delta=0.01,
                verbose=1,
                restore_best_weights=restore_best_weights),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='auc_pr',
                mode='max',
                factor=0.2,
                patience=5,
                min_lr=1e-6,
                verbose=1),
        ],
        loss='binary_crossentropy'
    )
    return model

def main(dataset_filename, hyperparams):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    DATASET_PATH = os.path.join(project_root, 'datasets', dataset_filename)
    MODEL_NAME = 'InceptionTime'
    # hyperparams = {
    #     'epochs': 5,
    #     'early_stopping_patience': 3,
    #     'restore_best_weights': True,
    # }

    print(f"\n--- Iniciando Experimento: {MODEL_NAME} | Dataset: {dataset_filename} ---")
    print(f"Hiperparâmetros: {hyperparams}")

    # 1. Carregar e pré-processar dados
    flux_global, flux_local, labels, splits = model_utils.load_and_preprocess_data(DATASET_PATH)
    if flux_global is None:
        print("Saindo do script pois o carregamento dos dados falhou.")
        exit()

    # 2. Divisão dos dados
    data_sets = model_utils.split_data_by_column(flux_global, flux_local, labels, splits)

    # 3. Padding e Combinação para criar a Série Temporal Multivariada
    n_timesteps_global = flux_global.shape[1]
    n_timesteps_local = flux_local.shape[1]

    pad_total = n_timesteps_global - n_timesteps_local
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before

    flux_train_local_padded = np.pad(data_sets['X_local_train'], ((0, 0), (pad_before, pad_after)), 'constant', constant_values=0)
    flux_val_local_padded = np.pad(data_sets['X_local_val'], ((0, 0), (pad_before, pad_after)), 'constant', constant_values=0)
    flux_test_local_padded = np.pad(data_sets['X_local_test'], ((0, 0), (pad_before, pad_after)), 'constant', constant_values=0)

    flux_local_padded = np.pad(flux_local, ((0, 0), (pad_before, pad_after)), 'constant', constant_values=0)
    print(f"\nForma da visão local após padding: {flux_train_local_padded.shape}")

    X_train_multivariate = np.stack([data_sets['X_global_train'], flux_train_local_padded], axis=1)
    X_val_multivariate = np.stack([data_sets['X_global_val'], flux_val_local_padded], axis=1)
    X_test_multivariate = np.stack([data_sets['X_global_test'], flux_test_local_padded], axis=1)
    # X_multivariate = np.stack([flux_global, flux_local_padded], axis=1)

    print(f"Forma final dos dados de entrada multivariados para sktime: {X_train_multivariate.shape}")

    # 4. Criação e treino do Modelo
    print("\nCombinando dados de treino e validação para o treinamento final...")
    X_train_final = np.concatenate((X_train_multivariate, X_val_multivariate), axis=0)
    y_train_final = np.concatenate((data_sets['y_train'], data_sets['y_val']), axis=0)

    model = create_model(hyperparams['epochs'], hyperparams['early_stopping_patience'], hyperparams['restore_best_weights'])

    print(f"\nIniciando o treinamento do modelo {MODEL_NAME} por {hyperparams['epochs']} épocas...")
    start_time = time.time()
    model.fit(X_train_multivariate, data_sets['y_train'])
    end_time = time.time()
    training_time = end_time - start_time
    print(f"\nTreinamento concluído em {training_time:.2f} segundos.")

    # 5. Avaliação do Modelo
    y_pred_proba = model.predict_proba(X_test_multivariate)
    y_pred_proba_positive_class = y_pred_proba[:, 1]

    # 6. Plotar e salvar os resultados
    model_utils.save_results(
        model_name=MODEL_NAME,
        dataset_path=DATASET_PATH,
        hyperparameters=hyperparams,
        history=model.history,
        y_true=data_sets['y_test'],
        y_pred_proba=y_pred_proba_positive_class,
        training_time=training_time,
        threshold=0.5
    )