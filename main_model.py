from models import cnn_astronet_model, inceptiontime_model

# =============================================================================
# PAINEL DE CONTROLE DE EXPERIMENTOS
# =============================================================================
EXPERIMENTS = [
    # --- CNN Experiments ---
    {
        'model_function': cnn_astronet_model.main,
        'dataset_filename': 'lcs_lag_99.npz',
        'hyperparams': {'epochs': 100, 'learning_rate': 0.001, 'early_stopping_patience': 25,
                        'restore_best_weights': True},
    },
    {
        'model_function': cnn_astronet_model.main,
        'dataset_filename': 'lcs_lag_90_2001.npz',
        'hyperparams': {'epochs': 100, 'learning_rate': 0.001, 'early_stopping_patience': 25,
                        'restore_best_weights': True},
    },

    # # --- Mamba Experiments ---
    # {
    #     'model_function': mamba_model.main,
    #     'dataset_filename': 'lcs_lag_99.npz',
    #     'hyperparams': {'epochs': 100, 'learning_rate': 0.001, 'early_stopping_patience': 25,
    #                     'd_model': 64, 'd_state': 8, 'd_conv': 4, 'expand': 2}
    # },
    # {
    #     'model_function': mamba_model.main,
    #     'dataset_filename': 'lcs_lag_90_2001.npz',
    #     'hyperparams': {'epochs': 100, 'learning_rate': 0.001, 'early_stopping_patience': 25,
    #                     'd_model': 64, 'd_state': 8, 'd_conv': 4, 'expand': 2}
    # },

    # --- InceptionTime Experiments ---
    {
        'model_function': inceptiontime_model.main,
        'dataset_filename': 'lcs_lag_99.npz',
        'hyperparams': {'epochs': 100, 'early_stopping_patience': 25,
                        'restore_best_weights': True}
    },
    {
        'model_function': inceptiontime_model.main,
        'dataset_filename': 'lcs_lag_90_2001.npz',
        'hyperparams': {'epochs': 100, 'early_stopping_patience': 25,
                        'restore_best_weights': True}
    },
]


# =============================================================================
# SCRIPT DE EXECUÇÃO
# =============================================================================
def run_all_experiments():
    """
    Itera sobre a lista de experimentos e executa cada um.
    """
    num_experiments = len(EXPERIMENTS)
    print(f"--- Iniciando a orquestração de {num_experiments} experimentos ---")

    for i, experiment_config in enumerate(EXPERIMENTS):
        model_func = experiment_config['model_function']
        dataset = experiment_config['dataset_filename']
        hyperparams = experiment_config['hyperparams']

        print(f"\n======================================================================")
        print(f"EXECUTANDO EXPERIMENTO {i + 1}/{num_experiments}: Modelo {model_func.__module__}")
        print(f"======================================================================")

        try:
            model_func(dataset, hyperparams)
            print(f"--- Experimento {i + 1}/{num_experiments} concluído com sucesso. ---")
        except Exception as e:
            print(f"!!!!!! ERRO AO EXECUTAR O EXPERIMENTO {i + 1}/{num_experiments} !!!!!!")
            print(f"Modelo: {model_func.__module__}, Dataset: {dataset}")
            print(f"Erro: {e}")
            print("!!!!!! Pulando para o próximo experimento. !!!!!!")
            continue


if __name__ == '__main__':
    run_all_experiments()