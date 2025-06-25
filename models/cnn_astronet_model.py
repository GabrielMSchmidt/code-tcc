import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ReLU, MaxPooling1D, Dropout, Flatten, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
import os
import time
from utils import model_utils
from sklearn.utils import class_weight


def create_model(global_view_shape=(201, 1), local_view_shape=(61, 1), learning_rate=0.001, dropout=0.5):
    """
    Cria um modelo CNN inspirado na Astronet para classificação binária de curvas de luz.
    """
    # Torre da Visão Global
    input_global = Input(shape=global_view_shape, name='global_view_input')
    x_global = input_global

    # Bloco 1: 16 filtros
    x_global = Conv1D(filters=16, kernel_size=5, padding='same', activation='relu')(x_global)
    x_global = BatchNormalization()(x_global)
    x_global = Conv1D(filters=16, kernel_size=5, padding='same', activation='relu')(x_global)
    x_global = BatchNormalization()(x_global)
    x_global = MaxPooling1D(pool_size=5, strides=2)(x_global)

    # Bloco 2: 32 filtros
    x_global = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(x_global)
    x_global = BatchNormalization()(x_global)
    x_global = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(x_global)
    x_global = BatchNormalization()(x_global)
    x_global = MaxPooling1D(pool_size=5, strides=2)(x_global)

    # Bloco 3: 64 filtros
    x_global = Conv1D(filters=64, kernel_size=5, padding='same', activation='relu')(x_global)
    x_global = BatchNormalization()(x_global)
    x_global = Conv1D(filters=64, kernel_size=5, padding='same', activation='relu')(x_global)
    x_global = BatchNormalization()(x_global)
    x_global = MaxPooling1D(pool_size=5, strides=2)(x_global)

    # Bloco 4: 128 filtros
    x_global = Conv1D(filters=128, kernel_size=5, padding='same', activation='relu')(x_global)
    x_global = BatchNormalization()(x_global)
    x_global = Conv1D(filters=128, kernel_size=5, padding='same', activation='relu')(x_global)
    x_global = BatchNormalization()(x_global)
    x_global = MaxPooling1D(pool_size=5, strides=2)(x_global)

    # Bloco 5: 256 filtros
    x_global = Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')(x_global)
    x_global = BatchNormalization()(x_global)
    x_global = Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')(x_global)
    x_global = BatchNormalization()(x_global)
    x_global = MaxPooling1D(pool_size=5, strides=2)(x_global)

    x_global = Flatten(name='global_flatten')(x_global)

    # Torre da Visão Local
    input_local = Input(shape=local_view_shape, name='local_view_input')
    x_local = input_local

    # Bloco 1: 16 filtros
    x_local = Conv1D(filters=16, kernel_size=5, padding='same', activation='relu')(x_local)
    x_local = BatchNormalization()(x_local)
    x_local = Conv1D(filters=16, kernel_size=5, padding='same', activation='relu')(x_local)
    x_local = BatchNormalization()(x_local)
    x_local = MaxPooling1D(pool_size=7, strides=2)(x_local)

    # Bloco 2: 32 filtros
    x_local = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(x_local)
    x_local = BatchNormalization()(x_local)
    x_local = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(x_local)
    x_local = BatchNormalization()(x_local)
    x_local = MaxPooling1D(pool_size=7, strides=2)(x_local)

    x_local = Flatten(name='local_flatten')(x_local)

    # Junção e Camadas Densas
    concatenated_features = Concatenate(name='concatenate_views')([x_global, x_local])
    x = concatenated_features

    # 4 Camadas Densas de 512 unidades
    for _ in range(4):
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)

    output = Dense(1, name='output_logits')(x)

    model = Model(inputs=[input_global, input_local], outputs=output, name='AstronetCNN')
    optimizer = Adam(learning_rate=learning_rate)
    loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=['accuracy', tf.keras.metrics.AUC(curve="PR", name='auc_pr')]
    )
    return model


def main(dataset_filename, hyperparams):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    DATASET_PATH = os.path.join(project_root, 'datasets', dataset_filename)
    MODEL_NAME = 'CNN-Astronet'
    # hyperparams = {
    #     'epochs': 100,
    #     'learning_rate': 0.001,
    #     'early_stopping_patience': 15,
    #     'restore_best_weights': True,
    # }

    print(f"\n--- Iniciando Experimento: {MODEL_NAME} | Dataset: {dataset_filename} ---")
    print(f"Hiperparâmetros: {hyperparams}")

    # 1. Carregar e pré-processar dados
    flux_global, flux_local, labels, splits = model_utils.load_and_preprocess_data(DATASET_PATH)
    if flux_global is None:
        print("Saindo do script pois o carregamento dos dados falhou.")
        exit()

    # 2. Reshape específico para CNN
    X_global_reshaped = np.expand_dims(flux_global, axis=-1)
    X_local_reshaped = np.expand_dims(flux_local, axis=-1)

    # 3. Divisão dos dados
    data_sets = model_utils.split_data_by_column(X_global_reshaped, X_local_reshaped, labels, splits)

    # 4. Criação e treino do Modelo
    model = create_model(
        global_view_shape=(X_global_reshaped.shape[1], 1),
        local_view_shape=(X_local_reshaped.shape[1], 1),
        learning_rate=hyperparams['learning_rate'],
        dropout=hyperparams['dropout']
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc_pr',
            mode='max',
            patience=hyperparams['early_stopping_patience'],
            verbose=1,
            restore_best_weights=hyperparams['restore_best_weights']
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_auc_pr',
            mode='max',
            factor=0.2,
            patience=10,
            min_lr=1e-6,
            verbose=1
        )
    ]

    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(data_sets['y_train']),
        y=data_sets['y_train']
    )
    weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
    print(f"Pesos de classe calculados: {weights_dict}")

    print(f"\nIniciando o treinamento do modelo {MODEL_NAME} por {hyperparams['epochs']} épocas...")
    start_time = time.time()
    history = model.fit(
        [data_sets['X_global_train'], data_sets['X_local_train']],
        data_sets['y_train'],
        epochs=hyperparams['epochs'],
        validation_data=([data_sets['X_global_val'], data_sets['X_local_val']], data_sets['y_val']),
        callbacks=callbacks,
        batch_size=64,
        verbose=1,
        class_weight=weights_dict
    )
    end_time = time.time()
    training_time = end_time - start_time
    print(f"\nTreinamento concluído em {training_time:.2f} segundos.")

    # 5. Avaliação do Modelo
    y_pred_proba = model.predict([data_sets['X_global_test'], data_sets['X_local_test']])

    # 6. Plotar e salvar os resultados
    model_utils.save_results(
        model_name=MODEL_NAME,
        dataset_path=DATASET_PATH,
        hyperparameters=hyperparams,
        history=history,
        y_true=data_sets['y_test'],
        y_pred_proba=y_pred_proba,
        training_time=training_time,
        threshold=0.5
    )