import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ReLU, MaxPooling1D, Dropout, Flatten, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
import os
from utils import model_utils
from sklearn.utils import class_weight


def create_model(global_view_shape=(201, 1), local_view_shape=(61, 1), learning_rate=0.001):
    """
    Cria um modelo CNN inspirado na Astronet para classificação binária de curvas de luz.
    """
    input_global = Input(shape=global_view_shape, name='global_view_input')
    x_global = Conv1D(filters=16, kernel_size=5, padding='same', name='global_conv1')(input_global)
    x_global = BatchNormalization(name='global_bn1')(x_global)
    x_global = ReLU(name='global_relu1')(x_global)
    x_global = MaxPooling1D(pool_size=2, strides=2, name='global_pool1')(x_global)
    x_global = Conv1D(filters=32, kernel_size=5, padding='same', name='global_conv2')(x_global)
    x_global = BatchNormalization(name='global_bn2')(x_global)
    x_global = ReLU(name='global_relu2')(x_global)
    x_global = MaxPooling1D(pool_size=2, strides=2, name='global_pool2')(x_global)
    x_global = Conv1D(filters=64, kernel_size=3, padding='same', name='global_conv3')(x_global)
    x_global = BatchNormalization(name='global_bn3')(x_global)
    x_global = ReLU(name='global_relu3')(x_global)
    x_global = MaxPooling1D(pool_size=2, strides=2, name='global_pool3')(x_global)
    x_global = Flatten(name='global_flatten')(x_global)
    x_global = Dropout(0.25, name='global_dropout_flatten')(x_global)

    input_local = Input(shape=local_view_shape, name='local_view_input')
    x_local = Conv1D(filters=16, kernel_size=3, padding='same', name='local_conv1')(input_local)
    x_local = BatchNormalization(name='local_bn1')(x_local)
    x_local = ReLU(name='local_relu1')(x_local)
    x_local = MaxPooling1D(pool_size=2, strides=2, name='local_pool1')(x_local)
    x_local = Conv1D(filters=32, kernel_size=3, padding='same', name='local_conv2')(x_local)
    x_local = BatchNormalization(name='local_bn2')(x_local)
    x_local = ReLU(name='local_relu2')(x_local)
    x_local = MaxPooling1D(pool_size=2, strides=2, name='local_pool2')(x_local)
    x_local = Flatten(name='local_flatten')(x_local)
    x_local = Dropout(0.25, name='local_dropout_flatten')(x_local)

    concatenated_features = Concatenate(name='concatenate_views')([x_global, x_local])
    x = Dense(128, name='dense1')(concatenated_features)
    x = BatchNormalization(name='dense_bn1')(x)
    x = ReLU(name='dense_relu1')(x)
    x = Dropout(0.5, name='dense_dropout1')(x)
    x = Dense(64, name='dense2')(x)
    x = BatchNormalization(name='dense_bn2')(x)
    x = ReLU(name='dense_relu2')(x)
    x = Dropout(0.5, name='dense_dropout2')(x)
    # output = Dense(1, activation='sigmoid', name='output_sigmoid')(x)
    output = Dense(1, name='output_logits')(x)  # Sem activation='sigmoid'

    model = Model(inputs=[input_global, input_local], outputs=output, name='AstronetInspiredCNN')
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy', tf.keras.metrics.AUC(curve="PR",name='auc_pr')])
    return model


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    DATASET_PATH = os.path.join(project_root, 'datasets', 'lcs_lag_90.npz')
    MODEL_NAME = 'CNN'
    hyperparams = {
        'epochs': 100,
        'learning_rate': 0.001,
        'early_stopping_patience': 25,
    }

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
        learning_rate=hyperparams['learning_rate']
    )
    model.summary()

    print(f"\nIniciando o treinamento do modelo {MODEL_NAME} por {hyperparams['epochs']} épocas...")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc_pr',
            mode='max',
            patience=hyperparams['early_stopping_patience'],
            verbose=1,
            restore_best_weights=False
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_auc_pr',
            mode='max',
            factor=0.2,  # Fator pelo qual a taxa de aprendizado será reduzida
            patience=10,  # Número de épocas sem melhora antes de reduzir
            min_lr=1e-6,  # Limite inferior para a taxa de aprendizado
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
        threshold=0.5
    )