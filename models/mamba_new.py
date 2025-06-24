import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import os
import time
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchmetrics import AveragePrecision
from mamba_ssm import Mamba
from utils import model_utils


class MambaDualClassifier(nn.Module):
    def __init__(self, d_model=64, d_state=8, d_conv=4, expand=2):
        super().__init__()
        self.mamba_global = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_local = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.global_proj = nn.Linear(1, d_model)
        self.local_proj = nn.Linear(1, d_model)
        self.classification_head = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x_global, x_local):
        x_global, x_local = self.global_proj(x_global), self.local_proj(x_local)
        global_out, local_out = self.mamba_global(x_global), self.mamba_local(x_local)
        global_embedding, local_embedding = global_out[:, -1, :], local_out[:, -1, :]
        fused_embedding = torch.cat((global_embedding, local_embedding), dim=1)
        output = self.classification_head(fused_embedding)
        return output


class ExoplanetDataset(Dataset):
    def __init__(self, global_views, local_views, labels):
        self.global_views = torch.tensor(global_views, dtype=torch.float32)
        self.local_views = torch.tensor(local_views, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.global_views[idx], self.local_views[idx], self.labels[idx]


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    DATASET_PATH = os.path.join(project_root, 'datasets', 'lcs_dataset_processed_mid.npz')
    MODEL_NAME = 'Mamba'
    hyperparams = {
        'd_model': 64,
        'd_state': 8,
        'd_conv': 4,
        'expand': 2,
        'learning_rate': 0.001,
        'batch_size': 64,
        'epochs': 100,
        'early_stopping_patience': 15,
    }

    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")
    print(f"Usando dispositivo: {DEVICE}")

    # 1. Carregar e pré-processar dados
    flux_global, flux_local, labels, splits = model_utils.load_and_preprocess_data(DATASET_PATH)
    if flux_global is None:
        exit()

    # 2. Reshape específico para Mamba
    X_global_reshaped = np.expand_dims(flux_global, axis=-1)
    X_local_reshaped = np.expand_dims(flux_local, axis=-1)

    # 3. Divisão dos dados
    data_sets = model_utils.split_data_by_column(X_global_reshaped, X_local_reshaped, labels, splits)

    train_dataset = ExoplanetDataset(data_sets['X_global_train'], data_sets['X_local_train'], data_sets['y_train'])
    val_dataset = ExoplanetDataset(data_sets['X_global_val'], data_sets['X_local_val'], data_sets['y_val'])
    test_dataset = ExoplanetDataset(data_sets['X_global_test'], data_sets['X_local_test'], data_sets['y_test'])

    train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hyperparams['batch_size'] * 2)
    test_loader = DataLoader(test_dataset, batch_size=hyperparams['batch_size'] * 2)

    # 4. Criação e treino do Modelo
    model = MambaDualClassifier(d_model=hyperparams['d_model'], d_state=hyperparams['d_state'],
                                d_conv=hyperparams['d_conv'], expand=hyperparams['expand']).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total de parâmetros treináveis do Mamba: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=hyperparams['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.2, patience=10, verbose=True, min_lr=1e-6)

    # 5. Loop de Treinamento
    print(f"\nIniciando o treinamento do modelo {MODEL_NAME} por {hyperparams['epochs']} épocas...")
    best_val_pr_auc = -1.0
    epochs_no_improve = 0
    best_model_state = None
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': [], 'auc_pr': [], 'val_auc_pr': []}
    pr_auc_metric = AveragePrecision(task="binary").to(DEVICE)

    start_time = time.time()
    for epoch in range(hyperparams['epochs']):
        model.train()
        total_train_loss, total_train_acc, total_train_samples = 0, 0, 0
        pr_auc_metric.reset()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{hyperparams['epochs']} [Train]")
        for batch_global, batch_local, batch_labels in progress_bar:
            batch_global, batch_local, batch_labels = batch_global.to(DEVICE), batch_local.to(DEVICE), batch_labels.to(
                DEVICE)

            optimizer.zero_grad()
            outputs = model(batch_global, batch_local)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * batch_labels.size(0)
            preds_proba = torch.softmax(outputs, dim=1)[:, 1]
            pr_auc_metric.update(preds_proba, batch_labels)
            predicted_classes = torch.argmax(outputs, dim=1)
            total_train_acc += (predicted_classes == batch_labels).sum().item()
            total_train_samples += batch_labels.size(0)
            progress_bar.set_postfix(loss=loss.item())

        epoch_train_loss = total_train_loss / total_train_samples
        epoch_train_acc = total_train_acc / total_train_samples
        epoch_train_pr_auc = pr_auc_metric.compute().item()
        history['loss'].append(epoch_train_loss)
        history['accuracy'].append(epoch_train_acc)
        history['auc_pr'].append(
            epoch_train_pr_auc)

        # Validação
        model.eval()
        total_val_loss, total_val_acc, total_val_samples = 0, 0, 0
        pr_auc_metric.reset()
        with torch.no_grad():
            for batch_global, batch_local, batch_labels in val_loader:
                batch_global, batch_local, batch_labels = batch_global.to(DEVICE), batch_local.to(
                    DEVICE), batch_labels.to(DEVICE)
                outputs = model(batch_global, batch_local)
                loss = criterion(outputs, batch_labels)

                total_val_loss += loss.item() * batch_labels.size(0)
                preds_proba = torch.softmax(outputs, dim=1)[:, 1]
                pr_auc_metric.update(preds_proba, batch_labels)
                predicted_classes = torch.argmax(outputs, dim=1)
                total_val_acc += (predicted_classes == batch_labels).sum().item()
                total_val_samples += batch_labels.size(0)

        epoch_val_loss = total_val_loss / total_val_samples
        epoch_val_acc = total_val_acc / total_val_samples
        epoch_val_pr_auc = pr_auc_metric.compute().item()
        history['val_loss'].append(epoch_val_loss)
        history['val_accuracy'].append(epoch_val_acc)
        history['val_auc_pr'].append(
            epoch_val_pr_auc)

        print(
            f"Epoch {epoch + 1}/{hyperparams['epochs']} | Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f} | Train PR-AUC: {epoch_train_pr_auc:.4f}, Val PR-AUC: {epoch_val_pr_auc:.4f}")

        # Early Stopping
        if epoch_val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = epoch_val_pr_auc
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= hyperparams['early_stopping_patience']:
            print(f"\nEarly stopping na época {epoch + 1}.")
            model.load_state_dict(best_model_state)
            break

        scheduler.step(epoch_val_pr_auc)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"\nTreinamento concluído em {training_time:.2f} segundos.")

    if best_model_state:
        model.load_state_dict(best_model_state)

    # 6. Avaliação do Modelo
    model.eval()
    all_preds_proba, all_labels = [], []
    with torch.no_grad():
        for batch_global, batch_local, batch_labels in test_loader:
            batch_global, batch_local = batch_global.to(DEVICE), batch_local.to(DEVICE)
            outputs = model(batch_global, batch_local)
            preds_proba = torch.softmax(outputs, dim=1)
            all_preds_proba.extend(preds_proba.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    y_pred_proba_all = np.array(all_preds_proba)
    y_pred_proba_positive = y_pred_proba_all[:, 1]
    y_true = np.array(all_labels)

    # 7. Plotar e salvar os resultados
    model_utils.save_results(
        model_name=MODEL_NAME,
        dataset_path=DATASET_PATH,
        hyperparameters=hyperparams,
        history=history,
        y_true=y_true,
        y_pred_proba=y_pred_proba_positive,
        training_time=training_time,
        threshold=0.5
    )