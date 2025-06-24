import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm

# Importações do Mamba
from mamba_ssm import Mamba


# =================================================================================
# 1. DEFINIÇÃO DA ARQUITETURA DO MODELO (DUAS TORRES)
# =================================================================================

class MambaDualClassifier(nn.Module):
    def __init__(self, d_model=64, d_state=8, d_conv=4, expand=2):
        """
        d_model: Dimensão do embedding interno do Mamba.
        d_state: Dimensão do estado latente (N).
        d_conv: Largura da convolução 1D interna.
        expand: Fator de expansão da dimensão interna.
        """
        super().__init__()

        # Torre 1: Processa a visão global (input dim = 1)
        self.mamba_global = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        # Torre 2: Processa a visão local (input dim = 1)
        self.mamba_local = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        # Camadas de projeção para mapear o input de dim=1 para d_model
        self.global_proj = nn.Linear(1, d_model)
        self.local_proj = nn.Linear(1, d_model)

        # Cabeça de Classificação: recebe os embeddings concatenados (2 * d_model)
        self.classification_head = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)  # Saída para 2 classes (planeta vs. não-planeta)
        )

    def forward(self, x_global, x_local):
        # Projetar inputs para a dimensão do modelo
        x_global = self.global_proj(x_global)
        x_local = self.local_proj(x_local)

        # Processar cada visão em sua torre Mamba
        global_out = self.mamba_global(x_global)
        local_out = self.mamba_local(x_local)

        # Usar o vetor de características do último passo de tempo
        global_embedding = global_out[:, -1, :]
        local_embedding = local_out[:, -1, :]

        # Concatenar os embeddings
        fused_embedding = torch.cat((global_embedding, local_embedding), dim=1)

        # Classificação final
        output = self.classification_head(fused_embedding)
        return output


# =================================================================================
# 2. CARREGAMENTO E PREPARAÇÃO DOS DADOS
# =================================================================================

# --- Classe Dataset customizada para PyTorch com múltiplos inputs ---
class ExoplanetDataset(Dataset):
    def __init__(self, global_views, local_views, labels):
        self.global_views = torch.tensor(global_views, dtype=torch.float32)
        self.local_views = torch.tensor(local_views, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "global": self.global_views[idx],
            "local": self.local_views[idx],
            "label": self.labels[idx]
        }


# --- Carregar e processar os dados (lógica de normalização mantida) ---
try:
    data = np.load('../datasets/lcs_dataset_processed_sml.npz', allow_pickle=True)
except FileNotFoundError:
    print("Arquivo 'lcs_dataset_processed_sml.npz' não encontrado.")
    exit()

flux_global = data['flux_global']
flux_local = data['flux_local']
labels = data['label']

flux_global = np.nan_to_num(flux_global)
flux_local = np.nan_to_num(flux_local)
epsilon = 1e-7
flux_global_norm = (flux_global - np.median(flux_global, axis=1, keepdims=True)) / (
            np.std(flux_global, axis=1, keepdims=True) + epsilon)
flux_local_norm = (flux_local - np.median(flux_local, axis=1, keepdims=True)) / (
            np.std(flux_local, axis=1, keepdims=True) + epsilon)

# A Mamba espera o formato (batch, length, dim), então adicionamos a última dimensão
X_global_final = np.expand_dims(flux_global_norm, axis=-1)
X_local_final = np.expand_dims(flux_local_norm, axis=-1)

# --- Divisão de dados 60/20/20 ---
X_g_train_val, X_g_test, X_l_train_val, X_l_test, y_train_val, y_test = train_test_split(
    X_global_final, X_local_final, labels, test_size=0.2, random_state=42, stratify=labels)

X_g_train, X_g_val, X_l_train, X_l_val, y_train, y_val = train_test_split(
    X_g_train_val, X_l_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)

# --- Criar DataLoaders do PyTorch ---
train_dataset = ExoplanetDataset(X_g_train, X_l_train, y_train)
val_dataset = ExoplanetDataset(X_g_val, X_l_val, y_val)
test_dataset = ExoplanetDataset(X_g_test, X_l_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128)
test_loader = DataLoader(test_dataset, batch_size=128)

# =================================================================================
# 3. LOOP DE TREINAMENTO E VALIDAÇÃO
# =================================================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

model = MambaDualClassifier().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
n_epochs = 25  # Mamba tende a convergir rápido

print("\nIniciando o treinamento do modelo Mamba (Duas Torres)...")

for epoch in range(n_epochs):
    model.train()
    total_train_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs} [Train]")

    for batch in progress_bar:
        optimizer.zero_grad()

        global_batch = batch['global'].to(device)
        local_batch = batch['local'].to(device)
        labels_batch = batch['label'].to(device)

        outputs = model(global_batch, local_batch)
        loss = criterion(outputs, labels_batch)

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    # Validação
    model.eval()
    total_val_loss = 0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for batch in val_loader:
            global_batch = batch['global'].to(device)
            local_batch = batch['local'].to(device)
            labels_batch = batch['label'].to(device)

            outputs = model(global_batch, local_batch)
            loss = criterion(outputs, labels_batch)
            total_val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_val += labels_batch.size(0)
            correct_val += (predicted == labels_batch).sum().item()

    avg_train_loss = total_train_loss / len(train_loader)
    avg_val_loss = total_val_loss / len(val_loader)
    val_accuracy = 100 * correct_val / total_val
    print(
        f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

# =================================================================================
# 4. AVALIAÇÃO FINAL NO CONJUNTO DE TESTE
# =================================================================================
print("\n--- Avaliação Final no Conjunto de Teste (Mamba) ---")
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for batch in test_loader:
        global_batch = batch['global'].to(device)
        local_batch = batch['local'].to(device)
        labels_batch = batch['label'].to(device)

        outputs = model(global_batch, local_batch)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels_batch.cpu().numpy())

y_pred_classes = np.array(all_preds)
y_test_final = np.array(all_labels)

# Reutilizando o mesmo código de avaliação para consistência
print("\nClassification Report:")
report = classification_report(y_test_final, y_pred_classes,
                               target_names=['Classe 0 (Não Planeta)', 'Classe 1 (Planeta)'])
print(report)

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test_final, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Previsto Não Planeta', 'Previsto Planeta'],
            yticklabels=['Real Não Planeta', 'Real Planeta'])
plt.xlabel('Previsão')
plt.ylabel('Real')
plt.title('Matriz de Confusão - Mamba (Duas Torres)')
plt.show()