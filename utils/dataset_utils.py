import pandas as pd
import numpy as np


def build_label(tces):
    label_cols = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8']

    def most_common(row):
        if pd.notna(row['Consensus Label']) and str(row['Consensus Label']).strip() != '':
            return row['Consensus Label']

        vals = row[label_cols].dropna()
        if vals.empty:
            return np.nan
        return vals.mode().iloc[0]

    tces['Label'] = tces.apply(most_common, axis=1)
    # tces['Label'] = tces['Label'].map({"J": 0, "E": 1, "S": 2, "B": 3, "N": 5})
    tces['Label'] = tces['Label'].map({"J": 0, "S": 0, "E": 1, "B": 1, "N": 5})
    tces = tces[tces['Label'] != 5]

    tces.drop(label_cols, axis=1, inplace=True)
    tces.drop('Consensus Label', axis=1, inplace=True)

    return tces



def tic_id_to_filename_id(tic_id):
    return f"{int(tic_id):016d}"


def build_dataset():
    CSV_TCES = "tces_with_labels_v3.csv"

    tces = pd.read_csv(CSV_TCES)
    tces.drop(0, inplace=True)
    tces.drop('Notes', axis=1, inplace=True)
    try:
        tces["TIC ID"] = tces["TIC ID"].astype(int)
        tces["Epoch"] = tces["Epoch"].astype(float)
        tces["Period"] = tces["Period"].astype(float)
        tces["Duration"] = tces["Duration"].astype(float)
    except Exception as e:
        print(f"Erro ao converter tipos: {e}")
        return []

    tces["FILE ID"] = tces["TIC ID"].apply(tic_id_to_filename_id)
    tces.dropna(subset=["TIC ID", "Epoch", "Period", "Duration"], inplace=True)
    print(f"TCES {len(tces):,} encontrados")

    tces = build_label(tces)
    return tces