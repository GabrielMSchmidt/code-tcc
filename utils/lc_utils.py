import os
import numpy as np
from tqdm import tqdm
from lightkurve import LightCurve
from astropy.io import fits
import warnings

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
FITS_DIR = os.path.join(project_root, '.LC')


def interpolate_empty_bins(binned_flux):
    binned_flux_interp = np.copy(binned_flux)
    nan_indices = np.where(np.isnan(binned_flux_interp))[0]
    for i in nan_indices:
        left_neighbor_val, right_neighbor_val = np.nan, np.nan
        idx_left, idx_right = -1, -1
        for j in range(i - 1, -1, -1):
            if not np.isnan(binned_flux_interp[j]):
                left_neighbor_val, idx_left = binned_flux_interp[j], j
                break
        for j in range(i + 1, len(binned_flux_interp)):
            if not np.isnan(binned_flux_interp[j]):
                right_neighbor_val, idx_right = binned_flux_interp[j], j
                break
        if not np.isnan(left_neighbor_val) and not np.isnan(right_neighbor_val):
            if idx_right != idx_left:
                binned_flux_interp[i] = left_neighbor_val + (i - idx_left) * (
                        right_neighbor_val - left_neighbor_val) / (idx_right - idx_left)
            else:
                binned_flux_interp[i] = (left_neighbor_val + right_neighbor_val) / 2.0
        elif not np.isnan(left_neighbor_val):
            binned_flux_interp[i] = left_neighbor_val
        elif not np.isnan(right_neighbor_val):
            binned_flux_interp[i] = right_neighbor_val
    return binned_flux_interp


def get_views(processed_lc, t0, period, duration):
    lc_no_nans = processed_lc.remove_nans()
    num_global_bins, num_local_bins = 2001, 201

    if len(lc_no_nans.time.value) == 0:
        empty_global, empty_local = np.full(num_global_bins, np.nan), np.full(num_local_bins, np.nan)
        return empty_global, empty_local, 1.0, 1.0, 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        folded_lc = lc_no_nans.fold(period=period, epoch_time=t0)

    phase, flux = folded_lc.phase.value, folded_lc.flux.value
    bins_global_edges = np.linspace(-0.5, 0.5, num_global_bins + 1)
    phase_global_view, flux_global_view = phase, flux
    local_phase_window = min(2 * duration / period, 0.5)
    bins_local_edges = np.linspace(-local_phase_window, local_phase_window, num_local_bins + 1)
    local_mask = (phase >= -local_phase_window) & (phase <= local_phase_window)
    phase_local_view, flux_local_view = phase[local_mask], flux[local_mask]
    points_in_local_view_unbinned = len(phase_local_view[~np.isnan(flux_local_view)])

    def bin_flux_local_helper(phase_array, flux_array, bins_edges_local):
        if len(phase_array) == 0: return np.full(len(bins_edges_local) - 1, np.nan)
        if np.all(bins_edges_local == bins_edges_local[0]): return np.full(len(bins_edges_local) - 1, np.nan)
        unique_edges, _ = np.unique(bins_edges_local, return_index=True)
        if len(unique_edges) < 2: return np.full(len(bins_edges_local) - 1, np.nan)
        aux = np.digitize(phase_array, bins_edges_local) - 1
        aux[aux == len(bins_edges_local) - 1] = len(bins_edges_local) - 2
        binned_values = [np.nanmedian(flux_array[aux == i]) if np.any(aux == i) else np.nan for i in
                         range(len(bins_edges_local) - 1)]
        return np.array(binned_values)

    binned_global_raw = bin_flux_local_helper(phase_global_view, flux_global_view, bins_global_edges)
    binned_local_raw = bin_flux_local_helper(phase_local_view, flux_local_view, bins_local_edges)

    binned_global_interp = interpolate_empty_bins(binned_global_raw)
    binned_local_interp = interpolate_empty_bins(binned_local_raw)

    nan_frac_global = np.sum(np.isnan(binned_global_raw)) / len(binned_global_raw) if len(
        binned_global_raw) > 0 else 1.0
    nan_frac_local = np.sum(np.isnan(binned_local_raw)) / len(binned_local_raw) if len(binned_local_raw) > 0 else 1.0
    return binned_global_interp, binned_local_interp, nan_frac_global, nan_frac_local, points_in_local_view_unbinned


def extraction(tces, flux_column='SAP_FLUX_MID', nan_threshold=0.9):
    """
    Função principal de extração e pré-processamento das curvas de luz.
    """
    print(f"\nIniciando extração com flux_column='{flux_column}' e nan_threshold={nan_threshold}...")
    lcs = []
    cont = 0
    for file in tqdm(os.listdir(FITS_DIR)):
        if not file.endswith(".fits"):
            continue
        try:
            base_name = os.path.basename(file).replace(".fits", "")
            try:
                file_id = base_name.split("-")[1].split("_")[0]
            except IndexError:
                continue

            row = tces[tces['FILE ID'] == file_id]
            if row.empty:
                continue

            epoch, period, duration_days = row['Epoch'].values[0], row['Period'].values[0], row['Duration'].values[0]
            label, split = row['Label'].values[0], row['Split'].values[0]

            with fits.open(os.path.join(FITS_DIR, file), memmap=False) as hdul:
                if len(hdul) < 2 or not hasattr(hdul[1], 'data') or hdul[1].data is None: continue
                data = hdul[1].data
                if flux_column not in data.columns.names: continue
                time_raw, flux_raw = data['TIME'], data[flux_column]

            valid_indices = ~np.isnan(time_raw) & ~np.isnan(flux_raw)
            time, flux = time_raw[valid_indices], flux_raw[valid_indices]
            if len(time) < 5: continue
            lc = LightCurve(time=time, flux=flux)
            try:
                lc_no_outliers = lc.remove_outliers(sigma=5)
                if len(lc_no_outliers.time.value) > 0: lc = lc_no_outliers
            except Exception:
                pass
            if len(lc.time.value) == 0: continue
            try:
                lc = lc.normalize()
            except Exception:
                continue
            if len(lc.time.value) > 1:
                median_cadence_days = np.nanmedian(np.diff(lc.time.value))
                if median_cadence_days <= 1e-9: median_cadence_days = 2.0 / (24.0 * 60.0)
            else:
                median_cadence_days = 2.0 / (24.0 * 60.0)
            fixed_window_days = 2.0
            window_length_cadences = int(round(fixed_window_days / median_cadence_days))
            min_wl_cadences, current_lc_len = 5, len(lc.time.value)
            max_wl_cadences = current_lc_len - (current_lc_len % 2) - 1
            if max_wl_cadences < min_wl_cadences:
                max_wl_cadences = max(min_wl_cadences,
                                      current_lc_len - 1 if current_lc_len > min_wl_cadences else min_wl_cadences)
                if max_wl_cadences % 2 == 0 and max_wl_cadences > min_wl_cadences: max_wl_cadences -= 1
            window_length_cadences = max(min_wl_cadences, window_length_cadences)
            window_length_cadences = min(max_wl_cadences, window_length_cadences)
            if window_length_cadences % 2 == 0:
                if window_length_cadences + 1 <= max_wl_cadences:
                    window_length_cadences += 1
                elif window_length_cadences - 1 >= min_wl_cadences:
                    window_length_cadences -= 1
            if window_length_cadences < 3: window_length_cadences = 3
            if current_lc_len > window_length_cadences > 0:
                try:
                    lc = lc.flatten(window_length=window_length_cadences, polyorder=2, break_tolerance=10, niters=3,
                                    sigma=3)
                except Exception:
                    pass
            median_flux_after_flatten = np.nanmedian(lc.flux.value)
            if not np.isnan(median_flux_after_flatten) and median_flux_after_flatten != 0 and not (
                    0.95 < median_flux_after_flatten < 1.05):
                lc.flux = lc.flux / median_flux_after_flatten
            elif median_flux_after_flatten == 0 and np.any(lc.flux.value != 0):
                pass

            fg, fl, nan_frac_g, nan_frac_l, pts_local = get_views(lc, epoch, period, duration_days)

            # Validação com o limiar (threshold)
            if nan_frac_g > nan_threshold or nan_frac_l > nan_threshold:
                continue

            lcs.append({'id': file_id, 'flux_global': fg, 'flux_local': fl, 'epoch': epoch, 'period': period,
                        'duration': duration_days, 'label': label, 'split': split, 'nan_frac_global': nan_frac_g,
                        'nan_frac_local': nan_frac_l, 'pts_in_local_view': pts_local})
        except Exception:
            continue
    return lcs


def save_lcs(lcs, output_filename):
    """
    Salva a lista de curvas de luz processadas em um arquivo .npz.
    """
    if not lcs:
        print("Nenhuma curva de luz processada para salvar.")
        return

    try:
        ids = [item['id'] for item in lcs]
        flux_global_list = [item['flux_global'] for item in lcs]
        flux_local_list = [item['flux_local'] for item in lcs]
        epochs = [item['epoch'] for item in lcs]
        periods = [item['period'] for item in lcs]
        durations = [item['duration'] for item in lcs]
        labels = [item['label'] for item in lcs]
        splits = [item['split'] for item in lcs]
        nan_frac_global_list = [item['nan_frac_global'] for item in lcs]
        nan_frac_local_list = [item['nan_frac_local'] for item in lcs]
        pts_in_local_list = [item['pts_in_local_view'] for item in lcs]

        flux_global_arr = np.array(flux_global_list)
        flux_local_arr = np.array(flux_local_list)
        epochs_arr, periods_arr, durations_arr, labels_arr = np.array(epochs), np.array(periods), np.array(
            durations), np.array(labels)
        splits_arr, ids_arr = np.array(splits, dtype=object), np.array(ids, dtype=object)
        nan_frac_global_arr, nan_frac_local_arr, pts_in_local_arr = np.array(nan_frac_global_list), np.array(
            nan_frac_local_list), np.array(pts_in_local_list)

        np.savez(output_filename, ids=ids_arr, flux_global=flux_global_arr, flux_local=flux_local_arr, epoch=epochs_arr,
                 period=periods_arr, duration=durations_arr, label=labels_arr, split=splits_arr,
                 nan_frac_global=nan_frac_global_arr, nan_frac_local=nan_frac_local_arr,
                 pts_in_local_view=pts_in_local_arr)
        print(f"Dataset salvo em {output_filename} com {len(ids_arr)} amostras.")
    except ValueError as ve:
        warnings.warn(f"Erro ao converter listas de fluxo para arrays NumPy. Verifique shapes inconsistentes: {ve}")