"""Contains tools for evaluating TTS model outputs."""

import numpy as np
import librosa
import fastdtw
from scipy.stats import pearsonr

def get_aligned_f0_contours(waveform_true: np.ndarray,
                             waveform_pred: np.ndarray):
    """Extracts F0 contours from the GT and predicted waveforms and aligns them using DTW."""

    f0_true, _, _ = librosa.pyin(waveform_true,
                                 fmin=librosa.note_to_hz('C2'),
                                 fmax=librosa.note_to_hz('C7'))
    f0_gen, _, _ = librosa.pyin(waveform_pred,
                                fmin=librosa.note_to_hz('C2'),
                                fmax=librosa.note_to_hz('C7'))
    
    f0_true[np.isnan(f0_true)] = 0.0
    f0_gen[np.isnan(f0_gen)] = 0.0
    f0_true[f0_true > 1000] = 0.0
    f0_gen[f0_gen > 1000] = 0.0
    
    mel_spec_true = librosa.feature.melspectrogram(y=waveform_true, sr=22050, n_mels=80)
    mel_spec_gen = librosa.feature.melspectrogram(y=waveform_pred, sr=22050, n_mels=80)
    mel_spec_true = librosa.power_to_db(mel_spec_true, ref=np.max)
    mel_spec_gen = librosa.power_to_db(mel_spec_gen, ref=np.max)

    _, path = fastdtw.fastdtw(mel_spec_true.T, mel_spec_gen.T, radius=10)

    path = np.array(path)

    aligned_true = f0_true[path[:, 0]]
    aligned_gen = f0_gen[path[:, 1]]

    max_len = max(len(aligned_true), len(aligned_gen))
    aligned_true = np.pad(aligned_true,
                          (0, max_len - len(aligned_true)),
                          mode='constant',
                          constant_values=0.0)
    aligned_gen = np.pad(aligned_gen,
                         (0, max_len - len(aligned_gen)),
                         mode='constant',
                         constant_values=0.0)
    
    return aligned_true, aligned_gen


def calculate_f0_metrics(pred_sample_path: str, true_sample_path: str):
    """Calculates F0-related evaluation metrics for predicted and ground-truth samples.
    
    The calculated metrics are:
        - F0 RMSE
        - F0 Pearson Correlation Coefficient
        - F0 MRE
    """

    waveform_true, _ = librosa.load(true_sample_path, sr=22050)
    waveform_pred, _ = librosa.load(pred_sample_path, sr=22050)

    aligned_true_f0, aligned_pred_f0 = get_aligned_f0_contours(
        waveform_true=waveform_true,
        waveform_pred=waveform_pred
    )

    pearson_corr, _ = pearsonr(aligned_true_f0, aligned_pred_f0)

    f0_rmse = np.sqrt(np.mean((aligned_true_f0 - aligned_pred_f0) ** 2))

    nonzero_indices = aligned_true_f0 > 0.0

    f0_err = np.abs(aligned_true_f0[nonzero_indices] - aligned_pred_f0[nonzero_indices])
    f0_mre = np.mean(f0_err / aligned_true_f0[nonzero_indices]) * 100.0

    return f0_rmse, pearson_corr, f0_mre