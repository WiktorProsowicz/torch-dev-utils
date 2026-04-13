"""Contains utilities for visualization during training/inference."""

import pathlib

import torch
import matplotlib.pyplot as plt
import numpy as np
import librosa


def plot_spectrograms(pred_spec: torch.Tensor,
                      target_spec: torch.Tensor):
    """Plots predicted and target mel-spectrograms side by side."""

    fig, axs = plt.subplots(2, 1, figsize=(10, 4))

    max_length = max(pred_spec.shape[1], target_spec.shape[1])

    pred_spec = torch.nn.functional.pad(pred_spec,
                                        (0, max_length - pred_spec.shape[1]),
                                        value=0.0)
    target_spec = torch.nn.functional.pad(target_spec,
                                          (0, max_length - target_spec.shape[1]),
                                          value=0.0)

    axs[0].imshow(pred_spec.cpu().numpy(), origin='lower')
    axs[0].set_title('Predicted Mel-Spectrogram')

    axs[1].imshow(target_spec.cpu().numpy(), origin='lower')
    axs[1].set_title('Target Mel-Spectrogram')

    mae = torch.mean(torch.abs(pred_spec - target_spec)).item()

    fig.suptitle(f'Mel-Spectrograms (MAE: {mae:.4f})')
    fig.tight_layout()

    return fig

def plot_spec_text_alignment(alignment: torch.Tensor):
    """Plots alignment matrix between text and spectrogram frames."""

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.imshow(alignment.cpu().numpy().T, origin='lower', interpolation='none')
    ax.set_xlabel('Spectrogram Frame Index')
    ax.set_ylabel('Text Token Index')
    ax.set_title('Alignment Matrix')

    fig.tight_layout()

    return fig

def plot_contours(pred_contour: torch.Tensor,
                  target_contour: torch.Tensor,
                  contour_name: str):
    """Plots predicted and target contours (e.g. pitch/energy/duration) over time."""

    fig, ax = plt.subplots(figsize=(10, 4))

    max_length = max(pred_contour.shape[0], target_contour.shape[0])

    pred_contour = torch.nn.functional.pad(pred_contour,
                                          (0, max_length - pred_contour.shape[0]),
                                          value=0.0)
    target_contour = torch.nn.functional.pad(target_contour,
                                            (0, max_length - target_contour.shape[0]),
                                            value=0.0)
    
    ax.plot(pred_contour.cpu().numpy(), label='Predicted', color='blue')
    ax.plot(target_contour.cpu().numpy(), label='Target', color='orange')
    ax.set_title(f'{contour_name} Contours')
    ax.set_xlabel('Frame Index')
    ax.set_ylabel(f'{contour_name} Value')
    ax.legend()

    fig.tight_layout()

    return fig


def plot_contour(contour: torch.Tensor,
                 contour_name: str):
    """Plots a single contour (e.g. pitch/energy/duration) over time."""

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(contour.cpu().numpy(), label=contour_name, color='green')
    ax.set_title(f'{contour_name} Contour')
    ax.set_xlabel('Frame Index')
    ax.set_ylabel(f'{contour_name} Value')
    ax.legend()

    fig.tight_layout()

    return fig

def plot_matrix(matrix: torch.Tensor,
                title: str,
                xlabel: str,
                ylabel: str):
    """Plots a generic matrix with labels."""

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(matrix.cpu().numpy(), origin='lower', interpolation='none')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    return fig

def plot_and_save_spectrogram(spec: np.ndarray,
                              sr: int,
                              hop_length: int,
                              title: str,
                              output_path: pathlib.Path):
    """Plots and saves a single mel-spectrogram."""

    fig, ax = plt.subplots(figsize=(10, 4))

    librosa.display.specshow(spec, x_axis='time', y_axis='mel', sr=sr, hop_length=hop_length, ax=ax)
    ax.set_title(title)

    fig.tight_layout()

    fig.savefig(output_path)
    plt.close(fig)


def plot_and_save_contour(contour: np.ndarray,
                          contour_name: str,
                          output_path: pathlib.Path):
    """Plots and saves a single contour (pitch/energy/duration) over time."""

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(contour, label=contour_name, color='#4A8F8E')
    ax.set_title(f'{contour_name} Contour')
    ax.set_xlabel('Time Index')
    ax.set_ylabel(f'{contour_name}')
    ax.legend()

    fig.tight_layout()

    fig.savefig(output_path)
    plt.close(fig)
