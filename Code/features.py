# -*- coding: utf-8 -*-

# Python standard library imports
import os
import numpy as np

# Third party imports
import librosa
import timbral_models


def compute_multi_features_ssm(audio_filename,
                               song_signal, sample_rate,
                               beats_vector, downbeats_vector,
                               features,
                               save_features=False):
    '''
    Compute features specified by the features list. Features and computed,
    beat-synchronized, organized barwise and normalized. Then, a SSM is
    computed.
    Returns an array of size:
        len(downbeats_vector)*len(downbeats_vector)*(number of features)
    '''

    if save_features:
        # Create directory to save/load features if it does not exist
        dir_fname = os.path.join(
            os.path.dirname(os.path.dirname(audio_filename)),
            'features',
            'CHM_features_SSM')
        if not os.path.exists(dir_fname):
            os.makedirs(dir_fname)

    # # Parameters for normalization
    # NORM_ORDER = 2
    # NORM_THRESHOLD = 1e-3

    multi_features_ssm = np.zeros((len(downbeats_vector) - 1,
                                   len(downbeats_vector) - 1,
                                   len(features)))
    feature_index = 0
    if 'chroma_stft' in features:
        print('\tComputing chroma STFT')
        multi_features_ssm[:, :, feature_index] = compute_chroma_stft(
            song_signal, sample_rate,
            beats_vector, downbeats_vector,
            save_feature=save_features)
        feature_index += 1

    if 'chroma_cqt' in features:
        print('\tComputing constant-Q chroma')
        multi_features_ssm[:, :, feature_index] = compute_constant_q(
            song_signal, sample_rate,
            beats_vector, downbeats_vector,
            save_feature=save_features)
        feature_index += 1

    if 'mfcc' in features:
        print('\tComputing MFCC')
        multi_features_ssm[:, :, feature_index] = compute_mfcc(
            song_signal, sample_rate,
            beats_vector, downbeats_vector,
            save_feature=save_features)
        feature_index += 1

    if 'ac_timbral_features' in features:
        print('\tComputing AudioCommons timbral features')
        multi_features_ssm[:, :, feature_index] = compute_ac_timbral(
            song_signal, sample_rate,
            beats_vector, downbeats_vector,
            save_feature=save_features)

    return multi_features_ssm


def compute_SM_dot(X, Y):
    '''
    Computes similarity matrix from feature sequences using dot (inner) product
    Notebook: C4/C4S2_SSM.ipynb

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    Y : TYPE
        DESCRIPTION.

    Returns
    -------
    S : TYPE
        DESCRIPTION.

    '''
    S = np.dot(np.transpose(Y), X)
    S[np.where(S < 0)] = 0
    return S


# def feature_beat_synchronization(features_matrix, beats_vector,
#                                  frame_times_vector):
#     '''old'''
#     number_of_frames = np.shape(features_matrix)[1]
#     number_of_beats = len(beats_vector)

#     # Averaging matrix
#     downsample_matrix = np.zeros((number_of_frames, number_of_beats - 1))

#     for j in range(number_of_beats-1):
#         downsample_matrix[:, j] = ((beats_vector[j] <= frame_times_vector) &
#                                    (frame_times_vector <= beats_vector[j+1]))

#     # Averaging
#     diag_vector = np.sum(downsample_matrix, axis=0)
#     diag_vector[np.where(diag_vector != 0)[0]] = (
#         1/diag_vector[np.where(diag_vector != 0)[0]])
#     downsample_matrix = np.matmul(downsample_matrix, np.diag(diag_vector))
#     beat_sync_feature_matrix = np.matmul(features_matrix, downsample_matrix)

#     return beat_sync_feature_matrix


def feature_beat_synchronization(feature,
                                 feature_hop_length,
                                 sample_rate,
                                 beats_vector):

    beat_sync_feature = np.zeros((np.size(feature, axis=0),
                                  np.size(beats_vector) - 1))
    for i in range(len(beats_vector) - 1):
        frame_index = round(beats_vector[i]*sample_rate / feature_hop_length)
        next_frame_index = round(beats_vector[i+1]*sample_rate
                                 / feature_hop_length)
        if frame_index == next_frame_index:
            next_frame_index += 1
        beat_sync_feature[:, i] = np.mean(
            feature[:, frame_index:next_frame_index], axis=1)
    # Last beat is NOT computed (this is intended)
    # beat_sync_feature[:, -1] = np.mean(
    #     feature[:, round(beats_vector[-1]*sample_rate/feature_hop_length):],
    #     axis=1)
    return beat_sync_feature


def organize_features_in_bars(beat_sync_feature_matrix, beats_vector,
                              downbeats_vector):
    feature_length = np.shape(beat_sync_feature_matrix)[0]
    [beats_per_bar_vector, start_beats_vector] = beats_per_bar(
        beats_vector, downbeats_vector)[0:2]
    beats_per_bar_vector = beats_per_bar_vector[1:-1]
    nb_bars = len(beats_per_bar_vector)
    bar_lengths_vector = np.unique(beats_per_bar_vector)
    nb_bar_length = len(bar_lengths_vector)
    nb_bars_per_length = np.zeros(nb_bar_length)

    for i in range(nb_bar_length):
        nb_bars_per_length[i] = sum(beats_per_bar_vector ==
                                    bar_lengths_vector[i])

    # Compare different meters
    width = int(sum(feature_length*bar_lengths_vector))
    bar_features_matrix = np.zeros((width, nb_bars))
    for n_bar in range(nb_bars):
        for n_beat in range(1, int(beats_per_bar_vector[n_bar] + 1)):
            bar_features_matrix[(feature_length*(n_beat - 1)):
                                (feature_length*n_beat), n_bar] = (
                                    beat_sync_feature_matrix[
                                        :, int((start_beats_vector[n_bar]
                                                + n_beat - 2))])
    return bar_features_matrix


def beats_per_bar(beats_vector, downbeats_vector):

    for downbeat in downbeats_vector:
        if not any(beats_vector == downbeat):
            return 'Error : every downbeat should be a beat !\n'

    nb_bars = len(downbeats_vector) - 1
    nb_beats = len(beats_vector)

    beats_per_bar_vector = np.zeros(nb_bars + 2)
    downbeat_index_vector = np.zeros(nb_bars + 1)

    n_bar = 0
    for i in range(nb_beats):
        if n_bar < nb_bars + 1:
            if beats_vector[i] == downbeats_vector[n_bar]:
                downbeat_index_vector[n_bar] = i
                n_bar += 1
        beats_per_bar_vector[n_bar] = beats_per_bar_vector[n_bar] + 1

    start_beats_vector = downbeat_index_vector[:-1]
    stop_beats_vector = downbeat_index_vector[1:] - 1

    beats_per_bar_matrix = np.zeros((nb_bars, nb_beats))
    for n_bar in range(nb_bars):
        beats_per_bar_matrix[int(start_beats_vector[n_bar]):
                             int(stop_beats_vector[n_bar])] = 1

    return (beats_per_bar_vector, start_beats_vector,
            stop_beats_vector, beats_per_bar_matrix)


def compute_chroma_stft(song_signal, sample_rate,
                        beats_vector, downbeats_vector,
                        save_feature=False):
    NORM_ORDER = 2
    NORM_THRESHOLD = 1e-3
    # STFT Chroma parameters
    CHROMA_N_FFT = 4410
    CHROMA_HOP_LENGTH = 2205
    # !!! Test
    # CHROMA_N_FFT = round(np.mean(np.diff(beats_vector))*sample_rate)
    # CHROMA_HOP_LENGTH = round(CHROMA_N_FFT/2)

    chroma_matrix = librosa.feature.chroma_stft(
        song_signal, sample_rate, n_fft=CHROMA_N_FFT,
        hop_length=CHROMA_HOP_LENGTH, norm=None)

    # Beat synchronization
    beat_sync_chroma_matrix = feature_beat_synchronization(chroma_matrix,
                                                           CHROMA_HOP_LENGTH,
                                                           sample_rate,
                                                           beats_vector)

    # Barwise organization
    barwise_chroma_matrix = organize_features_in_bars(
        beat_sync_chroma_matrix, beats_vector, downbeats_vector)

    # Normalization
    barwise_chroma_matrix = librosa.util.normalize(
        barwise_chroma_matrix, norm=NORM_ORDER, threshold=NORM_THRESHOLD)

    # Self-Similarity Matrix computation
    chroma_ssm = compute_SM_dot(barwise_chroma_matrix,
                                barwise_chroma_matrix)

    return chroma_ssm


def compute_constant_q(song_signal, sample_rate,
                       beats_vector, downbeats_vector,
                       save_feature=False):
    # Parameters for normalization
    NORM_ORDER = 2
    NORM_THRESHOLD = 1e-3

    # Constant-Q Chroma parameters
    CHROMA_HOP_LENGTH = 2205
    chroma_matrix = librosa.feature.chroma_cqt(
        song_signal, sample_rate,
        hop_length=CHROMA_HOP_LENGTH, norm=None)

    # Beat synchronization
    beat_sync_chroma_matrix = feature_beat_synchronization(chroma_matrix,
                                                           CHROMA_HOP_LENGTH,
                                                           sample_rate,
                                                           beats_vector)

    # Barwise organization
    barwise_chroma_matrix = organize_features_in_bars(
        beat_sync_chroma_matrix, beats_vector, downbeats_vector)

    # Normalization
    barwise_chroma_matrix = librosa.util.normalize(
        barwise_chroma_matrix, norm=NORM_ORDER,
        threshold=NORM_THRESHOLD)

    # Self-Similarity Matrix computation
    chroma_ssm = compute_SM_dot(barwise_chroma_matrix,
                                barwise_chroma_matrix)

    return chroma_ssm


def compute_mfcc(song_signal, sample_rate,
                 beats_vector, downbeats_vector,
                 save_feature=False):
    # Parameters for normalization
    NORM_ORDER = 2
    NORM_THRESHOLD = 1e-3

    # MFCC parameters
    # N_MFCC = 13
    N_MFCC = 12
    MFCC_N_FFT = int(0.025*sample_rate)  # 0.025 seconds
    MFCC_HOP_LENGTH = int(0.01*sample_rate)  # 0.01 seconds
    mfcc_matrix = librosa.feature.mfcc(song_signal, sample_rate,
                                       n_mfcc=N_MFCC,
                                       n_fft=MFCC_N_FFT,
                                       hop_length=MFCC_HOP_LENGTH,
                                       lifter=22*N_MFCC,
                                       # fmax=4000,
                                       norm=None)[1:]
    # Beat synchronization
    beat_sync_mfcc_matrix = feature_beat_synchronization(mfcc_matrix,
                                                         MFCC_HOP_LENGTH,
                                                         sample_rate,
                                                         beats_vector)

    # Barwise organization
    barwise_mfcc_matrix = organize_features_in_bars(beat_sync_mfcc_matrix,
                                                    beats_vector,
                                                    downbeats_vector)

    # Normalization
    barwise_mfcc_matrix = librosa.util.normalize(
        barwise_mfcc_matrix, norm=NORM_ORDER,
        threshold=NORM_THRESHOLD)

    # Self-Similarity Matrix computation
    mfcc_ssm = compute_SM_dot(barwise_mfcc_matrix, barwise_mfcc_matrix)

    return mfcc_ssm


def compute_ac_timbral(song_signal, sample_rate,
                       beats_vector, downbeats_vector,
                       save_feature=False):

    # Parameters for normalization
    NORM_ORDER = 2
    NORM_THRESHOLD = 1e-3

    ac_features_matrix = np.zeros((7, len(beats_vector)))

    for beat_idx in range(len(beats_vector)-1):
        b1 = beats_vector[beat_idx]
        b2 = beats_vector[beat_idx+1]
        patch = song_signal[int(b1*sample_rate):int(b2*sample_rate)]

        hardness = timbral_models.timbral_hardness(patch,
                                                   fs=sample_rate)
        depth = timbral_models.timbral_depth(patch,
                                             fs=sample_rate)
        brightness = timbral_models.timbral_brightness(patch,
                                                       fs=sample_rate)
        roughness = timbral_models.timbral_roughness(patch,
                                                     fs=sample_rate)
        warmth = timbral_models.timbral_warmth(patch,
                                               fs=sample_rate)
        sharpness = timbral_models.timbral_sharpness(patch,
                                                     fs=sample_rate)
        boominess = timbral_models.timbral_booming(patch,
                                                   fs=sample_rate)

        ac_features_matrix[:, beat_idx] = np.array([
            hardness if np.isfinite(hardness) else 0,
            depth if np.isfinite(depth) else 0,
            brightness if np.isfinite(brightness) else 0,
            roughness if np.isfinite(roughness) else 0,
            warmth if np.isfinite(warmth) else 0,
            sharpness if np.isfinite(sharpness) else 0,
            boominess if np.isfinite(boominess) else 0
            ])
    # Last beat
    patch = song_signal[int(beats_vector[-1]*sample_rate):]
    hardness = timbral_models.timbral_hardness(patch, fs=sample_rate)
    depth = timbral_models.timbral_depth(patch, fs=sample_rate)
    brightness = timbral_models.timbral_brightness(patch,
                                                   fs=sample_rate)
    roughness = timbral_models.timbral_roughness(patch, fs=sample_rate)
    warmth = timbral_models.timbral_warmth(patch, fs=sample_rate)
    sharpness = timbral_models.timbral_sharpness(patch, fs=sample_rate)
    boominess = timbral_models.timbral_booming(patch, fs=sample_rate)

    ac_features_matrix[:, beat_idx] = np.array([
        hardness if np.isfinite(hardness) else 0,
        depth if np.isfinite(depth) else 0,
        brightness if np.isfinite(brightness) else 0,
        roughness if np.isfinite(roughness) else 0,
        warmth if np.isfinite(warmth) else 0,
        sharpness if np.isfinite(sharpness) else 0,
        boominess if np.isfinite(boominess) else 0
        ])

    # Barwise organisation
    barwise_ac_features_matrix = organize_features_in_bars(
        ac_features_matrix, beats_vector, downbeats_vector)

    # Normalization
    barwise_ac_features_matrix = librosa.util.normalize(
        barwise_ac_features_matrix,
        norm=NORM_ORDER,
        threshold=NORM_THRESHOLD)

    # Self-Similarity Matrix computation
    ac_timbral_ssm = librosa.segment.recurrence_matrix(
        barwise_ac_features_matrix, mode='affinity',
        metric='euclidean')

    return ac_timbral_ssm
