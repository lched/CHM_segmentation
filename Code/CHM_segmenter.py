# -*- coding: utf-8 -*-

import os
import numpy as np
import six
import librosa
import madmom
import timbral_models


def segmenter(audio_filename, sample_rate=22050,
              beats_file=None,
              features=['chroma_stft', 'mfcc'],
              criteria=['repetition', 'novelty'],
              coefficients=[1, 1, 1, 1]):
    '''
    Estimate segments boundaries of a song. Does not estimate labels Based on :
    C. Gaudefroy H. Papadopoulos and M. Kowalski, “A Multi-Dimensional
    Meter-Adaptive Method For Automatic Segmentation Of Music”, in CBMI 2015.
    Default configuration should reproduce the method presented in this paper.

    The audio file should be in a dataset organized as follows (consistent
    with the Music Structure Analysis Framework):
        ./dataset_folder
            /audio
            /estimations (empty initially)
            /features (empty initially)
            /references
    To analyse a single file outside of a dataset, segment_full_analysis.py
    shoud be used instaed.

    Parameters
    ----------
    audio_filename : path, string
         Musical file to segment. Any codec supported by soundfile or
         audioread will work.
    sample_rate : number>0, optional
        Audio will be automatically resampled to the given rate
        (default sr=22050).‘None’ uses the native sampling rate.
    beats_file : path or numpy array, optional
        If 'None', an estimation of beats will be computed using madmom
        and saved. Sinon, on peut fournir une chaine de caractères vers un
        fichier au format mirex contenant beats et leur place dans la mesure
        (downbeats pour = 1), ou tuple contenant deux listes, une pour les
        beats et une pour les downbeats
    features : list of strings, optional
        List of features that should be used for segmentation. The list can
        contain 'chroma_stft', 'chroma_cqt', 'mfcc',
        'AudioCommons_timbral_features'.
        The default is ['chroma_stft', 'mfcc'].
    criteria : list of strings, optional
        List of criterion that should be used for segmentation. The list can
        contain 'repetition', 'novelty' and 'homogeneity(TODO)'.
        The default is ['repetition', 'novelty'].
    coefficients : list, optional
        Coefficients to apply weights to the different descriptors and
        associated methods. Coefficients must be given like this:
            [feature1_criterion1, feature1_criterion2, ..., feature2criterion1,
             feature2criterion2, ...]
        The default is [1, 1, 1, 1].

    Raises
    ------
    OSError
        Raised if segmenter could not open a file.
    ValueError
        Raised if segmenter is given a bad datatype for beats_file.

    Returns
    -------
    estimated_segments
        Estimated segments boundaries of the provided song in a format
        compatible with mir_eval.

    '''
    # =============================================================================
    # Get audio
    # =============================================================================
    song_signal, sample_rate = librosa.load(audio_filename, sr=sample_rate)

    # =============================================================================
    #  Get beats and downbeats
    # =============================================================================
    if beats_file is None:
        beats_file = os.path.join(
            os.path.dirname(os.path.dirname(audio_filename)),
            'estimations',
            'beats_and_downbeats',
            os.path.splitext(os.path.basename(audio_filename))[0]
            + '_madmom_estimated_beats.lab')
        try:
            # Try to open beats from file if already computed
            beats_vector = madmom.io.load_beats(beats_file)
            downbeats_vector = madmom.io.load_beats(beats_file, downbeats=True)
            print('\tBeats and downbeats already computed, loaded from file')
            # print('\tBeats already computed, loaded from file %s'
            #       % beats_file)
        except OSError:
            print('\tComputing beats using madmom...')
            beats_vector = estimate_beats_madmom_1(audio_filename)
            print('\tComputing downbeats using madmom')
            downbeats_vector = estimate_downbeats_madmom_1(audio_filename,
                                                           beats_vector)
            # Save results to .lab file
            save_beats(beats_vector, downbeats_vector, beats_file)
    else:
        if isinstance(beats_file, six.string_types):
            try:
                beats_vector = madmom.io.load_beats(beats_file)
                downbeats_vector = madmom.io.load_beats(beats_file,
                                                        downbeats=True)
                print('\tBeats and downbeats successfully  loaded from file:'
                      '{}'.format(beats_file))
            except OSError:
                print('Beats file failed to load: {}'.format(beats_file))
                raise OSError('Unable to read beats file.')
        elif isinstance(beats_file, tuple):
            beats_vector = beats_file[0]
            downbeats_vector = beats_file[1]
        else:
            raise ValueError('Input beats and downbeats must be either a '
                             'filename or a tuple of numpy arrays')

    # Divide beats in two
    beats_length_vector = (np.array(beats_vector[1:])
                           - np.array(beats_vector[:-1]))
    beats_vector = np.concatenate((beats_vector, beats_vector[:-1]
                                   + beats_length_vector/2))
    beats_vector = np.sort(beats_vector, axis=None)

    # =============================================================================
    # Features computation
    # =============================================================================
    multi_features_ssm = compute_multi_features_ssm(audio_filename,
                                                    song_signal, sample_rate,
                                                    beats_vector,
                                                    downbeats_vector,
                                                    features)

    # =============================================================================
    # Segmentation
    # =============================================================================
    # Check if there are enough coefficients
    if len(coefficients) != len(criteria)*len(features):
        raise ValueError('The number of provided coefficients must be equal to'
                         'the number of features * the number of criteria')

    segmentation_criterion_matrix = np.zeros((len(criteria)*len(features),
                                              len(downbeats_vector)-2))
    criterion_index = 0
    # Compute criteria
    if 'repetition' in criteria:
        for feature_index in range(len(features)):
            segmentation_criterion_matrix[criterion_index, :] = (
                compute_repetition_criterion(multi_features_ssm[:, :, feature_index]))
            criterion_index += 1

    if 'novelty' in criteria:
        for feature_index in range(len(features)):
            segmentation_criterion_matrix[criterion_index, :] = (
                compute_novelty_criterion(multi_features_ssm[:, :, feature_index]))
            criterion_index += 1

    segmentation_criterion_vector = np.sum(
        (segmentation_criterion_matrix
         * np.array(coefficients)[:, np.newaxis]), axis=0)

    song_duration = len(song_signal)/sample_rate
    estimated_boundaries_vector = select_boundaries(
        segmentation_criterion_vector, downbeats_vector, song_duration)

    # =============================================================================
    # Save results
    # =============================================================================
    estimated_boundaries_fname = os.path.join(
        os.path.dirname(os.path.dirname(audio_filename)),
        'estimations',
        'segments',
        os.path.splitext(os.path.basename(audio_filename))[0]
        )
    save_segments(estimated_boundaries_vector,
                  estimated_boundaries_fname)

    estimated_segments = (np.zeros((len(estimated_boundaries_vector)-1, 2)),
                          [])
    for i in range(len(estimated_boundaries_vector) - 1):
        estimated_segments[0][i, 0] = estimated_boundaries_vector[i]
        estimated_segments[0][i, 1] = estimated_boundaries_vector[i+1]
        estimated_segments[1].append('Segment{}'.format(i+1))

    return estimated_segments


def compute_multi_features_ssm(audio_filename,
                               song_signal, sample_rate,
                               beats_vector, downbeats_vector,
                               features):
    '''
    Compute features specified by the features list. Features and computed,
    beat-synchronized, organized barwise and normalized. Then, a SSM is
    computed.
    Returns an array of size:
        len(downbeats_vector)*len(downbeats_vector)*(number of features)
    '''
    # Create directory to save/load features if it does not exist
    dir_fname = os.path.join(os.path.dirname(os.path.dirname(audio_filename)),
                             'features',
                             'CHM_features_SSM')
    if not os.path.exists(dir_fname):
        os.makedirs(dir_fname)

    # Parameters for normalization
    NORM_ORDER = 2
    NORM_THRESHOLD = 1e-3
    multi_features_ssm = np.zeros((len(downbeats_vector) - 1,
                                   len(downbeats_vector) - 1,
                                   len(features)))
    feature_index = 0
    if 'chroma_stft' in features:
        chroma_stft_file = os.path.join(
            os.path.dirname(os.path.dirname(audio_filename)),
            'features',
            'CHM_features_SSM',
            os.path.splitext(os.path.basename(audio_filename))[0]
            + '_chroma_stft.txt')
        try:
            chroma_ssm = np.loadtxt(chroma_stft_file)
            multi_features_ssm[:, :, feature_index] = chroma_ssm
            feature_index += 1
            print('\tChroma STFT SSM already computed, loaded from file')
        except (OSError, ValueError) as error:
            if error == ValueError:
                print('Saved SSM matrix is not consistent with given downbeats'
                      'file. It will not be used.')
            print('\tComputing chroma STFT')
            # STFT Chroma parameters
            CHROMA_N_FFT = 4410
            CHROMA_HOP_LENGTH = 2205
            chroma_matrix = librosa.feature.chroma_stft(
                song_signal, sample_rate, n_fft=CHROMA_N_FFT,
                hop_length=CHROMA_HOP_LENGTH, norm=None)

            # Beat synchronization
            CHROMA_FEATURE_RATE = sample_rate/(CHROMA_N_FFT
                                               - CHROMA_HOP_LENGTH)
            dt = 1/CHROMA_FEATURE_RATE
            frame_times_vector = np.zeros(np.shape(chroma_matrix)[1]+1)

            for i in range(np.shape(chroma_matrix)[1]):
                frame_times_vector[i] = min(i*dt,
                                            (len(song_signal)-1)/sample_rate)
            frame_times_vector[-1] = frame_times_vector[-2]
            frame_mid_vector = (frame_times_vector[0:-1]
                                + frame_times_vector[1:])/2

            beat_sync_chroma_matrix = feature_beat_synchronization(
                chroma_matrix, beats_vector, frame_mid_vector)

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

            # Saving SSM to file
            np.savetxt(chroma_stft_file, chroma_ssm)

            multi_features_ssm[:, :, feature_index] = chroma_ssm
            feature_index += 1

    if 'chroma_cqt' in features:
        chroma_cqt_file = os.path.join(
            os.path.dirname(os.path.dirname(audio_filename)),
            'features',
            'CHM_features_SSM',
            os.path.splitext(os.path.basename(audio_filename))[0]
            + '_chroma_cqt.txt')
        try:
            chroma_ssm = np.loadtxt(chroma_cqt_file)
            multi_features_ssm[:, :, feature_index] = chroma_ssm
            feature_index += 1
            print('\tConstant-Q chroma SSM already computed, loaded from file')
        except (OSError, ValueError) as error:
            if error == ValueError:
                print('Saved SSM matrix is not consistent with given downbeats'
                      'file. It will not be used.')
            print('\tComputing constant-Q chroma')
            # Constant-Q Chroma parameters
            CHROMA_N_FFT = 4410
            CHROMA_HOP_LENGTH = 2205
            chroma_matrix = librosa.feature.chroma_cqt(
                song_signal, sample_rate,
                hop_length=CHROMA_HOP_LENGTH, norm=None)

            # Beat synchronization
            CHROMA_FEATURE_RATE = sample_rate/(CHROMA_N_FFT
                                               - CHROMA_HOP_LENGTH)
            dt = 1/CHROMA_FEATURE_RATE
            frame_times_vector = np.zeros(np.shape(chroma_matrix)[1]+1)

            for i in range(np.shape(chroma_matrix)[1]):
                frame_times_vector[i] = min(i*dt,
                                            (len(song_signal)-1)/sample_rate)
            frame_times_vector[-1] = frame_times_vector[-2]
            frame_mid_vector = (frame_times_vector[0:-1]
                                + frame_times_vector[1:])/2

            beat_sync_chroma_matrix = feature_beat_synchronization(
                chroma_matrix, beats_vector, frame_mid_vector)

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

            # Saving SSM to file
            np.savetxt(chroma_cqt_file, chroma_ssm)

    if 'mfcc' in features:
        mfcc_file = os.path.join(
            os.path.dirname(os.path.dirname(audio_filename)),
            'features',
            'CHM_features_SSM',
            os.path.splitext(os.path.basename(audio_filename))[0]
            + '_mfcc.txt')
        try:
            mfcc_ssm = np.loadtxt(mfcc_file)
            multi_features_ssm[:, :, feature_index] = mfcc_ssm
            feature_index += 1
            print('\tMFCC SSM already computed, loaded from file')
        except (OSError, ValueError) as error:
            if error == ValueError:
                print('Saved SSM matrix is not consistent with given downbeats'
                      'file. It will not be used.')
            print('\tComputing MFCC')
            N_MFCC = 13
            MFCC_N_FFT = int(0.025*sample_rate)  # 0.025 seconds
            MFCC_HOP_LENGTH = int(0.01*sample_rate)  # 0.01 seconds
            mfcc_matrix = librosa.feature.mfcc(song_signal,
                                               sample_rate,
                                               n_mfcc=N_MFCC,
                                               n_fft=MFCC_N_FFT,
                                               hop_length=MFCC_HOP_LENGTH,
                                               lifter=22*N_MFCC,
                                               fmax=4000,
                                               norm=None)[1:]
            # Beat synchronization
            num_frames = np.shape(mfcc_matrix)[1]
            dt = len(song_signal)/sample_rate/num_frames
            frame_mid_vector = np.linspace(dt, num_frames*dt, num_frames)

            beat_sync_mfcc_matrix = feature_beat_synchronization(
                mfcc_matrix, beats_vector, frame_mid_vector)

            # Barwise organisation
            barwise_mfcc_matrix = organize_features_in_bars(
                beat_sync_mfcc_matrix, beats_vector, downbeats_vector)

            # Normalization
            barwise_mfcc_matrix = librosa.util.normalize(
                barwise_mfcc_matrix, norm=NORM_ORDER,
                threshold=NORM_THRESHOLD)

            # Self-Similarity Matrix computation
            mfcc_ssm = compute_SM_dot(barwise_mfcc_matrix, barwise_mfcc_matrix)

            # Saving SSM to file
            np.savetxt(mfcc_file, mfcc_ssm)

            multi_features_ssm[:, :, feature_index] = mfcc_ssm
            feature_index += 1

    if 'ac_timbral_features' in features:
        # changer nom de ac_timbral_features
        ac_timbral_features_file = os.path.join(
            os.path.dirname(os.path.dirname(audio_filename)),
            'features',
            'CHM_features_SSM',
            os.path.splitext(os.path.basename(audio_filename))[0]
            + '_ac_timbral_features.txt')
        try:
            ac_timbral_ssm = np.loadtxt(ac_timbral_features_file)
            multi_features_ssm[:, :, feature_index] = ac_timbral_ssm
            print('\tAudioCommons features SSM already computed, loaded '
                  'from file')
        except (OSError, ValueError) as error:
            if error == ValueError:
                print('Saved SSM matrix is not consistent with given downbeats'
                      'file. It will not be used.')
            print('\tComputing AudioCommons timbral features')
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

            # Saving SSM to file
            np.savetxt(ac_timbral_features_file, ac_timbral_ssm)

            multi_features_ssm[:, :, feature_index] = ac_timbral_ssm

    return multi_features_ssm


def compute_SM_dot(X, Y):
    """
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

    """
    S = np.dot(np.transpose(Y), X)
    S[np.where(S < 0)] = 0
    return S


def check_beats_downbeats_alignment(beats_vector, downbeats_vector):
    """
    check_beats_downbeats_alignment checks if a set of beats and downbeats
    annotations match

    Parameters
    ----------
    beats_vector : numpy array
        Contains beats.
    downbeats_vector : numpy array
        Contains downbeats.

    Returns
    -------
    None.

    """
    max_error = 0
    db_closest_beat_vector = np.zeros(np.size(downbeats_vector))

    for i in range(len(downbeats_vector)):
        if not(downbeats_vector[i] in beats_vector):
            min_ind = np.argmin(abs(beats_vector - downbeats_vector[i]))
            min_val = min(abs(beats_vector - downbeats_vector[i]))
            max_error = max(max_error, min_val)
            db_closest_beat_vector[i] = beats_vector[min_ind]
        else:
            db_closest_beat_vector[i] = downbeats_vector[i]

    return max_error, db_closest_beat_vector


def feature_beat_synchronization(features_matrix, beats_vector,
                                 frame_times_vector):
    """

    Parameters
    ----------
    features_matrix : TYPE
        features for each time frame.
    beats_vector : TYPE
        times of the tactuses (seconds).
    frame_times_vector : TYPE
        time for which the features in features_m stand.

    Returns
    -------
     beat_synchronous_features_matrix  : resulting downsampled features

    """
    number_of_frames = np.shape(features_matrix)[1]
    number_of_beats = len(beats_vector)

    # Averaging matrix
    downsample_matrix = np.zeros((number_of_frames, number_of_beats - 1))

    for j in range(number_of_beats-1):
        downsample_matrix[:, j] = ((beats_vector[j] <= frame_times_vector) &
                                   (frame_times_vector <= beats_vector[j+1]))

    # Averaging
    diag_vector = np.sum(downsample_matrix, axis=0)
    diag_vector[np.where(diag_vector != 0)[0]] = (
        1/diag_vector[np.where(diag_vector != 0)[0]])
    downsample_matrix = np.matmul(downsample_matrix, np.diag(diag_vector))
    beat_sync_feature_matrix = np.matmul(features_matrix, downsample_matrix)

    return beat_sync_feature_matrix


def organize_features_in_bars(beat_sync_feature_matrix, beats_vector,
                              downbeats_vector):
    """
    NOTATIONS A CLARIFIER

    Parameters
    ----------
    beat_sync_feature_matrix : TYPE
        DESCRIPTION.
    beats_vector : TYPE
        DESCRIPTION.
    downbeats_vector : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

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
    """


    Parameters
    ----------
    beats_vector : TYPE
        DESCRIPTION.
    downbeats_vector : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    for downbeat in downbeats_vector:
        if not(any(beats_vector == downbeat)):
            return('Error : every downbeat should be a beat !\n')

    nb_bars = len(downbeats_vector) - 1
    nb_beats = len(beats_vector)

    beats_per_bar_vector = np.zeros(nb_bars + 2)
    downbeat_index_vector = np.zeros(nb_bars + 1)

    n_bar = 0
    for i in range(nb_beats):
        if (n_bar < nb_bars + 1):
            if (beats_vector[i] == downbeats_vector[n_bar]):
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


def compute_repetition_criterion(feature_SSM):
    """


    Parameters
    ----------
    feature_SSM : TYPE
        DESCRIPTION.

    Returns
    -------
    repetition_criterion_vector : TYPE
        DESCRIPTION.

    """

    SSM_length = len(feature_SSM)

    # Time-lag matrix
    time_lag_matrix = np.zeros((SSM_length, SSM_length))
    for i in range(SSM_length):
        for j in range(i+1):
            time_lag_matrix[i, j] = feature_SSM[i-j, i]

    smooth_time_lag_matrix = np.zeros((SSM_length, SSM_length))
    for lag in range(SSM_length):
        smooth_time_lag_matrix[lag:SSM_length, lag] = cancel_peaks(
            time_lag_matrix[lag:SSM_length, lag])

    # Mean value for each lag
    time_lag_mean_values = np.zeros(SSM_length)
    for i in range(SSM_length):
        time_lag_mean_values[i] = np.mean(
            smooth_time_lag_matrix[i:SSM_length, i])

    # Select the highest values among the peaks
    mean_lag_peaks_vector = select_peaks(time_lag_mean_values)
    lag_peaks_threshold = otsu_thresholding(mean_lag_peaks_vector[np.nonzero(
        mean_lag_peaks_vector)[0]])
    selected_lags_vector = mean_lag_peaks_vector > lag_peaks_threshold

    repetition_criterion_vector = np.zeros(SSM_length-1)

    for lag in np.where(selected_lags_vector)[0]:
        similarity_vector = smooth_time_lag_matrix[lag:SSM_length, lag]
        repetition_criterion_vector[lag:(SSM_length-1)] = np.maximum(
            repetition_criterion_vector[lag:(SSM_length-1)],
            abs(np.diff(similarity_vector)))
        repetition_criterion_vector[:(SSM_length-lag-1)] = np.maximum(
            repetition_criterion_vector[:(SSM_length-lag-1)],
            abs(np.diff(similarity_vector)))

    return repetition_criterion_vector


def cancel_peaks(vector):
    """
    CANCEL_PEAKS smoothens array vector by clearing its local extrema.

    Parameters
    ----------
    vector : array

    Returns
    -------
    smooth_vector : array

    """

    smooth_vector = vector

    for i in range(1, len(vector)-1):
        diff_last = smooth_vector[i] - smooth_vector[i-1]
        diff_next = smooth_vector[i+1] - smooth_vector[i]
        if ((diff_last > 0) & (diff_next < 0)):
            smooth_vector[i] = smooth_vector[i] - min(diff_last, -diff_next)
        elif ((diff_last < 0) & (diff_next > 0)):
            smooth_vector[i] = smooth_vector[i] + min(-diff_last, diff_next)

    return smooth_vector


def select_peaks(vector):
    """
    SELECT_PEAKS picks the peaks in vector (puts every non-local maximum value
    to zero).

    Parameters
    ----------
    vector : array

    Returns
    -------
    peaks_vector

    """

    peaks_vector = np.zeros(len(vector))
    for i in range(1, len(vector)-1):
        if((vector[i] > vector[i-1]) & (vector[i] >= vector[i+1])):
            peaks_vector[i] = vector[i]

    return peaks_vector


def otsu_thresholding(data, number_of_bins=10):
    """


    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    number_of_bins : TYPE, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    level : TYPE
        DESCRIPTION.

    """

    # Convert data to vector
    data = np.reshape(data, np.size(data), order='F')
    histogram = np.histogram(data, number_of_bins)[0]/np.size(data)
    mu = sum(histogram*np.array(range(1, number_of_bins+1)))

    omega0_vector = np.zeros(number_of_bins-1)
    mu0_vector = np.zeros(number_of_bins-1)

    for k in range(number_of_bins-1):
        omega0_vector[k] = sum(histogram[:k])
        mu0_vector[k] = sum(histogram[:k]*np.array(range(1, k+1)))

    omega1_vector = 1 - omega0_vector
    mu1_vector = (mu - omega0_vector*mu0_vector)/omega1_vector
    sigmaB_vector = omega0_vector*omega1_vector*(mu1_vector-mu0_vector)

    criterion = sigmaB_vector
    k_opt = np.argmax(criterion)
    level = np.quantile(data, omega0_vector[k_opt])

    return level


def compute_novelty_criterion(feature_SSM):
    """


    Parameters
    ----------
    feature_SSM : TYPE
        DESCRIPTION.

    Returns
    -------
    novelty_criterion_vector : TYPE
        DESCRIPTION.

    """

    novelty_criterion_vector = np.zeros(len(feature_SSM) - 1)

    for kernel_size in [4, 8, 12, 16, 32]:
        if kernel_size < len(feature_SSM):
            for kernel_type in ['H-H', 'N-H', 'H-N']:
                novelty_vector, min_val, max_val = compute_novelty_curve(
                    feature_SSM, kernel_size, kernel_type)
                novelty_norm_vector = (novelty_vector-min_val)/(max_val -
                                                                min_val)
                novelty_criterion_vector = np.maximum(novelty_criterion_vector,
                                                      novelty_norm_vector)
    return novelty_criterion_vector


def compute_novelty_curve(feature_SSM, kernel_size=64,
                          kernel_type='H-H',
                          kernel_balancing='after'):
    """


    Parameters
    ----------
    feature_SSM : TYPE
        DESCRIPTION.
    kernel_size : TYPE, optional
        DESCRIPTION. The default is 64.
    kernel_type : TYPE, optional
        DESCRIPTION. The default is 'H-H'.
    kernel_balancing : TYPE, optional
        DESCRIPTION. The default is 'after'.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    novelty_vector : TYPE
        DESCRIPTION.
    min_val : TYPE
        DESCRIPTION.
    max_val : TYPE
        DESCRIPTION.

    """

    if kernel_type == 'H-H':
        kernel_m, min_val, max_val = checker_gauss(kernel_size)
    elif kernel_type == 'H-N':
        kernel_m, min_val, max_val = block_diag_gauss(kernel_size,
                                                      kernel_balancing)
    elif kernel_type == 'N-H':
        kernel_m, min_val, max_val = block_diag_gauss(kernel_size,
                                                      kernel_balancing)
        kernel_m = kernel_m[-1::-1, -1::-1]
    else:
        raise ValueError("Error : The parameter.kernel.type must be either "
                         "''H-H'', ''H-N'' or ''N-H''.\n")

    if (len(feature_SSM) != len(feature_SSM[0])):
        raise ValueError('Error : A SSM is expected to be square! \n')

    hN = int(np.ceil(kernel_size/2))
    padded_ssm = np.ones((len(feature_SSM) + kernel_size,
                          len(feature_SSM) + kernel_size))
    padded_ssm = padded_ssm*np.mean(np.mean(feature_SSM))  # to avoid
    # artificial peaks
    padded_ssm[hN:len(feature_SSM)+hN, hN:len(feature_SSM)+hN] = feature_SSM

    novelty_vector = np.zeros((len(feature_SSM) - 1))  # values for transitions
    # between features
    for i in range(len(feature_SSM)-1):
        novelty_vector[i] = sum(sum(kernel_m*padded_ssm[1+i:kernel_size+i+1,
                                                        1+i:kernel_size+i+1]))

    return novelty_vector, min_val, max_val


def checker_gauss(N):
    """
    checker_gauss returns a checkerboard-like matrix of type [1 -1; -1 1]
    weighted by a gaussian.

    Parameters
    ----------
    N : int
        size of the matrix

    Returns
    -------
    y : TYPE
        DESCRIPTION.
    min_val : TYPE
        DESCRIPTION.
    max_val : TYPE
        DESCRIPTION.

    """
    hN = int(np.ceil(N/2))
    y = np.zeros((N, N))
    # Upper left (the biggest corner in case N is even)
    for i in range(1, hN + 1):
        for j in range(1, hN + 1):
            y[i-1, j-1] = np.exp(-(((i-hN)/hN)**2 + (((j-hN)/hN)**2))*4)

    # The other corners are defined by symetry
    if (N % 2 == 0):
        # Lower right
        y[hN:, hN:] = y[hN-1::-1, hN-1::-1]
        # Upper right
        y[0:hN, hN:] = -y[0:hN, hN-1::-1]
        # Lower left
        y[hN:, :hN] = -y[hN-1::-1, 0:hN]
    else:
        # Lower right
        y[hN:, hN:] = y[hN-1:0:-1, hN-1:0:-1]
        # Upper right
        y[0:hN, hN:] = -y[0:hN, hN-1:0:-1]
        # Lower left
        y[hN:, :hN] = -y[hN-1:0:-1, 0:hN]

    min_val = min(sum(np.diag(y)), 0)
    max_val = sum(y[np.where(y > 0)])

    return y, min_val, max_val


def block_diag_gauss(N, balancing):
    """
    % block_diag_gauss returns a homogeneous / non homogeneous transition
    % kernel such as proposed by Kaiser & Peeters (2013).
    %
    % INPUT
    % - N           : size of the matrix
    % - balancing   : 'before' the gaussian weighting (as K. & P. do), default
    %                 'after' the gaussian weighting (so that the final mean
                                                      is actually zero)
    %
    % OUTPUT
    % - y

    Parameters
    ----------
    N : TYPE
        DESCRIPTION.
    balancing : TYPE
        DESCRIPTION.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    y : TYPE
        DESCRIPTION.
    min_val : TYPE
        DESCRIPTION.
    max_val : TYPE
        DESCRIPTION.

    """

    hpN = int(np.ceil(N/2))
    hmN = int(np.floor(N/2))
    y = np.zeros((N, N))

    if balancing == 'before':
        kappa = 1
        nu = -kappa*(hpN**2 + hmN)/(hmN*(2*hpN + hmN - 1))
        for i in range(1, N+1):
            for j in range(1, N+1):
                g = np.exp(-(((i-hpN)/hpN)**2 + (((j-hpN)/hpN)**2))*4)
                if (((i <= hpN) & (j <= hpN)) | (i == j)):
                    y[i-1, j-1] = kappa*g
                else:
                    y[i-1, j-1] = nu*g

    elif balancing == 'after':
        g = np.zeros((hpN, hpN))
        for i in range(1, hpN+1):
            for j in range(1, hpN+1):
                g[i - 1, j - 1] = np.exp(-(((i - hpN)/hpN)**2 +
                                           (((j - hpN)/hpN)**2))*4)

        sum_p_g = np.sum(g)
        sum_m_g = np.sum(g[:hmN, :hmN])
        dg = np.diag(g)
        sum_d_g = sum(dg[:hmN])
        sum_s_g = np.sum(g[:hmN, :hmN])
        kappa = 1
        nu = -kappa*(sum_p_g + sum_d_g)/(2*sum_s_g + sum_m_g - sum_d_g)
        for i in range(1, N+1):
            for j in range(1, N+1):
                if (((i <= hpN) & (j <= hpN)) | (i == j)):
                    y[i-1, j-1] = kappa*g[min(i, N - i + 1)-1,
                                          min(j, N - j + 1)-1]
                else:
                    y[i-1, j-1] = nu*g[min(i, N - i + 1)-1,
                                       min(j, N - j + 1)-1]

    else:
        raise ValueError('Error : The balancing option must be either ''before'
                         ' or ''after''.\n')

    min_val = min(sum(np.diag(y)) + sum(y[y < 0]), 0)
    max_val = sum(y[y > 0])

    return y, min_val, max_val


def select_boundaries(segmentation_criterion_vector,
                      downbeat_vector,
                      song_duration,
                      minimum_segment_size=0):
    """


    Parameters
    ----------
    segmentation_criterion_vector : TYPE
        DESCRIPTION.
    downbeat_vector : TYPE
        DESCRIPTION.
    song_duration : TYPE
        DESCRIPTION.
    minimum_segment_size : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    boundaries_vector : TYPE
        DESCRIPTION.

    """

    segmentation_criterion_peaks_vector = select_peaks(
        segmentation_criterion_vector)
    # segmentation_criterion_peaks_vector = librosa.util.peak_pick(
    #     segmentation_criterion_vector,
    #     pre_max=2, post_max=2,
    #     pre_avg=8, post_avg=8,
    #     delta=0, wait=4)
    peaks_ind_vector = np.where(segmentation_criterion_peaks_vector != 0)[0]
    transition_times_vector = downbeat_vector[1:-1]
    for i in range(len(peaks_ind_vector) - 1):
        if((peaks_ind_vector[i + 1] - peaks_ind_vector[i] <
            minimum_segment_size) or (transition_times_vector[
                peaks_ind_vector[i + 1]]
                - transition_times_vector[peaks_ind_vector[i]] <
                minimum_segment_size)):

            if (segmentation_criterion_peaks_vector[
                    peaks_ind_vector[i + 1]] >
                    segmentation_criterion_peaks_vector[peaks_ind_vector[i]]):
                segmentation_criterion_peaks_vector[i] = 0
            else:
                segmentation_criterion_peaks_vector[i + 1] = 0

    threshold = otsu_thresholding(
        segmentation_criterion_peaks_vector[
            np.nonzero(segmentation_criterion_peaks_vector)])

    boundaries_vector = transition_times_vector[
        np.where(segmentation_criterion_peaks_vector > threshold)[0]]

    start_time = downbeat_vector[0]
    if start_time > 0.5:
        boundaries_vector = np.sort(np.concatenate(
            (boundaries_vector, np.array([0, start_time]))))

    stop_time = downbeat_vector[-1]
    if stop_time < (song_duration - 0.5):
        boundaries_vector = np.sort(np.concatenate(
            (boundaries_vector, np.array([stop_time, song_duration]))))
    else:
        boundaries_vector = np.sort(np.append(boundaries_vector,
                                              song_duration))

    return boundaries_vector


def estimate_beats_madmom_1(audio_filename):
    """
    From Harmonix dataset source
    Produces beat time estimates according to the paper:

        Florian Krebs, Sebastian Böck and Gerhard Widmer, “An Efficient State
        Space Model for Joint Tempo and Meter Tracking”, Proceedings of the
        16th International Society for Music Information
        Retrieval Conference (ISMIR), 2015.

    Args:
        filname: str - The filename (with path) to the mp3 audio file to be
        analyzed by this algorithm.

    Return:
        list(float) - The estimates of the beat positions in the audio as a
        list of positions in seconds.
    """
    proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
    act = madmom.features.beats.RNNBeatProcessor()(audio_filename)
    return proc(act)


def estimate_downbeats_madmom_1(filename, reference_beats_filename):
    """
    From Harmonix dataset source
    Estimates beats using reference beats and the `DBNBarTrackingProcessor`
    provided with madmom:

        S. Bock, F. Korzeniowski, J. Schlüter, F. Krebs, and G. Widmer,
        “Madmom: A new Python Audio and Music Signal Processing Library,”
        in Proceedings of the 24th ACM International Conference on Multimedia
        (ACMMM), Amsterdam, Netherlands, Oct. 2016.

    This estimator uses reference beat positions to estimate downbeat positions

    Args:
        filname: str - The filename (with path) to the mp3 audio file to be
        analyzed by this algorithm.

        reference_beats_filename: str - The filename (with path) to a csv file
        containing the beat positions as the first column.

    Return:
        list(float) - The estimates of the downbeat positions in the audio as
        a list of positions in seconds.
    """

    proc = madmom.features.downbeats.DBNBarTrackingProcessor(
        beats_per_bar=[3, 4])
    if isinstance(reference_beats_filename, six.string_types):
        try:
            beats = np.loadtxt(reference_beats_filename)
        except OSError:
            print('Beats file failed to load:' + str(reference_beats_filename))
            raise TypeError('Unable to read beats file.')
    elif hasattr(reference_beats_filename, 'shape'):
        beats = np.array(reference_beats_filename)
    else:
        raise ValueError('Input downbeats must be either a string or a'
                         ' numpy array.')
    act = madmom.features.downbeats.RNNBarProcessor()((filename, beats))
    downbeat_data = proc(act)
    estimated_beats = downbeat_data[:, 0]
    estimated_downbeats = downbeat_data[:, 1]
    downbeat_inds = np.argwhere(
        (estimated_downbeats[1:]-estimated_downbeats[:-1]) < 0)
    return estimated_beats[downbeat_inds].flatten()


def save_beats(beats_vector, downbeats_vector, filename):

    indexed_beats = np.zeros((len(beats_vector), 2))
    indexed_beats[:, 0] = beats_vector

    beat_index_in_bar = 0
    downbeat_index = 0
    for i in range(len(beats_vector)):
        if ((not downbeat_index == len(downbeats_vector))
                and (beats_vector[i] == downbeats_vector[downbeat_index])):
            beat_index_in_bar = 1
            downbeat_index += 1
        else:
            if downbeat_index != 0:
                beat_index_in_bar += 1
        indexed_beats[i, 1] = beat_index_in_bar
    madmom.io.write_beats(indexed_beats, filename)


def save_segments(segments_boundaries, filename):

    file_data = ''
    for segment_number in range(len(segments_boundaries) - 1):
        file_data += '{}\t{}\tSegment{}\n'.format(
            segments_boundaries[segment_number],
            segments_boundaries[segment_number+1],
            segment_number+1)
    with open(filename, 'w') as f:
        f.write(file_data)
