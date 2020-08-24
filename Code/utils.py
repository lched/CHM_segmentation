# -*- coding: utf-8 -*-

# Python standard library imports
import numpy as np

# Third party imports
import librosa
import madmom
import scipy.io.wavfile


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


def check_beats_downbeats_alignment(beats_vector, downbeats_vector):
    '''
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

    '''
    max_error = 0
    db_closest_beat_vector = np.zeros(np.size(downbeats_vector))

    for i in range(len(downbeats_vector)):
        if not downbeats_vector[i] in beats_vector:
            min_ind = np.argmin(abs(beats_vector - downbeats_vector[i]))
            min_val = min(abs(beats_vector - downbeats_vector[i]))
            max_error = max(max_error, min_val)
            db_closest_beat_vector[i] = beats_vector[min_ind]
        else:
            db_closest_beat_vector[i] = downbeats_vector[i]

    return max_error, db_closest_beat_vector


def divide_beats_in_two(beats_vector):
    beats_length_vector = (np.array(beats_vector[1:])
                           - np.array(beats_vector[:-1]))
    beats_vector = np.concatenate((beats_vector, beats_vector[:-1]
                                   + beats_length_vector/2))
    beats_vector = np.sort(beats_vector, axis=None)

    return beats_vector


def compute_repetition_criterion(feature_SSM):

    SSM_length = len(feature_SSM)
    # TODO: replace librosa.segment.recurrence_to_lag
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
    '''
    CANCEL_PEAKS smoothens array vector by clearing its local extrema.
    '''

    smooth_vector = vector

    for i in range(1, len(vector)-1):
        diff_last = smooth_vector[i] - smooth_vector[i-1]
        diff_next = smooth_vector[i+1] - smooth_vector[i]
        if (diff_last > 0) & (diff_next < 0):
            smooth_vector[i] = smooth_vector[i] - min(diff_last, -diff_next)
        elif (diff_last < 0) & (diff_next > 0):
            smooth_vector[i] = smooth_vector[i] + min(-diff_last, diff_next)

    return smooth_vector


def select_peaks(vector):
    '''
    SELECT_PEAKS picks the peaks in vector (puts every non-local maximum value
    to zero).
    '''

    peaks_vector = np.zeros(len(vector))
    for i in range(1, len(vector)-1):
        if (vector[i] > vector[i-1]) & (vector[i] >= vector[i+1]):
            peaks_vector[i] = vector[i]

    return peaks_vector


def otsu_thresholding(data, number_of_bins=10):

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
    '''

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

    '''

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

    if len(feature_SSM) != len(feature_SSM[0]):
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
    '''
    checker_gauss returns a checkerboard-like matrix of type [1 -1; -1 1]
    weighted by a gaussian.
    '''

    hN = int(np.ceil(N/2))
    y = np.zeros((N, N))
    # Upper left (the biggest corner in case N is even)
    for i in range(1, hN + 1):
        for j in range(1, hN + 1):
            y[i-1, j-1] = np.exp(-(((i-hN)/hN)**2 + (((j-hN)/hN)**2))*4)

    # The other corners are defined by symetry
    if N % 2 == 0:
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
    '''
    block_diag_gauss returns a homogeneous / non homogeneous transition
    kernel such as proposed by Kaiser & Peeters (2013).

    INPUT
    - N           : size of the matrix
    - balancing   : 'before' the gaussian weighting (as K. & P. do), default
                    'after' the gaussian weighting (so that the final mean
                                                      is actually zero)

    OUTPUT
    - y

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

    '''
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
    with open(filename, 'w') as segment_file:
        segment_file.write(file_data)


def create_click(beats_vector, sample_rate, output_fname):

    data = librosa.clicks(times=beats_vector,
                          sr=sample_rate,
                          click_duration=2e-2)
    scipy.io.wavfile.write(output_fname, sample_rate, data)
