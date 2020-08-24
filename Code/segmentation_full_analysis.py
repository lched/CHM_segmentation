# -*- coding: utf-8 -*-

# Python standard library imports
import os
from shutil import copy2
import numpy as np

# Third party imports
import librosa
import madmom

# Local imports
import estimate_beats
import estimate_downbeats
from CHM_utils import (save_segments, save_beats, compute_repetition_criterion,
                       compute_novelty_criterion, select_boundaries,
                       divide_beats_in_two, create_click)
from CHM_features import compute_multi_features_ssm


def segmenter(output_directory,
              audio_filename, sample_rate=22050,
              beats_file=None,
              features=['chroma_stft', 'mfcc'],
              criteria=['repetition', 'novelty'],
              coefficients=[1, 1, 1, 1]):
    '''
    The aim is to have everything you need in an output directory. (segments,
                                                                    beats,
                                                                    click)
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
    output_directory : path, string
        Output directory to save results.
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
    # Create ouput directory if it does not exist
    # =============================================================================
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # =============================================================================
    # Get audio
    # =============================================================================
    print('Computing: {}'.format(audio_filename))
    song_signal, sample_rate = librosa.load(audio_filename, sr=sample_rate)

    # Copy audio to output_directory
    copy2(audio_filename, output_directory)

    # =============================================================================
    #  Get beats and downbeats
    # =============================================================================
    if beats_file is None:
        print('\tComputing beats using madmom...')
        beats_vector = estimate_beats.ellis(audio_filename)[0]
        print('\tComputing downbeats using madmom')
        downbeats_vector = estimate_downbeats.madmom_1(audio_filename,
                                                       beats_vector)[0]
    else:
        if isinstance(beats_file, str):
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

    # In any case, save the beats to .lab file in output_directory
    beats_file = os.path.join(
        output_directory,
        os.path.splitext(os.path.basename(audio_filename))[0]
        + '_estimated_beats.lab')
    save_beats(beats_vector, downbeats_vector, beats_file)

    # Create click audio file to check beats. This audio file has the same
    # sample rate as the original file
    click_fname = os.path.join(
        output_directory,
        os.path.splitext(os.path.basename(audio_filename))[0]
        + '_click.wav')
    original_sample_rate = librosa.get_samplerate(audio_filename)
    create_click(beats_vector, original_sample_rate, click_fname)

    # Divide beats in two
    beats_vector = divide_beats_in_two(beats_vector)

    # =============================================================================
    # Features computation
    # =============================================================================
    multi_features_ssm = compute_multi_features_ssm(audio_filename,
                                                    song_signal, sample_rate,
                                                    beats_vector,
                                                    downbeats_vector,
                                                    features)

    # =============================================================================
    # Features display
    # =============================================================================

    # =============================================================================
    # Segmentation
    # =============================================================================
    # Check if there are enough coefficients
    if len(coefficients) != len(criteria)*len(features):
        raise ValueError('The number of provided coefficients must be equal to'
                         'the number of features * the number of criteria')

    segmentation_criterion_matrix = np.zeros((len(criteria)*len(features),
                                              len(downbeats_vector) - 2))
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

    segmentation_criterion_vector = np.sum((segmentation_criterion_matrix
                                            * np.array(coefficients)
                                            [:, np.newaxis]), axis=0)

    song_duration = len(song_signal)/sample_rate
    estimated_boundaries_vector = select_boundaries(
        segmentation_criterion_vector, downbeats_vector, song_duration)

    # =============================================================================
    # Save results
    # =============================================================================
    estimated_boundaries_fname = os.path.join(
        output_directory,
        os.path.splitext(os.path.basename(audio_filename))[0]
        + '_estimated_segments.lab')
    save_segments(estimated_boundaries_vector,
                  estimated_boundaries_fname)

    estimated_segments = (np.zeros((len(estimated_boundaries_vector)-1, 2)),
                          [])
    for i in range(len(estimated_boundaries_vector) - 1):
        estimated_segments[0][i, 0] = estimated_boundaries_vector[i]
        estimated_segments[0][i, 1] = estimated_boundaries_vector[i+1]
        estimated_segments[1].append('Segment{}'.format(i+1))

    return estimated_segments


# =============================================================================
# Test Code
# =============================================================================
output_directory = '/home/leo/Desktop/test'
test_fname = ('/media/leo/42A45DCCA45DC359/MIR_DATASETS/Beatles_Helene/audio/'
              '01_-_A_Hard_Day_s_Night.wav')
beats_file = ('/media/leo/42A45DCCA45DC359/MIR_DATASETS/Beatles_Helene/'
              'estimations/beats_and_downbeats/01_-_A_Hard_Day_s_Night.lab')
est_segments = segmenter(output_directory,
                         test_fname,
                         # beats_file=beats_file,
                         features=['chroma_stft', 'mfcc'],
                         criteria=['repetition', 'novelty'],
                         coefficients=[1, 1, 1, 1])
