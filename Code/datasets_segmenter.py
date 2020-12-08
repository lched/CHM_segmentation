# -*- coding: utf-8 -*-

# Python standard library imports
import os
import numpy as np

# Third party imports
import librosa
import madmom

# Local imports
import estimate_beats
import estimate_downbeats
from utils import (save_segments, save_beats, compute_repetition_criterion,
                   compute_novelty_criterion, select_boundaries)
from features import compute_multi_features_ssm


def segmenter(audio_filename, sample_rate=22050,
              beats_file=None,
              features=['chroma_stft', 'mfcc'],
              criteria=['repetition', 'novelty'],
              coefficients=[1, 1, 1, 1]):
    '''
    Estimate segments boundaries of a song. Does not estimate labels. Based on:
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
    # Get beats and downbeats
    # =============================================================================
    if beats_file is None:
        beats_file = os.path.join(
            os.path.dirname(os.path.dirname(audio_filename)),
            'estimations',
            'beats_and_downbeats',
            os.path.splitext(os.path.basename(audio_filename))[0] + '.lab'
            )
        try:
            # Try to open beats from file if already computed
            beats_vector = madmom.io.load_beats(beats_file)
            downbeats_vector = madmom.io.load_beats(beats_file, downbeats=True)
            print('\tBeats and downbeats already computed, loaded from file')
            # print('\tBeats already computed, loaded from file %s'
            #       % beats_file)
        except OSError:
            print('\tComputing beats using madmom...')
            beats_vector = estimate_beats.madmom_1(audio_filename)[0]
            print('\tComputing downbeats using madmom')
            downbeats_vector = estimate_downbeats.madmom_1(audio_filename,
                                                           beats_vector)[0]
            # Save results to .lab file
            save_beats(beats_vector, downbeats_vector, beats_file)
    else:
        if isinstance(beats_file, str):
            try:
                beats_vector = madmom.io.load_beats(beats_file)
                downbeats_vector = madmom.io.load_beats(beats_file,
                                                        downbeats=True)
                print('\tBeats and downbeats successfully loaded from file: '
                      '{}'.format(beats_file))
            except OSError:
                print('Beats file failed to load: {}'.format(beats_file))
                raise OSError('Unable to read beats file.')
        elif isinstance(beats_file, tuple):
            beats_vector = beats_file[0]
            downbeats_vector = beats_file[1]
            print('\tBeats and downbeats successfully loaded from tuple')
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
                                                    features,
                                                    save_features=True)

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
                compute_repetition_criterion(multi_features_ssm[:,
                                                                :,
                                                                feature_index]))
            criterion_index += 1

    if 'novelty' in criteria:
        for feature_index in range(len(features)):
            segmentation_criterion_matrix[criterion_index, :] = (
                compute_novelty_criterion(multi_features_ssm[:,
                                                             :,
                                                             feature_index]))
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
        os.path.splitext(os.path.basename(audio_filename))[0] + '.lab'
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
