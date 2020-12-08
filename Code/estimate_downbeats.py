#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 10-13-19 by Matt C. McCallum
"""

# Third party imports
import madmom
import numpy as np


def madmom_1(filename, reference_beats_filename):
    """
    Estimates beats using reference beats and the `DBNBarTrackingProcessor`
    provided with madmom:

        S. Bock, F. Korzeniowski, J. Schlüter, F. Krebs, and G. Widmer,
        “Madmom: A new Python Audio and Music Signal Processing Library,” in
        Proceedings of the 24th ACM International Conference on Multimedia
        (ACMMM), Amsterdam, Netherlands, Oct. 2016.

    This estimator uses reference beat positions to estimate downbeat
    positions.

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
    if isinstance(reference_beats_filename, str):
        try:
            beats = madmom.io.load_beats(reference_beats_filename)
        except OSError:
            print('Beats file failed to load:' + str(reference_beats_filename))
            raise TypeError('Unable to read beats file.')
    elif hasattr(reference_beats_filename, 'shape'):
        beats = np.array(reference_beats_filename)
    else:
        raise ValueError('Input beats must be either a string or a'
                         ' numpy array.')
    act = madmom.features.downbeats.RNNBarProcessor()((filename, beats))
    downbeat_data = proc(act)
    estimated_beats = downbeat_data[:, 0]
    estimated_downbeats = downbeat_data[:, 1]
    downbeat_inds = np.argwhere(
        (estimated_downbeats[1:] - estimated_downbeats[:-1]) < 0)
    return estimated_beats[downbeat_inds].flatten(), filename


def madmom_2(filename, reference_beats_filename):
    """
    Produces downbeat time estimates according to the algorithm described in:

        Sebastian Böck, Florian Krebs and Gerhard Widmer, “Joint Beat and
        Downbeat Tracking with Recurrent Neural Networks” Proceedings of the
        17th International Society for Music Information Retrieval Conference
        (ISMIR), 2016.

    Args:
        filname: str - The filename (with path) to the mp3 audio file to be
        analyzed by this algorithm.

        reference_beats_filename: str - Not used, only provided here for
        consistence of interface with other downbeat estimator functions.

    Return:
        list(float) - The estimates of the downbeat positions in the audio as
        a list of positions in seconds.
    """
    proc = madmom.features.downbeats.DBNDownBeatTrackingProcessor(
        beats_per_bar=[3, 4], fps=100)
    act = madmom.features.downbeats.RNNDownBeatProcessor()(filename)
    downbeat_data = proc(act)
    estimated_beats = downbeat_data[:, 0]
    estimated_downbeats = downbeat_data[:, 1]
    downbeat_inds = np.argwhere(
        (estimated_downbeats[1:] - estimated_downbeats[:-1]) < 0)
    return estimated_beats[downbeat_inds].flatten(), filename
