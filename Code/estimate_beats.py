#!/usr/bin/env python3
'''
Created 04-06-19 by Matt C. McCallum

Code originally for estimating the beat positions from the mp3 audio of the
Harmonix dataset using the range of algorithms evaluated in the paper.
'''

# Third party imports
import librosa
import madmom


def madmom_1(audio_filename):
    """
    Produces beat time estimates according to the paper:

        Florian Krebs, Sebastian Böck and Gerhard Widmer, “An Efficient State
        Space Model for Joint Tempo and Meter Tracking”, Proceedings of the
        16th International Society for Music Information Retrieval Conference
        (ISMIR), 2015.

    Args:
        filname: str - The filename (with path) to the mp3 audio file to be
        analyzed by this algorithm.

    Return:
        list(float) - The estimates of the beat positions in the audio as a
        list of positions in seconds.
    """
    proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
    act = madmom.features.beats.RNNBeatProcessor()(audio_filename)
    return proc(act), audio_filename


def madmom_2(audio_filename):
    """
    Produces beat time estimates according to the paper:

        Filip Korzeniowski, Sebastian Böck and Gerhard Widmer, “Probabilistic
        Extraction of Beat Positions from a Beat Activation Function”,
        Proceedings of the 15th International Society for Music Information
        Retrieval Conference (ISMIR), 2014.

    Args:
        filname: str - The filename (with path) to the mp3 audio file to be
        analyzed by this algorithm.

    Return:
        list(float) - The estimates of the beat positions in the audio as a
        list of positions in seconds.
    """
    proc = madmom.features.beats.CRFBeatDetectionProcessor(fps=100)
    act = madmom.features.beats.RNNBeatProcessor()(audio_filename)
    return proc(act), audio_filename


def madmom_3(audio_filename):
    """
    Produces beat time estimates according to the paper:

        Sebastian Böck and Markus Schedl, “Enhanced Beat Tracking with
        Context-Aware Neural Networks”, Proceedings of the 14th International
        Conference on Digital Audio Effects (DAFx), 2011.

    Args:
        filname: str - The filename (with path) to the mp3 audio file to be
        analyzed by this algorithm.

    Return:
        list(float) - The estimates of the beat positions in the audio as a
        list of positions in seconds.
    """
    proc = madmom.features.beats.BeatDetectionProcessor(fps=100)
    act = madmom.features.beats.RNNBeatProcessor()(audio_filename)
    return proc(act), audio_filename


def madmom_4(audio_filename):
    """
    Produces beat time estimates according to the paper:

        Sebastian Böck and Markus Schedl, “Enhanced Beat Tracking with
        Context-Aware Neural Networks”, Proceedings of the 14th International
        Conference on Digital Audio Effects (DAFx), 2011.

    Args:
        filname: str - The filename (with path) to the mp3 audio file to be
        analyzed by this algorithm.

    Return:
        list(float) - The estimates of the beat positions in the audio as a
        list of positions in seconds.
    """
    proc = madmom.features.beats.BeatTrackingProcessor(fps=100)
    act = madmom.features.beats.RNNBeatProcessor()(audio_filename)
    return proc(act), audio_filename


def ellis(audio_filename):
    """
    Produces beat time estimates according to the paper:

        Ellis, Daniel PW. “Beat tracking by dynamic programming.” Journal of
        New Music Research, 2007.

    Using the implementation contained in the librosa python module.

    Args:
        filname: str - The filename (with path) to the mp3 audio file to be
        analyzed by this algorithm.

    Return:
        list(float) - The estimates of the beat positions in the audio as a
        list of positions in seconds.
    """
    signal, _ = librosa.load(audio_filename)
    _, result = librosa.beat.beat_track(signal, units='time')
    return result, audio_filename
