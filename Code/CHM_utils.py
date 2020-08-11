# -*- coding: utf-8 -*-

import os
import numpy as np
import six
import librosa
import madmom
import wave
import timbral_models


def create_click(beats_vector, sample_rate):

    sound = librosa.clicks(times=beats, sr=sample_rate, click_duration=2e-2)

    with wave.open('sound.wav', 'w') as sound_object:
        sound_object.setnchannels(1)  # mono
        sound_object.setsampwidth(2)
        sound_object.setframerate(44100)
        sound_object.writeframesraw(sound)
