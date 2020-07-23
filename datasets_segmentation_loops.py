# -*- coding: utf-8 -*-

import os
import numpy as np
import mir_eval
import CHM_segmenter as CHM


def beatles_segmentation_loop(beatles_dataset_path,
                              features=['chroma_stft', 'mfcc'],
                              criteria=['repetition', 'novelty'],
                              coefficients=[1, 1, 1, 1]):
    '''
    Segmentation evaluation loop for the Beatles dataset.
    The dataset should be organized as follows (consistent
    with the Music Structure Analysis Framework):
        ./dataset_folder
            /audio
            /estimations (empty initially)
            /features (empty initially)
            /references

    Parameters
    ----------
    beatles_dataset_path : str
        Path of the dataset
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

    Returns
    -------
    Vectors of results. F-measure, Precision rate and Recall rate for 0.5s and
    3s windows.
    F05_vector : array
    P05_vector : array
    R05_vector : array
    F3_vector : array
    P3_vector : array
    R3_vector : array
    '''
    START_SONG = 1
    STOP_SONG = 181

    F05_vector = np.zeros(STOP_SONG - START_SONG)
    P05_vector = np.zeros(STOP_SONG - START_SONG)
    R05_vector = np.zeros(STOP_SONG - START_SONG)

    F3_vector = np.zeros(STOP_SONG - START_SONG)
    P3_vector = np.zeros(STOP_SONG - START_SONG)
    R3_vector = np.zeros(STOP_SONG - START_SONG)

    for song_number in range(START_SONG, STOP_SONG):
        # 150 skipped
        if song_number != 150:
            # Get information on the Beatles song
            album_number, track_number = beatles_num_to_album_track(
                song_number)
            album_name, track_name = beatles_album_track_to_names(album_number,
                                                                  track_number)
            print('({}) Computing: {}\t{}'.format(song_number,
                                                  album_name,
                                                  track_name))

            # Define path of audio file
            audio_filename = os.path.join(beatles_dataset_path,
                                          'audio',
                                          track_name)
            # Compute estimated segments boundaries
            estimated_segments = CHM.segmenter(audio_filename,
                                               sample_rate=None,
                                               features=features,
                                               criteria=criteria,
                                               coefficients=coefficients)[0]

            # Get ground truth boundaries for evaluation
            reference_segments_fname = os.path.join(
                beatles_dataset_path,
                'references',
                'segments',
                track_name[:-4] + '.lab')
            reference_segments = mir_eval.io.load_labeled_intervals(
                reference_segments_fname)[0]

            # Evaluate segmentation results
            P05, R05, F05 = mir_eval.segment.detection(
                reference_segments, estimated_segments, window=0.5)
            P3, R3, F3 = mir_eval.segment.detection(
                reference_segments, estimated_segments, window=3)

            P05_vector[song_number - START_SONG] = P05
            R05_vector[song_number - START_SONG] = R05
            F05_vector[song_number - START_SONG] = F05

            F3_vector[song_number - START_SONG] = F3
            P3_vector[song_number - START_SONG] = P3
            R3_vector[song_number - START_SONG] = R3

    print('Segmentation results')
    print('F0_5\t\t', round(100*np.mean(F05_vector), ndigits=1), '%')
    print('P0_5\t\t', round(100*np.mean(P05_vector), ndigits=1), '%')
    print('R0_5\t\t', round(100*np.mean(R05_vector), ndigits=1), '%')
    print('F3\t\t', round(100*np.mean(F3_vector), ndigits=1), '%')
    print('P3\t\t', round(100*np.mean(P3_vector), ndigits=1), '%')
    print('R3\t\t', round(100*np.mean(R3_vector), ndigits=1), '%')

    return (F05_vector, P05_vector, R05_vector,
            F3_vector, P3_vector, R3_vector)


def beatles_num_to_album_track(num):
    '''
    Convert a number between 1 and 180 (both included) to the corresponding
    album and track numbers

    Parameters
    ----------
    num : int
        Number between 1 and 180 (included)

    Returns
    -------
    num_album : int
        Number of the corresponding album.
    num_track : int
        Number of the corresponding track.

    '''
    N_start = np.array([0, 14, 28, 41, 55, 69, 83, 97,
                        110, 121, 138, 151, 168])
    N_stop = np.array([14, 28, 41, 55, 69, 83, 97, 110,
                       121, 138, 151, 168, 180])

    num_album = np.where((N_start < num) & (num <= N_stop))[0][0]
    num_track = num - N_start[num_album]

    return num_album+1, num_track


def beatles_album_track_to_names(num_album, num_track):
    '''
    Convert album and track numbers to name of the audio file.
    '''
    if (num_album == 1):
        album = '01-PleasePleaseMe'
        nb_songs = 14
        if (num_track == 1):
            track = '01_-_I_Saw_Her_Standing_There.wav'
        elif (num_track == 2):
            track = '02_-_Misery.wav'
        elif (num_track == 3):
            track = '03_-_Anna_(Go_To_Him).wav'
        elif (num_track == 4):
            track = '04_-_Chains.wav'
        elif (num_track == 5):
            track = '05_-_Boys.wav'
        elif (num_track == 6):
            track = '06_-_Ask_Me_Why.wav'
        elif (num_track == 7):
            track = '07_-_Please_Please_Me.wav'
        elif (num_track == 8):
            track = '08_-_Love_Me_Do.wav'
        elif (num_track == 9):
            track = '09_-_P._S._I_Love_You.wav'
        elif (num_track == 10):
            track = '10_-_Baby_It_s_You.wav'
        elif (num_track == 11):
            track = '11_-_Do_You_Want_To_Know_A_Secret.wav'
        elif (num_track == 12):
            track = '12_-_A_Taste_Of_Honey.wav'
        elif (num_track == 13):
            track = '13_-_There_s_A_Place.wav'
        elif (num_track == 14):
            track = '14_-_Twist_And_Shout.wav'
        else:
            return('The song number for album %s must lie between 1 and %d.\n',
                   album, nb_songs)
    elif (num_album == 2):
        album = '02_With_The_Beatles'
        nb_songs = 14
        if (num_track == 1):
            track = '01_-_It_Won_t_Be_Long.wav'
        elif (num_track == 2):
            track = '02_-_All_I_ve_Got_To_Do.wav'
        elif (num_track == 3):
            track = '03_-_All_My_Loving.wav'
        elif (num_track == 4):
            track = '04_-_Don_t_Bother_Me.wav'
        elif (num_track == 5):
            track = '05_-_Little_Child.wav'
        elif (num_track == 6):
            track = '06_-_Till_There_Was_You.wav'
        elif (num_track == 7):
            track = '07_-_Please_Mister_Postman.wav'
        elif (num_track == 8):
            track = '08_-_Roll_Over_Beethoven.wav'
        elif (num_track == 9):
            track = '09_-_Hold_Me_Tight.wav'
        elif (num_track == 10):
            track = '10_-_You_Really_Got_A_Hold_On_Me.wav'
        elif (num_track == 11):
            track = '11_-_I_Wanna_Be_Your_Man.wav'
        elif (num_track == 12):
            track = '12_-_Devil_In_Her_Heart.wav'
        elif (num_track == 13):
            track = '13_-_Not_A_Second_Time.wav'
        elif (num_track == 14):
            track = '14_-_Money.wav'
        else:
            return('The song number for album %s must lie between 1 and %d.\n',
                   album, nb_songs)

    elif (num_album == 3):
        album = '03_A_Hard_Days_Night'
        nb_songs = 13
        if (num_track == 1):
            track = '01_-_A_Hard_Day_s_Night.wav'
        elif (num_track == 2):
            track = '02_-_I_Should_Have_Known_Better.wav'
        elif (num_track == 3):
            track = '03_-_If_I_Fell.wav'
        elif (num_track == 4):
            track = '04_-_I_m_Happy_Just_To_Dance_With_You.wav'
        elif (num_track == 5):
            track = '05_-_And_I_Love_Her.wav'
        elif (num_track == 6):
            track = '06_-_Tell_Me_Why.wav'
        elif (num_track == 7):
            track = '07_-_Can_t_Buy_Me_Love.wav'
        elif (num_track == 8):
            track = '08_-_Any_Time_At_All.wav'
        elif (num_track == 9):
            track = '09_-_I_ll_Cry_Instead.wav'
        elif (num_track == 10):
            track = '10_-_Things_We_Said_Today.wav'
        elif (num_track == 11):
            track = '11_-_When_I_Get_Home.wav'
        elif (num_track == 12):
            track = '12_-_You_Can_t_Do_That.wav'
        elif (num_track == 13):
            track = '13_-_I_ll_Be_Back.wav'
        else:
            return('The song number for album %s must lie between 1 and %d.\n',
                   album, nb_songs)

    elif (num_album == 4):
        album = '04_Beatles_For_Sale'
        nb_songs = 14
        if (num_track == 1):
            track = '01_-_No_Reply.wav'
        elif (num_track == 2):
            track = '02_-_I_m_a_Loser.wav'
        elif (num_track == 3):
            track = '03_-_Baby_s_In_Black.wav'
        elif (num_track == 4):
            track = '04_-_Rock_and_Roll_Music.wav'
        elif (num_track == 5):
            track = '05_-_I_ll_Follow_the_Sun.wav'
        elif (num_track == 6):
            track = '06_-_Mr._Moonlight.wav'
        elif (num_track == 7):
            track = '07_-_Kansas_City-_Hey,_Hey,_Hey,_Hey.wav'
        elif (num_track == 8):
            track = '08_-_Eight_Days_a_Week.wav'
        elif (num_track == 9):
            track = '09_-_Words_of_Love.wav'
        elif (num_track == 10):
            track = '10_-_Honey_Don_t.wav'
        elif (num_track == 11):
            track = '11_-_Every_Little_Thing.wav'
        elif (num_track == 12):
            track = '12_-_I_Don_t_Want_to_Spoil_the_Party.wav'
        elif (num_track == 13):
            track = '13_-_What_You_re_Doing.wav'
        elif (num_track == 14):
            track = '14_-_Everybody_s_Trying_to_Be_My_Baby.wav'
        else:
            return('The song number for album %s must lie between 1 and %d.\n',
                   album, nb_songs)

    elif (num_album == 5):
        album = '05_Help'
        nb_songs = 14
        if (num_track == 1):
            track = '01_-_Help!.wav'
        elif (num_track == 2):
            track = '02_-_The_Night_Before.wav'
        elif (num_track == 3):
            track = '03_-_You_ve_Got_To_Hide_Your_Love_Away.wav'
        elif (num_track == 4):
            track = '04_-_I_Need_You.wav'
        elif (num_track == 5):
            track = '05_-_Another_Girl.wav'
        elif (num_track == 6):
            track = '06_-_You_re_Going_to_Lose_That_Girl.wav'
        elif (num_track == 7):
            track = '07_-_Ticket_To_Ride.wav'
        elif (num_track == 8):
            track = '08_-_Act_Naturally.wav'
        elif (num_track == 9):
            track = '09_-_It_s_Only_Love.wav'
        elif (num_track == 10):
            track = '10_-_You_Like_Me_Too_Much.wav'
        elif (num_track == 11):
            track = '11_-_Tell_Me_What_You_See.wav'
        elif (num_track == 12):
            track = '12_-_I_ve_Just_Seen_a_Face.wav'
        elif (num_track == 13):
            track = '13_-_Yesterday.wav'
        elif (num_track == 14):
            track = '14_-_Dizzy_Miss_Lizzie.wav'
        else:
            return('The song number for album %s must lie between 1 and %d.\n',
                   album, nb_songs)

    elif (num_album == 6):
        album = '06_Rubber_Soul'
        nb_songs = 14
        if (num_track == 1):
            track = '01_-_Drive_My_Car.wav'
        elif (num_track == 2):
            track = '02_-_Norwegian_Wood_(This_Bird_Has_Flown).wav'
        elif (num_track == 3):
            track = '03_-_You_Won_t_See_Me.wav'
        elif (num_track == 4):
            track = '04_-_Nowhere_Man.wav'
        elif (num_track == 5):
            track = '05_-_Think_For_Yourself.wav'
        elif (num_track == 6):
            track = '06_-_The_Word.wav'
        elif (num_track == 7):
            track = '07_-_Michelle.wav'
        elif (num_track == 8):
            track = '08_-_What_Goes_On.wav'
        elif (num_track == 9):
            track = '09_-_Girl.wav'
        elif (num_track == 10):
            track = '10_-_I_m_Looking_Through_You.wav'
        elif (num_track == 11):
            track = '11_-_In_My_Life.wav'
        elif (num_track == 12):
            track = '12_-_Wait.wav'
        elif (num_track == 13):
            track = '13_-_If_I_Needed_Someone.wav'
        elif (num_track == 14):
            track = '14_-_Run_For_Your_Life.wav'
        else:
            return('The song number for album %s must lie between 1 and %d.\n',
                   album, nb_songs)

    elif (num_album == 7):
        album = '07_Revolver'
        nb_songs = 14
        if (num_track == 1):
            track = '01_-_Taxman.wav'
        elif (num_track == 2):
            track = '02_-_Eleanor_Rigby.wav'
        elif (num_track == 3):
            track = '03_-_I_m_Only_Sleeping.wav'
        elif (num_track == 4):
            track = '04_-_Love_You_To.wav'
        elif (num_track == 5):
            track = '05_-_Here,_There_And_Everywhere.wav'
        elif (num_track == 6):
            track = '06_-_Yellow_Submarine.wav'
        elif (num_track == 7):
            track = '07_-_She_Said_She_Said.wav'
        elif (num_track == 8):
            track = '08_-_Good_Day_Sunshine.wav'
        elif (num_track == 9):
            track = '09_-_And_Your_Bird_Can_Sing.wav'
        elif (num_track == 10):
            track = '10_-_For_No_One.wav'
        elif (num_track == 11):
            track = '11_-_Doctor_Robert.wav'
        elif (num_track == 12):
            track = '12_-_I_Want_To_Tell_You.wav'
        elif (num_track == 13):
            track = '13_-_Got_To_Get_You_Into_My_Life.wav'
        elif (num_track == 14):
            track = '14_-_Tomorrow_Never_Knows.wav'
        else:
            return('The song number for album %s must lie between 1 and %d.\n',
                   album, nb_songs)

    elif (num_album == 8):
        album = '08_Sgt_Peppers_Lonely_Hearts_Club_Band'
        nb_songs = 13
        if (num_track == 1):
            track = '01_-_Sgt._Pepper_s_Lonely_Hearts_Club_Band.wav'
        elif (num_track == 2):
            track = '02_-_With_A_Little_Help_From_My_Friends.wav'
        elif (num_track == 3):
            track = '03_-_Lucy_In_The_Sky_With_Diamonds.wav'
        elif (num_track == 4):
            track = '04_-_Getting_Better.wav'
        elif (num_track == 5):
            track = '05_-_Fixing_A_Hole.wav'
        elif (num_track == 6):
            track = '06_-_She_s_Leaving_Home.wav'
        elif (num_track == 7):
            track = '07_-_Being_For_The_Benefit_Of_Mr._Kite!.wav'
        elif (num_track == 8):
            track = '08_-_Within_You_Without_You.wav'
        elif (num_track == 9):
            track = '09_-_When_I_m_Sixty-Four.wav'
        elif (num_track == 10):
            track = '10_-_Lovely_Rita.wav'
        elif (num_track == 11):
            track = '11_-_Good_Morning_Good_Morning.wav'
        elif (num_track == 12):
            track = '12_-_Sgt._Pepper_s_Lonely_Hearts_Club_Band_(Reprise).wav'
        elif (num_track == 13):
            track = '13_-_A_Day_In_The_Life.wav'
        else:
            return('The song number for album %s must lie between 1 and %d.\n',
                   album, nb_songs)

    elif (num_album == 9):
        album = '09_-_Magical_Mystery_Tour'
        nb_songs = 11
        if (num_track == 1):
            track = '01_-_Magical_Mystery_Tour.wav'
        elif (num_track == 2):
            track = '02_-_The_Fool_On_The_Hill.wav'
        elif (num_track == 3):
            track = '03_-_Flying.wav'
        elif (num_track == 4):
            track = '04_-_Blue_Jay_Way.wav'
        elif (num_track == 5):
            track = '05_-_Your_Mother_Should_Know.wav'
        elif (num_track == 6):
            track = '06_-_I_Am_The_Walrus.wav'
        elif (num_track == 7):
            track = '07_-_Hello_Goodbye.wav'
        elif (num_track == 8):
            track = '08_-_Strawberry_Fields_Forever.wav'
        elif (num_track == 9):
            track = '09_-_Penny_Lane.wav'
        elif (num_track == 10):
            track = '10_-_Baby_You_re_A_Rich_Man.wav'
        elif (num_track == 11):
            track = '11_-_All_You_Need_Is_Love.wav'
        else:
            return('The song number for album %s must lie between 1 and %d.\n',
                   album, nb_songs)

    elif (num_album == 10):
        album = '10_-_CD1_-_The_Beatles'
        nb_songs = 17
        if (num_track == 1):
            track = 'CD1_-_01_-_Back_in_the_USSR.wav'
        elif (num_track == 2):
            track = 'CD1_-_02_-_Dear_Prudence.wav'
        elif (num_track == 3):
            track = 'CD1_-_03_-_Glass_Onion.wav'  # the file had been replaced
            # because it is a different version ! (in order to use the
            # isophonics annotations easily)
        elif (num_track == 4):
            track = 'CD1_-_04_-_Ob-La-Di,_Ob-La-Da.wav'
        elif (num_track == 5):
            track = 'CD1_-_05_-_Wild_Honey_Pie.wav'
        elif (num_track == 6):
            track = 'CD1_-_06_-The_Continuing_Story_of_Bungalow_Bill.wav'
        elif (num_track == 7):
            track = 'CD1_-_07_-_While_My_Guitar_Gently_Weeps.wav'
        elif (num_track == 8):
            track = 'CD1_-_08_-_Happiness_is_a_Warm_Gun.wav'
        elif (num_track == 9):
            track = 'CD1_-_09_-_Martha_My_Dear.wav'
        elif (num_track == 10):
            track = 'CD1_-_10_-_I_m_So_Tired.wav'
        elif (num_track == 11):
            track = 'CD1_-_11_-_Black_Bird.wav'
        elif (num_track == 12):
            track = 'CD1_-_12_-_Piggies.wav'
        elif (num_track == 13):
            track = 'CD1_-_13_-_Rocky_Raccoon.wav'
        elif (num_track == 14):
            track = 'CD1_-_14_-_Don_t_Pass_Me_By.wav'
        elif (num_track == 15):
            track = 'CD1_-_15_-_Why_Don_t_We_Do_It_In_The_Road.wav'
        elif (num_track == 16):
            track = 'CD1_-_16_-_I_Will.wav'
        elif (num_track == 17):
            track = 'CD1_-_17_-_Julia.wav'
        else:
            return('The song number for album %s must lie between 1 and %d.\n',
                   album, nb_songs)

    elif (num_album == 11):
        album = '11_-_CD2_-_The_Beatles'
        nb_songs = 13
        if (num_track == 1):
            track = 'CD2_-_01_-_Birthday.wav'
        elif (num_track == 2):
            track = 'CD2_-_02_-_Yer_Blues.wav'
        elif (num_track == 3):
            track = 'CD2_-_03_-_Mother_Nature_s_Son.wav'
        elif (num_track == 4):
            track = ('CD2_-_04_-_Everybody_s_Got_Something_To_Hide_Except_Me_'
                     'and_M.wav')
        elif (num_track == 5):
            track = 'CD2_-_05_-_Sexy_Sadie.wav'
        elif (num_track == 6):
            track = 'CD2_-_06_-_Helter_Skelter.wav'
        elif (num_track == 7):
            track = 'CD2_-_07_-_Long_Long_Long.wav'
        elif (num_track == 8):
            track = 'CD2_-_08_-_Revolution_1.wav'
        elif (num_track == 9):
            track = 'CD2_-_09_-_Honey_Pie.wav'
        elif (num_track == 10):
            track = 'CD2_-_10_-_Savoy_Truffle.wav'
        elif (num_track == 11):
            track = 'CD2_-_11_-_Cry_Baby_Cry.wav'
        elif (num_track == 12):
            track = 'CD2_-_12_-_Revolution_9.wav'
        elif (num_track == 13):
            track = 'CD2_-_13_-_Good_Night.wav'
        else:
            return('The song number for album %s must lie between 1 and %d.\n',
                   album, nb_songs)

    elif (num_album == 12):
        album = '12_-_Abbey_Road'
        nb_songs = 17

        if (num_track == 1):
            track = '01_-_Come_Together.wav'
        elif (num_track == 2):
            track = '02_-_Something.wav'
        elif (num_track == 3):
            track = '03_-_Maxwell_s_Silver_Hammer.wav'
        elif (num_track == 4):
            track = '04_-_Oh!_Darling.wav'
        elif (num_track == 5):
            track = '05_-_Octopus_s_Garden.wav'
        elif (num_track == 6):
            track = '06_-_I_Want_You.wav'
        elif (num_track == 7):
            track = '07_-_Here_Comes_The_Sun.wav'  # the file had been replaced
            # because it is a different version ! (in order to use the
            # isophonics annotations easily)
        elif (num_track == 8):
            track = '08_-_Because.wav'
        elif (num_track == 9):
            track = '09_-_You_Never_Give_Me_Your_Money.wav'
        elif (num_track == 10):
            track = '10_-_Sun_King.wav'
        elif (num_track == 11):
            track = '11_-_Mean_Mr_Mustard.wav'
        elif (num_track == 12):
            track = '12_-_Polythene_Pam.wav'
        elif (num_track == 13):
            track = '13_-_She_Came_In_Through_The_Bathroom_Window.wav'
        elif (num_track == 14):
            track = '14_-_Golden_Slumbers.wav'
        elif (num_track == 15):
            track = '15_-_Carry_That_Weight.wav'
        elif (num_track == 16):
            track = '16_-_The_End.wav'
        elif (num_track == 17):
            track = '17_-_Her_Majesty.wav'
        else:
            return('The song number for album %s must lie between 1 and %d.\n',
                   album, nb_songs)

    elif (num_album == 13):
        album = '13_-_Let_It_Be'
        nb_songs = 12
        if (num_track == 1):
            track = '01_-_Two_of_Us.wav'
        elif (num_track == 2):
            track = '02_-_Dig_a_Pony.wav'
        elif (num_track == 3):
            track = '03_-_Across_the_Universe.wav'  # the file had been
            # replaced because it is a different version ! (in order to use the
            # isophonics annotations easily)
        elif (num_track == 4):
            track = '04_-_I_Me_Mine.wav'
        elif (num_track == 5):
            track = '05_-_Dig_It.wav'
        elif (num_track == 6):
            track = '06_-_Let_It_Be.wav'
        elif (num_track == 7):
            track = '07_-_Maggie_Mae.wav'
        elif (num_track == 8):
            track = '08_-_I_ve_Got_A_Feeling.wav'
        elif (num_track == 9):
            track = '09_-_One_After_909.wav'
        elif (num_track == 10):
            track = '10_-_The_Long_and_Winding_Road.wav'
        elif (num_track == 11):
            track = '11_-_For_You_Blue.wav'
        elif (num_track == 12):
            track = '12_-_Get_Back.wav'  # the file had been replaced
            # because it is a different version ! (in order to use the
            # isophonics annotations easily)
        else:
            return('The song number for album %s must lie between 1 and %d.\n',
                   album, nb_songs)

    else:
        return('The album number must lie between 1 and 13.\n')

    return album, track


def harmonix_segmentation_loop(harmonix_dataset_path,
                               features=['chroma_stft', 'mfcc'],
                               criteria=['repetition', 'novelty'],
                               coefficients=[1, 1, 1, 1]):
    '''
    Segmentation evaluation loop for the Harmonix dataset.
    The dataset should be organized as follows (consistent
    with the Music Structure Analysis Framework):
        ./dataset_folder
            /audio
            /estimations (empty initially)
            /features (empty initially)
            /references

    Parameters
    ----------
    beatles_dataset_path : str
        Path of the dataset
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

    Returns
    -------
    Vectors of results. F-measure, Precision rate and Recall rate for 0.5s and
    3s windows.
    F05_vector : array
    P05_vector : array
    R05_vector : array
    F3_vector : array
    P3_vector : array
    R3_vector : array
    '''
    metadata_filename = os.path.join(harmonix_dataset_path, 'metadata.csv')
    with open(metadata_filename, encoding='latin_1') as f:
        tracks_metadata = f.readlines()

    F05_vector = np.array([])
    P05_vector = np.array([])
    R05_vector = np.array([])
    F3_vector = np.array([])
    P3_vector = np.array([])
    R3_vector = np.array([])

    for track_number in range(1, len(tracks_metadata)):

        track_filename = tracks_metadata[track_number].split(',')[0] + '.wav'
        track_name = tracks_metadata[track_number].split(',')[1]
        track_artist = tracks_metadata[track_number].split(',')[2]
        track_album = tracks_metadata[track_number].split(',')[3]

        print('({}) Computing: {}\t{}\t{}'.format(track_number,
                                                  track_name,
                                                  track_artist,
                                                  track_album))

        audio_filename = os.path.join(harmonix_dataset_path,
                                      'audio',
                                      track_filename)
        files_not_found = []
        try:
            estimated_segments = CHM.segmenter(
                audio_filename,
                sample_rate=None,
                features=features,
                criteria=criteria,
                coefficients=coefficients)[0]
            # Get ground truth structure for evaluation
            reference_segments_filename = os.path.join(
                harmonix_dataset_path,
                'references',
                'segments',
                os.path.splitext(track_filename)[0] + '.txt')

            reference_segments = read_ref_segments_harmonix(
                reference_segments_filename)[0]

            # Evaluate segments
            P05, R05, F05 = mir_eval.segment.detection(
                reference_segments, estimated_segments, window=0.5)
            P3, R3, F3 = mir_eval.segment.detection(
                reference_segments, estimated_segments, window=3)

            F05_vector = np.append(F05_vector, F05)
            P05_vector = np.append(P05_vector, P05)
            R05_vector = np.append(R05_vector, R05)
            F3_vector = np.append(F3_vector, F3)
            P3_vector = np.append(P3_vector, P3)
            R3_vector = np.append(R3_vector, R3)
        except FileNotFoundError:
            files_not_found.append(track_filename)
            print('Audio or reference segments file not found : skipping...')

    print('Segmentation results')
    print('F0_5\t\t', round(100*np.mean(F05_vector), ndigits=1), '%')
    print('P0_5\t\t', round(100*np.mean(P05_vector), ndigits=1), '%')
    print('R0_5\t\t', round(100*np.mean(R05_vector), ndigits=1), '%')
    print('F3\t\t', round(100*np.mean(F3_vector), ndigits=1), '%')
    print('P3\t\t', round(100*np.mean(P3_vector), ndigits=1), '%')
    print('R3\t\t', round(100*np.mean(R3_vector), ndigits=1), '%')
    print('The following files were not found and could not be computed')
    print(files_not_found)

    return (F05_vector, P05_vector, R05_vector,
            F3_vector, P3_vector, R3_vector)


def read_ref_beats_harmonix(fname):

    beats_vector = np.array([])
    downbeats_vector = np.array([])

    with open(fname) as f:
        for line in f.readlines():
            beats_vector = np.append(beats_vector, float(line.split()[0]))
            if int(line.split()[1]) == 1:
                downbeats_vector = np.append(downbeats_vector,
                                             float(line.split()[0]))
    return beats_vector, downbeats_vector


def read_ref_segments_harmonix(fname):

    estimated_boundaries = mir_eval.io.load_labeled_events(fname)[0]
    estimated_segments = (np.zeros((len(estimated_boundaries)-1, 2)),
                          [])
    for i in range(len(estimated_boundaries) - 1):
        estimated_segments[0][i, 0] = estimated_boundaries[i]
        estimated_segments[0][i, 1] = estimated_boundaries[i+1]
        estimated_segments[1].append('Segment{}'.format(i+1))
    return estimated_segments


def DTL1000_segmentation_loop(DTL1000_dataset_path):

    DTL1000_csv_file = os.path.join(DTL1000_dataset_path,
                                    'DTL_1000_segmentations.csv')
    with open(DTL1000_csv_file) as f:
        lines = f.readlines()

    # Get all song filenames and ground truth boundaries
    # Initialization
    filenames_list = [lines[1].split(';')[4].strip('"')[:-4]]
    ground_truth_boundaries_list = []
    song_ground_truth_boundaries_list = [float(lines[1].split(';')[0])]

    for line_idx in range(2, len(lines)):
        line = lines[line_idx].split(';')

        if line[4].strip('"')[:-4] != filenames_list[-1]:
            filenames_list.append(line[4].strip('"')[:-4])
            ground_truth_boundaries_list.append(
                song_ground_truth_boundaries_list)
            song_ground_truth_boundaries_list = [float(line[0])]
        else:
            song_ground_truth_boundaries_list.append(float(line[0]))
    # Last line
    ground_truth_boundaries_list.append(song_ground_truth_boundaries_list)

    F05_vector = np.array([])
    P05_vector = np.array([])
    R05_vector = np.array([])
    F3_vector = np.array([])
    P3_vector = np.array([])
    R3_vector = np.array([])

    for song_number in range(len(filenames_list)):
        print('({}) Computing: {}'.format(song_number,
                                          filenames_list[song_number]))

        # Compute estimated segments boundaries
        try:
            audio_fname = os.path.join(DTL1000_dataset_path,
                                       'audio',
                                       filenames_list[song_number] + '.wav')
            estimated_segments = CHM.segmenter(audio_fname)[0]
        except FileNotFoundError:
            audio_fname = os.path.join(DTL1000_dataset_path,
                                       'audio',
                                       filenames_list[song_number] + '.aiff')
            estimated_segments = CHM.segmenter(audio_fname)[0]

        # Get ground truth boundaries for evaluation
        reference_segments_fname = os.path.join(
            DTL1000_dataset_path,
            'references',
            'segments',
            filenames_list[song_number] + '.lab')
        reference_segments = mir_eval.io.load_labeled_intervals(
                reference_segments_fname)[0]

        # Evaluate segments
        P05, R05, F05 = mir_eval.segment.detection(
            reference_segments, estimated_segments, window=0.5)
        P3, R3, F3 = mir_eval.segment.detection(
            reference_segments, estimated_segments, window=3)

        F05_vector = np.append(F05_vector, F05)
        P05_vector = np.append(P05_vector, P05)
        R05_vector = np.append(R05_vector, R05)
        F3_vector = np.append(F3_vector, F3)
        P3_vector = np.append(P3_vector, P3)
        R3_vector = np.append(R3_vector, R3)

    print('Segmentation results')
    print('F0_5\t\t', round(100*np.mean(F05_vector), ndigits=1), '%')
    print('P0_5\t\t', round(100*np.mean(P05_vector), ndigits=1), '%')
    print('R0_5\t\t', round(100*np.mean(R05_vector), ndigits=1), '%')
    print('F3\t\t', round(100*np.mean(F3_vector), ndigits=1), '%')
    print('P3\t\t', round(100*np.mean(P3_vector), ndigits=1), '%')
    print('R3\t\t', round(100*np.mean(R3_vector), ndigits=1), '%')

    return (F05_vector, P05_vector, R05_vector,
            F3_vector, P3_vector, R3_vector)


# =============================================================================
# CODE
# =============================================================================

# beatles_segmentation_loop('/media/leo/42A45DCCA45DC359/'
#                           'MIR_DATASETS/Beatles_Helene',
#                           features=['chroma_stft', 'mfcc'])
# harmonix_segmentation_loop('/media/leo/42A45DCCA45DC359/MIR_DATASETS/Harmonix/'
#                            'harmonixset-master/dataset')
DTL1000_segmentation_loop('/media/leo/42A45DCCA45DC359/MIR_DATASETS/DTL1000')
