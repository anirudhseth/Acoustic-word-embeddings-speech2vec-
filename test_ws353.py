import sounddevice as sd
import numpy as np
import librosa
import os
import warnings
import torch
import speechpy
from tqdm import tqdm
import pandas as pd

################
words353_filename = "./reza/wordsim353_sim_rel/wordsim353_agreed.txt"
cname = ['desc','word1','word2','extra']
df = pd.read_csv(words353_filename, skiprows=11, sep='\t', names = cname )
 
column_values = df[["word1", "word2"]].values.ravel()
words_req =  pd.unique(column_values)

words_mfccs = {}
for w_req in words_req:
    words_mfccs[w_req.upper()] = []
##################


warnings.filterwarnings("ignore", category=UserWarning)
### Go to https://github.com/CorentinJ/librispeech-alignments


librispeech_root = "../LibriSpeech-alignments"    # Replace with yours
librispeech_root_audio = "../LibriSpeech"    # Replace with yours


def split_on_silences(audio_fpath, words, end_times):
    # Load the audio waveform
    sample_rate = 16000     # Sampling rate of LibriSpeech 
    win_lenth = 0.032  #windows length for mfccs, 25ms of signal (400 ffts)
    win_shift = 0.010 #10ms windows shift

    wav, _ = librosa.load(audio_fpath, sample_rate)
    words = np.array(words)
    #print('words',words)
    start_times = np.array([0.0] + end_times[:-1])
    #print('start times',start_times)
    end_times = np.array(end_times)
    #print('end times:',end_times)
    assert len(words) == len(end_times) == len(start_times)
    assert words[0] == '' and words[-1] == ''
    if end_times[0] < 1e-6:
        end_times[0] = 0.1



    wavs = [wav[(start_times[t]*sample_rate).astype(np.int):(end_times[t]*sample_rate).astype(np.int)] for t in range(len(start_times))]
    mfccs=[librosa.feature.mfcc(wavs[i],sr=sample_rate, n_mfcc=13, norm='ortho', hop_length=int(win_shift*sample_rate), n_fft=int(win_lenth*sample_rate)) for i in range(len(wavs))]
    mfccs_extend = [np.pad(mfccs[i], ((0, 0),(0, 400 - mfccs[i].shape[1])),'constant') for i in range(len(mfccs))]

    #mfcc_cmvn = [speechpy.processing.cmvnw(mfccs_extend[i],win_size=301,variance_normalization=True) for i in range(len(mfccs_extend))]
    #mfcc_cmvn = [speechpy.processing.cmvn(mfccs_extend[i].T, variance_normalization=True).T for i in range(len(mfccs_extend))]
    ### extend cmvn on mfccs ###
    mfcc_cmvn_modified = [speechpy.processing.cmvn(mfccs[i].T, variance_normalization=True).T for i in range(len(mfccs))]
    

    texts = [t for t in words]

    return wavs, texts, mfccs, mfcc_cmvn_modified

# Select sets (e.g. dev-clean, train-other-500, ...)

for set_name in ['dev-clean']:
    set_dir = os.path.join(librispeech_root, set_name)
    set_dir_audio = os.path.join(librispeech_root_audio, set_name)

    if not os.path.isdir(set_dir):
        continue
    
    # Select speakers
    for speaker_id in tqdm(os.listdir(set_dir)):
        #print("-------", speaker_id)
        if speaker_id == '.DS_Store':
            continue
        speaker_dir = os.path.join(set_dir, speaker_id)
        speaker_dir_audio = os.path.join(set_dir_audio, speaker_id)
        # Select books
        for book_id in os.listdir(speaker_dir):
            if book_id == '.DS_Store':
                continue
            book_dir = os.path.join(speaker_dir, book_id)
            book_dir_audio = os.path.join(speaker_dir_audio, book_id)
            
            
            # Get the alignment file
            alignment_fpath = os.path.join(book_dir, "%s-%s.alignment.txt" % 
                                            (speaker_id, book_id))
            if not os.path.exists(alignment_fpath):
                raise Exception("Alignment file not found. Did you download and merge the txt "
                                "alignments with your LibriSpeech dataset?")
            
            # Parse each utterance present in the file
            alignment_file = open(alignment_fpath, "r")
            for line in alignment_file:
                
                # Retrieve the utterance id, the words as a list and the end_times as a list
                utterance_id, words, end_times = line.strip().split(' ')
                words = words.replace('\"', '').split(',')
                #endtimes = end_times.replace('\"', '').split(',')
                # print(words)
                end_times = [float(e) for e in end_times.replace('\"', '').split(',')]
                audio_fpath = os.path.join(book_dir_audio, utterance_id + '.flac')
                # Check if the one of the requested list of available here
                
                ########################
                # find if the w353 exists in this dev-clean file 
                indexPosList = []
                for i, w in enumerate(words):
                    for w_req in words_req:
                        if w == w_req.upper():
                            #print('found', i)
                            indexPosList.append(i)

                # If w353 exists in datasest(dev-clean), computer mfccs and add it to the dictionary, words_mfccs
                if indexPosList:
                    wavs, texts, mfccs, mfcc_cmvn = split_on_silences(audio_fpath, words, end_times)
                     
                    for i, w in enumerate(texts):
                        for w_req in words_req:
                            if w.upper() == w_req.upper():
                                words_mfccs[w_req.upper()].append(mfccs[i])
                ########################

            alignment_file.close()



torch.save(words_mfccs, 'ws353_mfcc_features.pt')



        