import sounddevice as sd
import numpy as np
import librosa
import os


### Go to https://github.com/CorentinJ/librispeech-alignments


librispeech_root = "./LibriSpeech"    # Replace with yours

def split_on_silences(audio_fpath, words, end_times):
    # Load the audio waveform
    sample_rate = 16000     # Sampling rate of LibriSpeech 
    wav, _ = librosa.load(audio_fpath, sample_rate)
    words = np.array(words)
    print('words',words)
    start_times = np.array([0.0] + end_times[:-1])
    print('start times',start_times)
    end_times = np.array(end_times)
    print('end times:',end_times)
    assert len(words) == len(end_times) == len(start_times)
    assert words[0] == '' and words[-1] == ''
    wavs = [wav[(start_times[t]*sample_rate).astype(np.int):(end_times[t]*sample_rate).astype(np.int)] for t in range(len(start_times))]
    # mfcc=  run mfcc
    texts = [t for t in words]

    return wavs, texts


w=[]
t=[]
# Select sets (e.g. dev-clean, train-other-500, ...)
for set_name in ['dev-clean']:
    set_dir = os.path.join(librispeech_root, set_name)
    if not os.path.isdir(set_dir):
        continue
    
    # Select speakers
    for speaker_id in os.listdir(set_dir):
        speaker_dir = os.path.join(set_dir, speaker_id)
        
        # Select books
        for book_id in os.listdir(speaker_dir):
            book_dir = os.path.join(speaker_dir, book_id)
            
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
                # print(words)
                end_times = [float(e) for e in end_times.replace('\"', '').split(',')]
                audio_fpath = os.path.join(book_dir, utterance_id + '.flac')
                
                # Split utterances on silences
                wavs, texts = split_on_silences(audio_fpath, words, end_times)
                w.append(wavs)
                t.append(texts)
                
            alignment_file.close()