import librosa
from glob import glob
import pandas as pd
from scipy import signal


num_mfcc=20
sample_rate=22050
n_fft=2048
hop_length=512

my_csv={"filename": [], "length": [], "chroma_stft_mean": [], "chroma_stft_var": [], "rms_mean": [], "rms_var": [], "spectral_centroid_mean": [],
        "spectral_centroid_var": [], "spectral_bandwidth_mean": [], "spectral_bandwidth_var": [], "rolloff_mean": [], "rolloff_var": [],
        "zero_crossing_rate_mean": [], "zero_crossing_rate_var": [], "harmony_mean": [], "harmony_var": [], "perceptr_mean": [],
        "perceptr_var": [], "tempo": [], "mfcc1_mean": [], "mfcc1_var" : [], "mfcc2_mean" : [], "mfcc2_var" : [],
        "mfcc3_mean" : [], "mfcc3_var" : [], "mfcc4_mean" : [], "mfcc4_var" : [], "mfcc5_mean" : [], 
        "mfcc5_var" : [], "mfcc6_mean" : [], "mfcc6_var" : [], "mfcc7_mean" : [], "mfcc7_var" : [],
        "mfcc8_mean" : [], "mfcc8_var" : [], "mfcc9_mean" : [], "mfcc9_var" : [], "mfcc10_mean" : [], 
        "mfcc10_var" : [], "mfcc11_mean" : [], "mfcc11_var" : [], "mfcc12_mean" : [], "mfcc12_var" : [], 
        "mfcc13_mean" : [], "mfcc13_var" : [], "mfcc14_mean" : [], "mfcc14_var" : [], "mfcc15_mean" : [], 
        "mfcc15_var" : [], "mfcc16_mean" : [], "mfcc16_var" : [], "mfcc17_mean" : [], "mfcc17_var" : [], 
        "mfcc18_mean" : [], "mfcc18_var" : [], "mfcc19_mean" : [], "mfcc19_var" : [], "mfcc20_mean" : [], 
        "mfcc20_var":[], "label":[]}
my_3_csv=my_csv.copy()


num_segment=10
samples_per_segment = int(sample_rate*30/num_segment)
dataset_path="Data/MyGeneras"
audio_files = glob(dataset_path + "/*/*")
genre = glob(dataset_path + "/*")
n_genres=len(genre)
genre=[genre[x].split('/')[-1] for x in range(n_genres)]

num_segment=10

samples_per_segment = int(sample_rate*30/num_segment)
genre=""
for f in sorted(audio_files):
    if genre!=f.split('/')[-2]:
        genre=f.split('/')[-2]
        print("Procesassing " + genre + "...")
    fname=f.split('/')[-1]

    print(fname)
    try:
        y, sr = librosa.load(f, sr=sample_rate)
    except Exception as e:
        print(e)
        continue
    
    for n in range(num_segment):
        y_seg = y[samples_per_segment*n: samples_per_segment*(n+1)]

        chroma_hop_length = 512
        chromagram = librosa.feature.chroma_stft(y=y_seg, sr=sample_rate, hop_length=chroma_hop_length)
        my_3_csv["chroma_stft_mean"].append(chromagram.mean())
        my_3_csv["chroma_stft_var"].append(chromagram.var())


        RMSEn= librosa.feature.rms(y=y_seg)
        my_3_csv["rms_mean"].append(RMSEn.mean())
        my_3_csv["rms_var"].append(RMSEn.var())


        spec_cent=librosa.feature.spectral_centroid(y=y_seg)
        my_3_csv["spectral_centroid_mean"].append(spec_cent.mean())
        my_3_csv["spectral_centroid_var"].append(spec_cent.var())


        spec_band=librosa.feature.spectral_bandwidth(y=y_seg,sr=sample_rate)
        my_3_csv["spectral_bandwidth_mean"].append(spec_band.mean())
        my_3_csv["spectral_bandwidth_var"].append(spec_band.var())

        spec_roll=librosa.feature.spectral_rolloff(y=y_seg,sr=sample_rate)
        my_3_csv["rolloff_mean"].append(spec_roll.mean())
        my_3_csv["rolloff_var"].append(spec_roll.var())

        zero_crossing=librosa.feature.zero_crossing_rate(y=y_seg)
        my_3_csv["zero_crossing_rate_mean"].append(zero_crossing.mean())
        my_3_csv["zero_crossing_rate_var"].append(zero_crossing.var())

        harmony, perceptr = librosa.effects.hpss(y=y_seg)
        my_3_csv["harmony_mean"].append(harmony.mean())
        my_3_csv["harmony_var"].append(harmony.var())
        my_3_csv["perceptr_mean"].append(perceptr.mean())
        my_3_csv["perceptr_var"].append(perceptr.var())


        tempo, _ = librosa.beat.beat_track(y=y_seg, sr=sample_rate)
        my_3_csv["tempo"].append(tempo)

        mfcc=librosa.feature.mfcc(y=y_seg,sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc=mfcc.T
        fseg_name='.'.join(fname.split('.')[:2])+f'.{n}.wav'
        my_3_csv["filename"].append(fseg_name)
        my_3_csv["label"].append(genre)
        for x in range(20):
            feat1 = "mfcc" + str(x+1) + "_mean"
            feat2 = "mfcc" + str(x+1) + "_var"
            my_3_csv[feat1].append(mfcc[:,x].mean())
            my_3_csv[feat2].append(mfcc[:,x].var())
        
        my_3_csv["length"].append(66149)
    
    
    

df = pd.DataFrame(my_3_csv)
df.to_csv('myfeatures_3_sec.csv', index=False)

