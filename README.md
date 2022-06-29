# **Sebastian Ramos and Tomas Vera Music Recommendation Project**

In this project, we created a Python program that recommends music recordings based on their similarity to a reference (query) recording. Our recommendation function follows the following format:


def MusicRecommender(queryfilename, Nrecommendations)

where

queryfilename = path and filename of a single query recording

Nrecommendations = number of recommendations to return (you can assume this is a number between 1 and 5)

Project datafiles provided by Yon Visell




```python
import numpy as np
import matplotlib.pyplot as plt
import essentia.standard as es
import IPython as ipy
```


```python
import glob

#load files into dictionary with key value corresponding to genre
mediaDir = './music_dataFolder'
files = {}
files['classical'] = glob.glob(mediaDir + '/classical/'+'*.wav')
files['jazz'] = glob.glob(mediaDir + '/jazz/'+'*.wav')
files['rockblues'] = glob.glob(mediaDir + '/rockblues/'+'*.wav')

classical_names = [file.split('/')[-1] for file in files['classical']]
jazz_names = [file.split('/')[-1] for file in files['jazz']]
rockblues_names = [file.split('/')[-1] for file in files['rockblues']]

song_names = classical_names + jazz_names + rockblues_names

Fs = 44100
Ts = 1/Fs 

print(files)
```

    {'classical': ['./music_dataFolder/classical/classical1.wav', './music_dataFolder/classical/classical2.wav', './music_dataFolder/classical/classical.wav', './music_dataFolder/classical/copland.wav', './music_dataFolder/classical/copland2.wav', './music_dataFolder/classical/vlobos.wav', './music_dataFolder/classical/brahms.wav', './music_dataFolder/classical/debussy.wav', './music_dataFolder/classical/bartok.wav'], 'jazz': ['./music_dataFolder/jazz/ipanema.wav', './music_dataFolder/jazz/duke.wav', './music_dataFolder/jazz/moanin.wav', './music_dataFolder/jazz/russo.wav', './music_dataFolder/jazz/jazz1.wav', './music_dataFolder/jazz/mingus1.wav', './music_dataFolder/jazz/tony.wav', './music_dataFolder/jazz/misirlou.wav', './music_dataFolder/jazz/corea1.wav', './music_dataFolder/jazz/beat.wav', './music_dataFolder/jazz/georose.wav', './music_dataFolder/jazz/caravan.wav', './music_dataFolder/jazz/bmarsalis.wav', './music_dataFolder/jazz/mingus.wav', './music_dataFolder/jazz/unpoco.wav', './music_dataFolder/jazz/jazz.wav', './music_dataFolder/jazz/corea.wav'], 'rockblues': ['./music_dataFolder/rockblues/rock2.wav', './music_dataFolder/rockblues/hendrix.wav', './music_dataFolder/rockblues/beatles.wav', './music_dataFolder/rockblues/rock.wav', './music_dataFolder/rockblues/blues.wav', './music_dataFolder/rockblues/chaka.wav', './music_dataFolder/rockblues/redhot.wav', './music_dataFolder/rockblues/u2.wav', './music_dataFolder/rockblues/led.wav', './music_dataFolder/rockblues/eguitar.wav', './music_dataFolder/rockblues/cure.wav']}


# Feature Extraction and Normalization

The recomendation engine compares the similarity between the query song and the database songs of the following feature set:


1.   Mean Spectral Centroid
2.   Standard Deviation of RMS Amplitude
3.   Mean Spectral Rolloff
4.   Mean Spectral Flatness

For each feature in the feature set, we normalize the feature vector to give equal weight to the selected feature, F, as shown:

$F_{Norm}=\frac{F_{Val} - F_{Mean}}{F_{std}}$


## Mean Spectral Centroid

The magnitude spectrum variation of the spectral centroid is calculated in order to determine the average frequency in which the weighted spectrum resides.
The average of the spectral centroids for each block is determined for each audio file as shown:

$v_{SC,block}(i)=\frac{\sum_{k=0}^{block\_size-1} k|X(i,k)|}{\sum_{k=0}^{block\_size-1} |X(i, k)|}$

<br>

$v_{SC,mean}=\frac{\sum_{i=0}^{n\_blocks-1} v_{SC,block}(i)}{n\_blocks}$


```python
def calculate_mean_spectral_centroid(song):
  #get number of samples for the current audio recording
  n_samples = song.shape[0]

  #define basic parameters for block based analysis of recording
  block_size = 100000 
  overlap = 0.25  #25% overlap between blocks
  block_step = (1-overlap)*block_size  
  n_blocks = np.ceil(n_samples/block_step)

  #initialize zero np array to store spectral centroid for each block
  centroid_list = np.zeros(int(n_blocks))

  #loop through all predefined blocks of audio recording
  for i in range(int(n_blocks)):

    #calculate start and end sample step for current block
    block_s = int(i*block_step)
    block_e = int(min(block_s + block_size, n_samples))

    #calculate audio data for the specific block
    block = song[block_s:block_e]

    #define number of samples for the current block
    Nsamples = block.shape[0]

    ff = np.arange(0,int(Fs),Fs/Nsamples)

    #calculate the fast fourier transform for the current block
    block_fft = np.abs(np.fft.fft(block))

    #only display half of the samples of the audio file for better FFT graph plot
    ff = ff[:int(Nsamples/2)]
    block_fft = block_fft[:int(Nsamples/2)]

    #calculate current block's spectral centroid
    centroid_list[i] = np.sum(ff*block_fft)/np.sum(block_fft)

  #take the mean of all of the audio file's block centroids and round to two decimal places
  mean_spectral_centroid = round(np.mean(centroid_list), 2)

  return mean_spectral_centroid
```


```python
#define dictionary for calculated mean spectral centroid values for each genre
mean_spectral_centroids = {
    'classical': [],
    'jazz': [],
    'rockblues': []
}

#loop through dictionary of audio recording genres
for k, v in files.items():

  #loop through each recording for the corresponding genre
  for i in range(len(v)):

    #print basic information of audio file
    print()
    print("-----" + k + " file " + str(i+1) + "-----")
    print(v[i])
    
    #load audio recording through essentia
    song = es.MonoLoader(filename=v[i], sampleRate=Fs)()

    mean_centroid_for_file = calculate_mean_spectral_centroid(song)

    #print the mean spectral centroid for current file
    print(f'{mean_centroid_for_file} Hz')

    #append the mean spectral centroid to the genre mean spectral centroid dictionary
    mean_spectral_centroids[k].append(mean_centroid_for_file)
    
#print the dictionary of spectral centroids
print(mean_spectral_centroids)
```

    
    -----classical file 1-----
    ./music_dataFolder/classical/classical1.wav
    1440.18 Hz
    
    -----classical file 2-----
    ./music_dataFolder/classical/classical2.wav
    1446.49 Hz
    
    -----classical file 3-----
    ./music_dataFolder/classical/classical.wav
    1607.0 Hz
    
    -----classical file 4-----
    ./music_dataFolder/classical/copland.wav
    1872.75 Hz
    
    -----classical file 5-----
    ./music_dataFolder/classical/copland2.wav
    1471.75 Hz
    
    -----classical file 6-----
    ./music_dataFolder/classical/vlobos.wav
    1450.13 Hz
    
    -----classical file 7-----
    ./music_dataFolder/classical/brahms.wav
    1226.17 Hz
    
    -----classical file 8-----
    ./music_dataFolder/classical/debussy.wav
    1528.46 Hz
    
    -----classical file 9-----
    ./music_dataFolder/classical/bartok.wav
    1826.7 Hz
    
    -----jazz file 1-----
    ./music_dataFolder/jazz/ipanema.wav
    2404.56 Hz
    
    -----jazz file 2-----
    ./music_dataFolder/jazz/duke.wav
    1473.18 Hz
    
    -----jazz file 3-----
    ./music_dataFolder/jazz/moanin.wav
    1584.5 Hz
    
    -----jazz file 4-----
    ./music_dataFolder/jazz/russo.wav
    1921.62 Hz
    
    -----jazz file 5-----
    ./music_dataFolder/jazz/jazz1.wav
    2474.58 Hz
    
    -----jazz file 6-----
    ./music_dataFolder/jazz/mingus1.wav
    1742.44 Hz
    
    -----jazz file 7-----
    ./music_dataFolder/jazz/tony.wav
    2324.38 Hz
    
    -----jazz file 8-----
    ./music_dataFolder/jazz/misirlou.wav
    1782.58 Hz
    
    -----jazz file 9-----
    ./music_dataFolder/jazz/corea1.wav
    1696.76 Hz
    
    -----jazz file 10-----
    ./music_dataFolder/jazz/beat.wav
    2254.55 Hz
    
    -----jazz file 11-----
    ./music_dataFolder/jazz/georose.wav
    2633.23 Hz
    
    -----jazz file 12-----
    ./music_dataFolder/jazz/caravan.wav
    1219.48 Hz
    
    -----jazz file 13-----
    ./music_dataFolder/jazz/bmarsalis.wav
    2360.85 Hz
    
    -----jazz file 14-----
    ./music_dataFolder/jazz/mingus.wav
    1782.91 Hz
    
    -----jazz file 15-----
    ./music_dataFolder/jazz/unpoco.wav
    2434.07 Hz
    
    -----jazz file 16-----
    ./music_dataFolder/jazz/jazz.wav
    1859.11 Hz
    
    -----jazz file 17-----
    ./music_dataFolder/jazz/corea.wav
    1214.99 Hz
    
    -----rockblues file 1-----
    ./music_dataFolder/rockblues/rock2.wav
    2567.87 Hz
    
    -----rockblues file 2-----
    ./music_dataFolder/rockblues/hendrix.wav
    1874.48 Hz
    
    -----rockblues file 3-----
    ./music_dataFolder/rockblues/beatles.wav
    2091.79 Hz
    
    -----rockblues file 4-----
    ./music_dataFolder/rockblues/rock.wav
    2690.58 Hz
    
    -----rockblues file 5-----
    ./music_dataFolder/rockblues/blues.wav
    2154.8 Hz
    
    -----rockblues file 6-----
    ./music_dataFolder/rockblues/chaka.wav
    2479.06 Hz
    
    -----rockblues file 7-----
    ./music_dataFolder/rockblues/redhot.wav
    2673.32 Hz
    
    -----rockblues file 8-----
    ./music_dataFolder/rockblues/u2.wav
    2277.85 Hz
    
    -----rockblues file 9-----
    ./music_dataFolder/rockblues/led.wav
    2647.33 Hz
    
    -----rockblues file 10-----
    ./music_dataFolder/rockblues/eguitar.wav
    2570.5 Hz
    
    -----rockblues file 11-----
    ./music_dataFolder/rockblues/cure.wav
    2125.07 Hz
    {'classical': [1440.18, 1446.49, 1607.0, 1872.75, 1471.75, 1450.13, 1226.17, 1528.46, 1826.7], 'jazz': [2404.56, 1473.18, 1584.5, 1921.62, 2474.58, 1742.44, 2324.38, 1782.58, 1696.76, 2254.55, 2633.23, 1219.48, 2360.85, 1782.91, 2434.07, 1859.11, 1214.99], 'rockblues': [2567.87, 1874.48, 2091.79, 2690.58, 2154.8, 2479.06, 2673.32, 2277.85, 2647.33, 2570.5, 2125.07]}


Feature mean subtraction and normalization by the standard deviation


```python
centroids = []

# Placing mean spectral centroid values into a single continuous list
for genre in mean_spectral_centroids.keys():
  for value in mean_spectral_centroids[genre]:
    centroids.append(value)

# Finding the mean and standard deviation of the list of spectral centroids
mean_of_centroids = np.round(np.mean(centroids), 2)
std_of_centroids = np.round(np.std(centroids), 2)

# The spectral centroid values are normalized using mean subtraction and normalization
centroid_norm_list = np.round((centroids - mean_of_centroids) / std_of_centroids, 3)

print(centroid_norm_list)
```

    [-1.171 -1.157 -0.808 -0.229 -1.102 -1.149 -1.637 -0.979 -0.329  0.929
     -1.099 -0.857 -0.123  1.081 -0.513  0.754 -0.425 -0.612  0.602  1.427
     -1.651  0.834 -0.425  0.993 -0.259 -1.661  1.284 -0.225  0.248  1.551
      0.385  1.091  1.514  0.653  1.457  1.29   0.32 ]


## Standard Deviation of RMS Amplitude

The standard deviation of the RMS amplitude of each block is determined in order to analyze the variation of the frequency components around the average frequency magnitude. It is calculated by the following:

$v_{RMS,block}(i)=20log_{10}(\sqrt{\frac{\sum_{k=0}^{block\_size-1} |X(i, k)|^2}{block\_size}})$

<br>

$v_{RMS,SD}=\sqrt{\frac{\sum_{i=0}^{n\_blocks-1} (v_{RMS,block}(i) - v_{RMS,mean})^2}{n\_blocks}}$


```python
def calculate_std_rms_amp(song):
 #get number of samples for the current audio recording
  n_samples = song.shape[0]

  #define basic parameters for block based analysis of recording
  block_size = 100000
  overlap = 0.25 #25% overlap between blocks
  block_step = (1-overlap)*block_size
  n_blocks = np.ceil(n_samples/block_step)

  #initialize zero np array to store RMS amplitude for each block
  rms_amp = np.zeros(int(n_blocks))

  #loop through all predefined blocks of audio recording
  for i in range(int(n_blocks)):

    #calculate start and end sample step for current block
    block_s = int(i*block_step)
    block_e = int(min(block_s + block_size, n_samples))

    #get audio data for the specific block
    block = song[block_s:block_e]

    #define number of samples for the current block
    Nsamples = block.shape[0]

    #calculate RMS amplitude for the current block
    rms = np.sqrt(np.mean(np.square(song[block_s:block_e])))

    #convert RMS amplitude units to decibels
    if rms<0.00001:
      rms = 0.00001
    db_rms = 20*np.log10(rms)

    #add calculated RMS amplitude to list of all block RMS amplitudes
    rms_amp[i] = db_rms
  
  #take the standard diviation of all blocks in the current audio file and round to two decimal places
  output_rms_amp = round(np.std(rms_amp), 2)
  return output_rms_amp
```


```python
#define dictionary for calculated standard deviation of RMS values for each genre
std_rms_amp = {
    'classical': [],
    'jazz': [],
    'rockblues': [],
}

#loop through dictionary of audio recording genres
for k, v in files.items():
  
  #loop through each recording for the corresponding genre
  for i in range(len(v)):

    #print basic information of audio file
    print()
    print("-----" + k + " file " + str(i+1) + "-----")
    print(v[i])
    
    #load audio recording through essentia
    song = es.MonoLoader(filename=v[i], sampleRate=Fs)()

    std_rms_for_file = calculate_std_rms_amp(song)

    #print the standard deviation for the RMS ampltude of the audio file among all blocks
    print(f'{std_rms_for_file} dB')

    #append the standard deviation for the RMS ampltude of the audio file to the genre standard diviation RMS amplitude dictionary
    std_rms_amp[k].append(std_rms_for_file)
  
#print the dictionary of standard deviation RMS amplitude values among all files in all genres
print(std_rms_amp)
```

    
    -----classical file 1-----
    ./music_dataFolder/classical/classical1.wav
    2.81 dB
    
    -----classical file 2-----
    ./music_dataFolder/classical/classical2.wav
    2.19 dB
    
    -----classical file 3-----
    ./music_dataFolder/classical/classical.wav
    4.2 dB
    
    -----classical file 4-----
    ./music_dataFolder/classical/copland.wav
    1.57 dB
    
    -----classical file 5-----
    ./music_dataFolder/classical/copland2.wav
    3.86 dB
    
    -----classical file 6-----
    ./music_dataFolder/classical/vlobos.wav
    2.24 dB
    
    -----classical file 7-----
    ./music_dataFolder/classical/brahms.wav
    3.22 dB
    
    -----classical file 8-----
    ./music_dataFolder/classical/debussy.wav
    5.13 dB
    
    -----classical file 9-----
    ./music_dataFolder/classical/bartok.wav
    2.21 dB
    
    -----jazz file 1-----
    ./music_dataFolder/jazz/ipanema.wav
    2.25 dB
    
    -----jazz file 2-----
    ./music_dataFolder/jazz/duke.wav
    1.34 dB
    
    -----jazz file 3-----
    ./music_dataFolder/jazz/moanin.wav
    2.9 dB
    
    -----jazz file 4-----
    ./music_dataFolder/jazz/russo.wav
    1.76 dB
    
    -----jazz file 5-----
    ./music_dataFolder/jazz/jazz1.wav
    0.63 dB
    
    -----jazz file 6-----
    ./music_dataFolder/jazz/mingus1.wav
    3.77 dB
    
    -----jazz file 7-----
    ./music_dataFolder/jazz/tony.wav
    1.26 dB
    
    -----jazz file 8-----
    ./music_dataFolder/jazz/misirlou.wav
    0.81 dB
    
    -----jazz file 9-----
    ./music_dataFolder/jazz/corea1.wav
    0.78 dB
    
    -----jazz file 10-----
    ./music_dataFolder/jazz/beat.wav
    0.69 dB
    
    -----jazz file 11-----
    ./music_dataFolder/jazz/georose.wav
    0.28 dB
    
    -----jazz file 12-----
    ./music_dataFolder/jazz/caravan.wav
    0.9 dB
    
    -----jazz file 13-----
    ./music_dataFolder/jazz/bmarsalis.wav
    0.29 dB
    
    -----jazz file 14-----
    ./music_dataFolder/jazz/mingus.wav
    2.78 dB
    
    -----jazz file 15-----
    ./music_dataFolder/jazz/unpoco.wav
    1.24 dB
    
    -----jazz file 16-----
    ./music_dataFolder/jazz/jazz.wav
    1.14 dB
    
    -----jazz file 17-----
    ./music_dataFolder/jazz/corea.wav
    2.26 dB
    
    -----rockblues file 1-----
    ./music_dataFolder/rockblues/rock2.wav
    0.29 dB
    
    -----rockblues file 2-----
    ./music_dataFolder/rockblues/hendrix.wav
    0.5 dB
    
    -----rockblues file 3-----
    ./music_dataFolder/rockblues/beatles.wav
    1.41 dB
    
    -----rockblues file 4-----
    ./music_dataFolder/rockblues/rock.wav
    0.39 dB
    
    -----rockblues file 5-----
    ./music_dataFolder/rockblues/blues.wav
    2.28 dB
    
    -----rockblues file 6-----
    ./music_dataFolder/rockblues/chaka.wav
    0.65 dB
    
    -----rockblues file 7-----
    ./music_dataFolder/rockblues/redhot.wav
    0.28 dB
    
    -----rockblues file 8-----
    ./music_dataFolder/rockblues/u2.wav
    0.91 dB
    
    -----rockblues file 9-----
    ./music_dataFolder/rockblues/led.wav
    0.79 dB
    
    -----rockblues file 10-----
    ./music_dataFolder/rockblues/eguitar.wav
    0.51 dB
    
    -----rockblues file 11-----
    ./music_dataFolder/rockblues/cure.wav
    0.42 dB
    {'classical': [2.81, 2.19, 4.2, 1.57, 3.86, 2.24, 3.22, 5.13, 2.21], 'jazz': [2.25, 1.34, 2.9, 1.76, 0.63, 3.77, 1.26, 0.81, 0.78, 0.69, 0.28, 0.9, 0.29, 2.78, 1.24, 1.14, 2.26], 'rockblues': [0.29, 0.5, 1.41, 0.39, 2.28, 0.65, 0.28, 0.91, 0.79, 0.51, 0.42]}


Feature mean subtraction and normalization by the standard deviation


```python
rms_amp = []

# Placing standard deviatrion of the RMS amplitude values into a single continuous list
for genre in std_rms_amp.keys():
  for value in std_rms_amp[genre]:
    rms_amp.append(value)

# Finding the mean and standard deviation of the list of the RMS amplitude values
mean_of_rms_amp = np.round(np.mean(rms_amp), 2)
std_of_rms_amp = np.round(np.std(rms_amp), 2)

# The RMS amplitude values are normalized using mean subtraction and normalization
rms_norm_list = np.round((rms_amp - mean_of_rms_amp) / std_of_rms_amp, 3)

print(rms_norm_list)
```

    [ 0.935  0.435  2.056 -0.065  1.782  0.476  1.266  2.806  0.452  0.484
     -0.25   1.008  0.089 -0.823  1.71  -0.315 -0.677 -0.702 -0.774 -1.105
     -0.605 -1.097  0.911 -0.331 -0.411  0.492 -1.097 -0.927 -0.194 -1.016
      0.508 -0.806 -1.105 -0.597 -0.694 -0.919 -0.992]


## Mean Spectral Rolloff

The mean spectral rolloff is found to determine the bandwidth in which a certain percentage of the frequency spectrum resides. A value of 90% is chosen, and the calculation is implemented as follows:

$v_{SR,block}(i)=m|_{\sum_{k=0}^m|X(i,k)|\text{ = }P\cdot\sum_{k=0}^{block\_size-1}|X(i,k)|}$

<br>

$v_{SR,mean}=\frac{\sum_{i=0}^{n\_blocks-1} v_{SR,block}(i)}{n\_blocks}$


```python
def calculate_spectral_rolloff(song, BW):
#get number of samples for the current audio recording
  n_samples = song.shape[0]

  #define basic parameters for block based analysis of recording
  block_size = 100000
  overlap = 0.25 #25% overlap between blocks
  block_step = (1-overlap)*block_size
  n_blocks = np.ceil(n_samples/block_step)

  #initialize zero np array to store spectral rolloff for each block
  rolloff_list = np.zeros(int(n_blocks))
  rolloff = []

  #loop through all predefined blocks of audio recording
  for i in range(int(n_blocks)):

    #calculate start and end sample step for current block
    block_s = int(i*block_step)
    block_e = int(min(block_s + block_size, n_samples))

    #calculate audio data for the specific block
    block = song[block_s:block_e]

    #define number of samples for the current block
    Nsamples = block.shape[0]

    ff = np.arange(0,int(Fs),Fs/Nsamples)

    #calculate the fast fourier transform for the current block
    block_fft = np.abs(np.fft.fft(block))

    #only display half of the samples of the audio file for more accurate FFT
    ff = ff[:int(Nsamples/2)]
    block_fft = block_fft[:int(Nsamples/2)]

    #sum over the entire block's spectral range of FFT
    total_sum = np.sum(block_fft)


    #variable to hold the accumulated magnitude of the rolloff calculation
    accumilated_mag = 0
    
    #calculate the spectral rolloff of the block
    for increment, value in enumerate(block_fft):
      accumilated_mag += value
      if accumilated_mag >= BANDWIDTH_PERCENTAGE*total_sum:
        rolloff.append(increment)
        break

  #calculate the scaled percentage for the rolloff calculation
  scaled_percentage = np.average(rolloff) / len(block_fft)

  #calculate the rolloff frequency for the audio file, serving as the mean spectral rolloff
  output_spectral_rolloff = round(scaled_percentage * Fs, 2)

  return output_spectral_rolloff
```


```python
#define dictionary for calculated spectral rolloff values for each genre
spectral_rolloff = {
    'classical': [],
    'jazz': [],
    'rockblues': [],
}

#define the bandwidth percentage when calculating the spectral rolloff
BANDWIDTH_PERCENTAGE = 0.90

#loop through dictionary of audio recording genres
for k, v in files.items():
  
  #loop through each recording for the corresponding genre
  for i in range(len(v)):

    #print basic information of audio file
    print()
    print("-----" + k + " file " + str(i+1) + "-----")
    print(v[i])
    
    #load audio recording through essentia
    song = es.MonoLoader(filename=v[i], sampleRate=Fs)()

    rolloff_freq = calculate_spectral_rolloff(song, BANDWIDTH_PERCENTAGE)

    #print the mean spectral rolloff for the audio file
    print(f'{rolloff_freq} Hz')

    #append the mean spectral rolloff to the dictionary holding the genre spectral rolloffs 
    spectral_rolloff[k].append(rolloff_freq)

#print the dictionary of spectral rolloffs for all genres
print(spectral_rolloff)
```

    
    -----classical file 1-----
    ./music_dataFolder/classical/classical1.wav
    13766.86 Hz
    
    -----classical file 2-----
    ./music_dataFolder/classical/classical2.wav
    12165.68 Hz
    
    -----classical file 3-----
    ./music_dataFolder/classical/classical.wav
    17048.32 Hz
    
    -----classical file 4-----
    ./music_dataFolder/classical/copland.wav
    14536.87 Hz
    
    -----classical file 5-----
    ./music_dataFolder/classical/copland2.wav
    13685.7 Hz
    
    -----classical file 6-----
    ./music_dataFolder/classical/vlobos.wav
    12723.16 Hz
    
    -----classical file 7-----
    ./music_dataFolder/classical/brahms.wav
    10509.17 Hz
    
    -----classical file 8-----
    ./music_dataFolder/classical/debussy.wav
    13717.35 Hz
    
    -----classical file 9-----
    ./music_dataFolder/classical/bartok.wav
    16452.46 Hz
    
    -----jazz file 1-----
    ./music_dataFolder/jazz/ipanema.wav
    22320.42 Hz
    
    -----jazz file 2-----
    ./music_dataFolder/jazz/duke.wav
    13588.82 Hz
    
    -----jazz file 3-----
    ./music_dataFolder/jazz/moanin.wav
    14277.89 Hz
    
    -----jazz file 4-----
    ./music_dataFolder/jazz/russo.wav
    21317.25 Hz
    
    -----jazz file 5-----
    ./music_dataFolder/jazz/jazz1.wav
    26464.19 Hz
    
    -----jazz file 6-----
    ./music_dataFolder/jazz/mingus1.wav
    18481.78 Hz
    
    -----jazz file 7-----
    ./music_dataFolder/jazz/tony.wav
    25618.32 Hz
    
    -----jazz file 8-----
    ./music_dataFolder/jazz/misirlou.wav
    19922.18 Hz
    
    -----jazz file 9-----
    ./music_dataFolder/jazz/corea1.wav
    19931.67 Hz
    
    -----jazz file 10-----
    ./music_dataFolder/jazz/beat.wav
    23953.65 Hz
    
    -----jazz file 11-----
    ./music_dataFolder/jazz/georose.wav
    32395.23 Hz
    
    -----jazz file 12-----
    ./music_dataFolder/jazz/caravan.wav
    13316.36 Hz
    
    -----jazz file 13-----
    ./music_dataFolder/jazz/bmarsalis.wav
    27292.29 Hz
    
    -----jazz file 14-----
    ./music_dataFolder/jazz/mingus.wav
    19059.16 Hz
    
    -----jazz file 15-----
    ./music_dataFolder/jazz/unpoco.wav
    26639.77 Hz
    
    -----jazz file 16-----
    ./music_dataFolder/jazz/jazz.wav
    17194.3 Hz
    
    -----jazz file 17-----
    ./music_dataFolder/jazz/corea.wav
    11358.3 Hz
    
    -----rockblues file 1-----
    ./music_dataFolder/rockblues/rock2.wav
    27690.92 Hz
    
    -----rockblues file 2-----
    ./music_dataFolder/rockblues/hendrix.wav
    17733.0 Hz
    
    -----rockblues file 3-----
    ./music_dataFolder/rockblues/beatles.wav
    22248.25 Hz
    
    -----rockblues file 4-----
    ./music_dataFolder/rockblues/rock.wav
    26637.01 Hz
    
    -----rockblues file 5-----
    ./music_dataFolder/rockblues/blues.wav
    23150.66 Hz
    
    -----rockblues file 6-----
    ./music_dataFolder/rockblues/chaka.wav
    26092.91 Hz
    
    -----rockblues file 7-----
    ./music_dataFolder/rockblues/redhot.wav
    27187.65 Hz
    
    -----rockblues file 8-----
    ./music_dataFolder/rockblues/u2.wav
    24764.7 Hz
    
    -----rockblues file 9-----
    ./music_dataFolder/rockblues/led.wav
    16443.69 Hz
    
    -----rockblues file 10-----
    ./music_dataFolder/rockblues/eguitar.wav
    27060.66 Hz
    
    -----rockblues file 11-----
    ./music_dataFolder/rockblues/cure.wav
    22527.34 Hz
    {'classical': [13766.86, 12165.68, 17048.32, 14536.87, 13685.7, 12723.16, 10509.17, 13717.35, 16452.46], 'jazz': [22320.42, 13588.82, 14277.89, 21317.25, 26464.19, 18481.78, 25618.32, 19922.18, 19931.67, 23953.65, 32395.23, 13316.36, 27292.29, 19059.16, 26639.77, 17194.3, 11358.3], 'rockblues': [27690.92, 17733.0, 22248.25, 26637.01, 23150.66, 26092.91, 27187.65, 24764.7, 16443.69, 27060.66, 22527.34]}


Feature mean subtraction and normalization by the standard deviation


```python
rolloff = []

# Placing spectral rolloff values into a single continuous list
for genre in spectral_rolloff.keys():
  for value in spectral_rolloff[genre]:
    rolloff.append(value)

# Finding the mean and standard deviation of the list of the spectral rolloff values
mean_of_rolloff = np.round(np.mean(rolloff), 2)
std_of_rolloff = np.round(np.std(rolloff), 2)

# The spectral rolloff values are normalized using mean subtraction and normalization
spectral_rolloff_norm_list = np.round((rolloff - mean_of_rolloff) / std_of_rolloff, 3)

print(spectral_rolloff_norm_list)
```

    [-1.078 -1.356 -0.509 -0.945 -1.092 -1.259 -1.644 -1.087 -0.612  0.406
     -1.109 -0.99   0.232  1.125 -0.26   0.978 -0.01  -0.008  0.689  2.154
     -1.156  1.269 -0.16   1.156 -0.483 -1.496  1.338 -0.39   0.394  1.155
      0.55   1.061  1.251  0.83  -0.614  1.229  0.442]


## Mean Spectral Flatness

The specral flatness is defined as the ratio of the signal noisiness to signal tonalness. Higher values of spectral flatness would attribute to a noisier spectrum, while lower values will suggest a more tonal spectrum. The logarithmic magnitude spectrum is used:

$v_{SF,block}(i)=\frac{exp(\sum_{i=0}^{block\_size-1} log(|X(i, k)|))}{\sum_{i=0}^{block\_size-1} |X(i, k)|}$

<br>

$v_{SF,mean}=\frac{\sum_{i=0}^{n\_blocks-1} v_{SF,block}(i)}{n\_blocks}$


```python
def calculate_spectral_flatness(song):
#get number of samples for the current audio recording
  n_samples = song.shape[0]

  #define basic parameters for block based analysis of recording
  block_size = 100000
  overlap = 0.25
  block_step = (1-overlap)*block_size
  n_blocks = np.ceil(n_samples/block_step)

  #initialize zero np array to store spectral flatness for each block
  flatness_list = np.zeros(int(n_blocks))
  flatness = []

  #loop through all predefined blocks of audio recording
  for i in range(int(n_blocks)):

    #calculate start and end sample step for current block
    block_s = int(i*block_step)
    block_e = int(min(block_s + block_size, n_samples))

    #calculate audio data for the specific block
    block = song[block_s:block_e]

    #define number of samples for the current block
    Nsamples = block.shape[0]

    ff = np.arange(0,int(Fs),Fs/Nsamples)

    #calculate the fast fourier transform for the current block
    block_fft = np.abs(np.fft.fft(block))

    #only display half of the samples of the audio file to avoid repeating FFT
    ff = ff[:int(Nsamples/2)]
    block_fft = block_fft[:int(Nsamples/2)]

    #calculate log magnitude for the block's FFT
    log_mag = np.log(block_fft)

    #calculate the sum along the entire log magnitude of the FFT
    log_sum = np.sum(log_mag)

    #define exponent section of spectral flatness calculation
    exp = np.exp(1/len(block_fft)*log_sum)

    #calculate the sum along the entire frequency amplitudes of the FFT
    total_sum = np.sum(block_fft)

    #calculate current block's spectral flatness and append to flatness list
    flatness.append(exp / (1/len(block_fft)*total_sum))

  #calculate the mean spectral flatness along all blocks of the audio recording
  output_spectral_flatness = round(np.average(flatness), 6)

  return output_spectral_flatness
```


```python
#define dictionary for calculated spectral flatness values for each genre
spectral_flatness = {
    'classical': [],
    'jazz': [],
    'rockblues': [],
}

#loop through dictionary of audio recording genres
for k, v in files.items():
  
  #loop through each recording for the corresponding genre
  for i in range(len(v)):

    #print basic information of audio file
    print()
    print("-----" + k + " file " + str(i+1) + "-----")
    print(v[i])
    
    #load audio recording through essentia
    song = es.MonoLoader(filename=v[i], sampleRate=Fs)()

    mean_flatness = calculate_spectral_flatness(song)

    #print the current recording's mean spectral flatness
    print(mean_flatness)

    #append the mean spectral flatness to the genre mean spectral flatness dictionary
    spectral_flatness[k].append(mean_flatness)

#print the dictionary of mean spectral flatness
print(spectral_flatness)
```

    
    -----classical file 1-----
    ./music_dataFolder/classical/classical1.wav
    0.050081
    
    -----classical file 2-----
    ./music_dataFolder/classical/classical2.wav
    0.067264
    
    -----classical file 3-----
    ./music_dataFolder/classical/classical.wav
    0.070513
    
    -----classical file 4-----
    ./music_dataFolder/classical/copland.wav
    0.053465
    
    -----classical file 5-----
    ./music_dataFolder/classical/copland2.wav
    0.050819
    
    -----classical file 6-----
    ./music_dataFolder/classical/vlobos.wav
    0.043345
    
    -----classical file 7-----
    ./music_dataFolder/classical/brahms.wav
    0.041181
    
    -----classical file 8-----
    ./music_dataFolder/classical/debussy.wav
    0.050647
    
    -----classical file 9-----
    ./music_dataFolder/classical/bartok.wav
    0.046758
    
    -----jazz file 1-----
    ./music_dataFolder/jazz/ipanema.wav
    0.061072
    
    -----jazz file 2-----
    ./music_dataFolder/jazz/duke.wav
    0.044529
    
    -----jazz file 3-----
    ./music_dataFolder/jazz/moanin.wav
    0.05593
    
    -----jazz file 4-----
    ./music_dataFolder/jazz/russo.wav
    0.070319
    
    -----jazz file 5-----
    ./music_dataFolder/jazz/jazz1.wav
    0.066865
    
    -----jazz file 6-----
    ./music_dataFolder/jazz/mingus1.wav
    0.060528
    
    -----jazz file 7-----
    ./music_dataFolder/jazz/tony.wav
    0.07055
    
    -----jazz file 8-----
    ./music_dataFolder/jazz/misirlou.wav
    0.055627
    
    -----jazz file 9-----
    ./music_dataFolder/jazz/corea1.wav
    0.065826
    
    -----jazz file 10-----
    ./music_dataFolder/jazz/beat.wav
    0.066172
    
    -----jazz file 11-----
    ./music_dataFolder/jazz/georose.wav
    0.075322
    
    -----jazz file 12-----
    ./music_dataFolder/jazz/caravan.wav
    0.048848
    
    -----jazz file 13-----
    ./music_dataFolder/jazz/bmarsalis.wav
    0.06751
    
    -----jazz file 14-----
    ./music_dataFolder/jazz/mingus.wav
    0.06727
    
    -----jazz file 15-----
    ./music_dataFolder/jazz/unpoco.wav
    0.069288
    
    -----jazz file 16-----
    ./music_dataFolder/jazz/jazz.wav
    0.042168
    
    -----jazz file 17-----
    ./music_dataFolder/jazz/corea.wav
    0.072225
    
    -----rockblues file 1-----
    ./music_dataFolder/rockblues/rock2.wav
    0.078339
    
    -----rockblues file 2-----
    ./music_dataFolder/rockblues/hendrix.wav
    0.044682
    
    -----rockblues file 3-----
    ./music_dataFolder/rockblues/beatles.wav
    0.101521
    
    -----rockblues file 4-----
    ./music_dataFolder/rockblues/rock.wav
    0.072254
    
    -----rockblues file 5-----
    ./music_dataFolder/rockblues/blues.wav
    0.057363
    
    -----rockblues file 6-----
    ./music_dataFolder/rockblues/chaka.wav
    0.065912
    
    -----rockblues file 7-----
    ./music_dataFolder/rockblues/redhot.wav
    0.060787
    
    -----rockblues file 8-----
    ./music_dataFolder/rockblues/u2.wav
    0.076254
    
    -----rockblues file 9-----
    ./music_dataFolder/rockblues/led.wav
    0.049484
    
    -----rockblues file 10-----
    ./music_dataFolder/rockblues/eguitar.wav
    0.063069
    
    -----rockblues file 11-----
    ./music_dataFolder/rockblues/cure.wav
    0.059316
    {'classical': [0.050081, 0.067264, 0.070513, 0.053465, 0.050819, 0.043345, 0.041181, 0.050647, 0.046758], 'jazz': [0.061072, 0.044529, 0.05593, 0.070319, 0.066865, 0.060528, 0.07055, 0.055627, 0.065826, 0.066172, 0.075322, 0.048848, 0.06751, 0.06727, 0.069288, 0.042168, 0.072225], 'rockblues': [0.078339, 0.044682, 0.101521, 0.072254, 0.057363, 0.065912, 0.060787, 0.076254, 0.049484, 0.063069, 0.059316]}


Feature mean subtraction and normalization by the standard deviation


```python
flatness = []

# Placing spectral flatness values into a single continuous list
for genre in spectral_flatness.keys():
  for value in spectral_flatness[genre]:
    flatness.append(value)

# Finding the mean and standard deviation of the list of the spectral flatness values
mean_of_flatness = np.round(np.mean(flatness), 2)
std_of_flatness = np.round(np.std(flatness), 2)

# The spectral flatness values are normalized using mean subtraction and normalization
spectral_flatness_norm_list = np.round((flatness - mean_of_flatness) / std_of_flatness, 3)

print(spectral_flatness_norm_list)
```

    [-0.992  0.726  1.051 -0.653 -0.918 -1.665 -1.882 -0.935 -1.324  0.107
     -1.547 -0.407  1.032  0.686  0.053  1.055 -0.437  0.583  0.617  1.532
     -1.115  0.751  0.727  0.929 -1.783  1.222  1.834 -1.532  4.152  1.225
     -0.264  0.591  0.079  1.625 -1.052  0.307 -0.068]


# Music Recommender

Specify the music recommender function. Takes as input a query filename for the selected query song, **queryfilename**, and the number of recommendations that will be provided by the algorithm, **Nrecommendations**.

As our distance measure between features we use the squared euclidian distance metric:
$D=\sum_{k} (f_{norm} - g_{norm})^2$

Create a database of feature norms


```python
norm_database_features = []

# The normalized features are placed in a continuous list, matching indicies with the song name list
for i in range(len(centroid_norm_list)):
  norm_database_features.append([centroid_norm_list[i], rms_norm_list[i], spectral_rolloff_norm_list[i], spectral_flatness_norm_list[i]])

print(norm_database_features)
```

    [[-1.171, 0.935, -1.078, -0.992], [-1.157, 0.435, -1.356, 0.726], [-0.808, 2.056, -0.509, 1.051], [-0.229, -0.065, -0.945, -0.653], [-1.102, 1.782, -1.092, -0.918], [-1.149, 0.476, -1.259, -1.665], [-1.637, 1.266, -1.644, -1.882], [-0.979, 2.806, -1.087, -0.935], [-0.329, 0.452, -0.612, -1.324], [0.929, 0.484, 0.406, 0.107], [-1.099, -0.25, -1.109, -1.547], [-0.857, 1.008, -0.99, -0.407], [-0.123, 0.089, 0.232, 1.032], [1.081, -0.823, 1.125, 0.686], [-0.513, 1.71, -0.26, 0.053], [0.754, -0.315, 0.978, 1.055], [-0.425, -0.677, -0.01, -0.437], [-0.612, -0.702, -0.008, 0.583], [0.602, -0.774, 0.689, 0.617], [1.427, -1.105, 2.154, 1.532], [-1.651, -0.605, -1.156, -1.115], [0.834, -1.097, 1.269, 0.751], [-0.425, 0.911, -0.16, 0.727], [0.993, -0.331, 1.156, 0.929], [-0.259, -0.411, -0.483, -1.783], [-1.661, 0.492, -1.496, 1.222], [1.284, -1.097, 1.338, 1.834], [-0.225, -0.927, -0.39, -1.532], [0.248, -0.194, 0.394, 4.152], [1.551, -1.016, 1.155, 1.225], [0.385, 0.508, 0.55, -0.264], [1.091, -0.806, 1.061, 0.591], [1.514, -1.105, 1.251, 0.079], [0.653, -0.597, 0.83, 1.625], [1.457, -0.694, -0.614, -1.052], [1.29, -0.919, 1.229, 0.307], [0.32, -0.992, 0.442, -0.068]]


Function to calculate feature norms for an input query song. Uses same method outlined above in the feature set calculations.


```python
BANDWIDTH_PERCENTAGE = 0.90

# Function finds specific feature, and normalizes with the database mean and standard deviation. Normalized feature is placed in a list in the same feature order as the database feature list
def find_features(song):
  centroid_norm = np.round((calculate_mean_spectral_centroid(song) - mean_of_centroids) / std_of_centroids, 3)
  std_rms_norm = np.round((calculate_std_rms_amp(song) - mean_of_rms_amp) / std_of_rms_amp, 3)
  spectral_rolloff_norm = np.round((calculate_spectral_rolloff(song, BANDWIDTH_PERCENTAGE) - mean_of_rolloff) / std_of_rolloff, 3)
  spectral_flatness_norm = np.round((calculate_spectral_flatness(song) - mean_of_flatness) / std_of_flatness, 3)
  
  features = [centroid_norm, std_rms_norm, spectral_rolloff_norm, spectral_flatness_norm]
  return features
```

Create music recommender function as outlined


```python
def MusicRecommender(queryfilename, Nrecommendations):
  song = es.MonoLoader(filename=queryfilename, sampleRate=Fs)()
  song_features = find_features(song)
  avg_distance = []

  #calculate the average distances for all songs and query song
  for song_from_db in norm_database_features:
    avg_distance.append(np.round(np.mean((np.array(song_features) - np.array(song_from_db))**2), 3))
  print(f'Distances: {avg_distance}')

  #sort the average distances for all songs and query song and get lowest n recommendation indexes
  recommend_idx = np.argsort(avg_distance)[:Nrecommendations]
  print(f'\nSong Recommendations by Index (Including Itself If Applicable): {recommend_idx}\n')

  #add closest n recommendations to output list
  recommendations = []
  for i in recommend_idx:
    recommendations.append(song_names[i])

  return recommendations
```


```python
# Input song path and number of recommendations is loaded
song_dir = './music_test_dataFolder/classical/classical.00036.wav'

no_of_recommendations = 4

# Music recommendation function is utilized and recommendations are printed
music_recommendations = MusicRecommender(song_dir, no_of_recommendations)
print(music_recommendations)
```

    Distances: [1.543, 1.376, 0.196, 2.577, 0.896, 2.715, 2.444, 0.683, 2.473, 2.737, 3.507, 1.063, 2.302, 5.3, 0.538, 4.031, 3.525, 3.199, 4.374, 7.696, 3.731, 5.714, 1.059, 4.439, 4.211, 1.512, 6.575, 4.828, 5.938, 6.299, 2.499, 5.22, 6.553, 4.44, 5.266, 5.852, 4.525]
    
    Song Recommendations by Index (Including Itself If Applicable): [ 2 14  7  4]
    
    ['classical.wav', 'mingus1.wav', 'debussy.wav', 'copland2.wav']



```python
import IPython

# Song paths are placed in a continuous list
full_song_arr = list(files.values())[0] + list(files.values())[1] + list(files.values())[2]
```


```python
# Audio players for the origninal song, and the n recommendations are placed in order to hear similarities between the music

audio_path1 = song_dir
audio_path2 = full_song_arr[song_names.index(music_recommendations[0])]
audio_path3 = full_song_arr[song_names.index(music_recommendations[1])]
audio_path4 = full_song_arr[song_names.index(music_recommendations[2])]
audio_path5 = full_song_arr[song_names.index(music_recommendations[3])]

print('Original Song')
IPython.display.display(IPython.display.Audio(audio_path1))
print(f'\n\nFirst Recommendation: {music_recommendations[0]}')
IPython.display.display(IPython.display.Audio(audio_path2))
print(f'\n\nSecond Recommendation: {music_recommendations[1]}')
IPython.display.display(IPython.display.Audio(audio_path3))
print(f'\n\nThird Recommendation: {music_recommendations[2]}')
IPython.display.display(IPython.display.Audio(audio_path4))
print(f'\n\nFourth Recommendation: {music_recommendations[3]}')
IPython.display.display(IPython.display.Audio(audio_path5))

```

    Original Song



<audio controls="controls">

    
    
    First Recommendation: classical.wav


<audio controls="controls">

    
    Second Recommendation: mingus1.wav



<audio controls="controls">

    
    
    Third Recommendation: debussy.wav



<audio controls="controls">

    
    
    Fourth Recommendation: copland2.wav



<audio controls="controls">

# Project Synthesis

**1.1: Recommendation Algorithm Overview and Approach**

In this project we have created a music recommendation engine that takes a song as a query and outputs the n closest songs as recommendations for a user. In order to do this we computed the following list of features:
1.   Mean Spectral Centroid
2.   Standard Deviation of RMS Amplitude
3.   Mean Spectral Rolloff
4.   Mean Spectral Flatness
 
We computed this feature set prior to the recommendation algorithm as a hard coded data set on a list of songs given in the folder miniproject. Additionally we applied a feature vector normalization technique as follows:
$F_{Norm}=\frac{F_{Val} - F_{Mean}}{F_{std}}$
 
We applied this normalization for each of our selected feature vectors. Once these values are precomputed, our algorithm takes a new query song, and performs the same feature set calculations and normalization. This feature vector can be compared to the distance for every song in our database using the squared euclidean distance metric. Songs with lower squared euclidean distance should be categorized as more similar to our query song because this means that they share similar feature properties. As a result, our algorithm collects the n database songs with the lowest distance to our query song and outputs these to the user.

**1.2: Recommendation Algorithm Choices and Explanations**
 
In this particular algorithm we had a set of choices we made in computing these recommendations. Explanations to each of these choices is given as follows:
 
 
1.   *Feature set*
 
      Explanation: For our algorithm we incorporated the same feature set used in our first mini project. We decided to do this because in mini project 1 we found and demonstrated that different genres were accurately separated by this feature set. We feel that genre is the biggest factor of similarity in music and recommendation of similar songs, and we decided to structure our engine to follow this principle. Additionally, the feature set heavily leans towards an analysis of the frequency qualities of a song, which we feel are important in the enjoyment of a recommended song. If a user likes bass heavy music, they will likely enjoy music that is also bass heavy, and vice versa with other spectral qualities of a song.
 
 
2.   *Distance measure*
 
      Explanation: For our algorithm we made use of the squared euclidean distance measure. After researching good distance measures, we found that this particular distance measure is very commonly used in applications comparing distances, particularly due to its speed of calculation. When performing distance comparisons of a large feature set, omitting the square root component of euclidean distance can dramatically increase its speed. Our algorithm can be extended to support these larger feature sets, and as such, we decided that this was a good choice of distance measure.
 
 
3.   *Feature vector normalization*
 
      Explanation: As outlined in the project description, normalizing each feature vector by subtracting the mean and dividing by the standard deviation allows for each feature to have equal weight in deciding the prediction of the algorithm. We initially intended on weighing specific features based on our initial results, however the function's performance proved that the unweighted features we selected provided sufficiently good music recommendations.

**1.3: Data Used in Algorithm Development and Testing**

The audio files located in the minimusicspeech folder were used as the feature dataset, and also for the initial testing. The GTZAN dataset was then used for final testing purposes. The goal was to have the algorithm produce the most recommendations in the same genre as the query music file. Due to the GTZAN dataset having far more choices in genre and music files, similarities beyond music genre were observed by placing an audio file with a genre outside of the minimusicspeech dataset into the query. This is explored further in the algorithm performance section.

**1.4: Algorithm Performance**

The final algorithm performance was based on a variety of factors, including genre similarity, tempo, style of the music or subgenre, and overall feel of the song recommendations. The genre similarity performance was graded using the audio metadata provided in the datasets. The remaining factors were graded using human interpretation through multiple human listeners. The following table shows the music recommendation grading from three human listeners.

![image.png](data:image/png;base64,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)

Songs outside the minimusicspeech dataset genres were also used in the recommendation performance evaluation. Interestingly, all three "wildcard" queries (rock, country, and reggae) provided a higher satisfaction score than the blues genre queries. This could possibly be attributed to the similarities blues has with many other genres, most notably jazz.
 
Overall, aside from the blues genre, the algorithm provided decently good music recommendations. The music deemed most similar by the algorithm is represented by the first recommendation, and scored an average of 7.2, leaving the human interpreters generally satisfied with music similarity. 


**1.5: Project Contributions**

Both group members provided great effort toward the project, with each providing great ideas and implementations. Specifically, TV & SR divided the feature extractions equally with TV implementing spectral centroid, and standard deviation of RMS amplitude, and SR implementing spectral rolloff and spectral flatness. Each respective group member developed the normalization for their respective features. SR implemented the distance calculation and evaluation, and TV wrote a large portion of the project synthesis portion, such as details on our algorithm, group approach, and decision explanations. Both TV & SR assisted with algorithm testing, and performance evaluation.

**Example plot of query and recommended songs using Mini Project 1 features**


```python
query_song = es.MonoLoader(filename=audio_path1, sampleRate=Fs)()
query_song_features = find_features(query_song)
print(query_song_features)

```

    [-1.02, 2.734, -0.908, 0.706]


The normalized dataset features are plotted by list index in the scatter plot shown below. The query song features values are plotted by corresponding feature color and denoted as bold "x" marks. The indices for the first, second, and third song recommendations are highlighted. The three recommendations demonstrate a very small distance in STD of RMS amplitude, and medium distances for spectral centroid and spectral rolloff. Spectral flatness seems to provide the largest disparity. Based on this example, and the performance of this particular recommendation, it may be advantageous to place more weight on the STD of RMS amplitude feature, and less weight on the spectral flatness feature in future implementations.


```python
plt.figure(figsize=(16, 10))
plt.scatter(np.arange(0, len(norm_database_features), 1), [a[0] for a in norm_database_features], label='Spectral Centroid')
plt.scatter(np.arange(0, len(norm_database_features), 1), [a[1] for a in norm_database_features], label='STD of RMS Amp')
plt.scatter(np.arange(0, len(norm_database_features), 1), [a[2] for a in norm_database_features], label='Spectral Rolloff')
plt.scatter(np.arange(0, len(norm_database_features), 1), [a[3] for a in norm_database_features], label='Spectral Flatness')
plt.scatter(song_names.index(music_recommendations[0]), query_song_features[0], s=100, marker='X', facecolors='blue')
plt.scatter(song_names.index(music_recommendations[0]), query_song_features[1], s=100, marker='X', facecolors='orange')
plt.scatter(song_names.index(music_recommendations[0]), query_song_features[2], s=100, marker='X', facecolors='green')
plt.scatter(song_names.index(music_recommendations[0]), query_song_features[3], s=100, marker='X', facecolors='red')
plt.scatter(song_names.index(music_recommendations[1]), query_song_features[0], s=100, marker='X', facecolors='blue')
plt.scatter(song_names.index(music_recommendations[1]), query_song_features[1], s=100, marker='X', facecolors='orange')
plt.scatter(song_names.index(music_recommendations[1]), query_song_features[2], s=100, marker='X', facecolors='green')
plt.scatter(song_names.index(music_recommendations[1]), query_song_features[3], s=100, marker='X', facecolors='red')
plt.scatter(song_names.index(music_recommendations[2]), query_song_features[0], s=100, marker='X', facecolors='blue')
plt.scatter(song_names.index(music_recommendations[2]), query_song_features[1], s=100, marker='X', facecolors='orange')
plt.scatter(song_names.index(music_recommendations[2]), query_song_features[2], s=100, marker='X', facecolors='green')
plt.scatter(song_names.index(music_recommendations[2]), query_song_features[3], s=100, marker='X', facecolors='red')
plt.vlines(x=song_names.index(music_recommendations[0]), ymin=-1.5, ymax=3.3, color='r', alpha=0.1, linewidth=12, label='First Rec.')
plt.vlines(x=song_names.index(music_recommendations[1]), ymin=-1.5, ymax=3.3, color='b', alpha=0.1, linewidth=12, label='Second Rec.')
plt.vlines(x=song_names.index(music_recommendations[2]), ymin=-1.5, ymax=3.3, color='g', alpha=0.1, linewidth=12, label='Third Rec.')
plt.xlabel('Position in List')
plt.ylabel('Normalized Feature')
plt.title('Normalized Features vs. List Position')
plt.grid()
plt.legend()
plt.show()
```


    
![png](recommendation_engine_files/recommendation_engine_50_0.png)
    

