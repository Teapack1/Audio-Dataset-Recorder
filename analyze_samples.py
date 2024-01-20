import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import argparse

def get_rms(audio):
    return np.sqrt(np.mean(np.square(audio)))

def rms_to_dbfs(rms_value):
    #Convert RMS to dBFS
    return 20 * np.log10(rms_value) if rms_value > 0 else -100

def analyze_microphones(base_dir):
    #Analyze the audio samples for each microphone to get average loudness
    mic_results = {}
    audio_extensions = ('.wav', '.aiff', '.flac', '.mp3')

    for dirname, _, filenames in os.walk(base_dir):
        print(filenames)
        mic_name = os.path.basename(dirname)
        if not filenames:
            continue  # Skip directories without files
        elif not filenames[0].endswith(audio_extensions):
            continue  # Skip directories with non-wav files
        loudness = []
        
        for filename in filenames:
            file_path = os.path.join(dirname, filename)
            audio, sr = librosa.load(file_path, sr=None)
            rms_value = get_rms(audio)
            loudness.append(rms_to_dbfs(rms_value))
        
        # Store the average loudness in dBFS
        mic_results[mic_name] = {'loudness': np.mean(loudness)}
    
    return mic_results

def plot_results(mic_results):
    
    names = list(mic_results.keys())
    loudness_dbfs = [result['loudness'] for result in mic_results.values()]

    # Set the minimum loudness value to the lowest dBFS value found -1
    min_loudness_value = min(loudness_dbfs) - 1
    
    x = np.arange(len(names))
    width = 0.35
    
    fig, ax1 = plt.subplots()
    

    ax1.bar(x, [value - min_loudness_value for value in loudness_dbfs], 
            bottom=min_loudness_value, width=width, color='tab:blue')
    ax1.set_xlabel('Microphone')
    ax1.set_ylabel('Loudness (dBFS)', color='tab:blue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(bottom=min_loudness_value) 
    
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record audio from the command line.")
    parser.add_argument("-dir", "--directory", default=".", help="Directory to save the recordings (default: current directory).")
    args = parser.parse_args()
    
    
    base_dir = args.directory
    mic_results = analyze_microphones(base_dir)
    plot_results(mic_results)

    # Determine the best microphone based on the highest average loudness
    best_mic = max(mic_results, key=lambda mic: mic_results[mic]['loudness'])
    print(f"The best microphone based on the average loudness is: {best_mic}")


