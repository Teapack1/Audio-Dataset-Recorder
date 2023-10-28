# Record 5s from mic and play it back
import sounddevice as sd
import soundfile as sf
import os
import argparse

SAMPLE_RATE = 16000

# Record audio
def record_audio(duration, samplerate=SAMPLE_RATE):
    audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=2, dtype='float32')
    sd.wait()
    return audio

def main(args):
    directory = args.directory
    filename = args.filename
    duration = args.duration
    num_recordings = args.num_recordings
    

    if not filename.endswith('.wav'):
        filename += '.wav'
        
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    for i in range(num_recordings):
        if num_recordings > 1:
            print(f"\nRecording {i+1} of {num_recordings}")
        print(f"Recording for {duration} seconds...")
        audio_data = record_audio(duration)
        
        base_name, ext = os.path.splitext(filename)
        current_filename = os.path.join(directory, f"{base_name}_{i+1}{ext}" if num_recordings > 1 else filename)

        sf.write(current_filename, audio_data, SAMPLE_RATE)
        print(f"Audio saved to {current_filename}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record audio from the command line.")
    parser.add_argument("-f", "--filename", default="sample", help="Base name for the file (default: 'sample').")
    parser.add_argument("-d", "--duration", type=float, default=1, help="Duration of recording in seconds (default: 1).")
    parser.add_argument("-dir", "--directory", default=".", help="Directory to save the recordings (default: current directory).")
    parser.add_argument("-n", "--num_recordings", type=int, default=1, help="Number of recordings to make (default: 1).")
    args = parser.parse_args()

    main(args)