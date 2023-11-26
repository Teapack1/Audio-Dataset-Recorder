import sounddevice as sd
import soundfile as sf
import os
import argparse
import numpy as np

SAMPLE_RATE = 16000


def find_mic_index(sounddevice):
    mic_index = None
    devices = sounddevice.query_devices()

    for i, dev in enumerate(devices):
        print('Device {}: {}'.format(i, dev['name']))

        if dev['max_input_channels'] > 0:
            print("------------------------------------")
            print('Found an input: device {} - {}'.format(i, dev['name']))
            print(dev)
            mic_index = i
            return mic_index

    if mic_index is None:
        print('Using default input device.')
        return sd.default.device[0] 

    return mic_index

# Record audio
def record_regular(record_seconds=1, channels=1, rate=16000, chunk_size=128, device=0):
    recording = sd.rec(
        int(record_seconds * rate),
        samplerate=rate,
        channels=channels,
        device=device,
        dtype="int16",
    )
    sd.wait()
    return recording


def record_auto( threshold=0.6, record_seconds=1, channels=1, rate=16000, chunk_size=128, device=0):
    def get_rms(block):
        return np.sqrt(np.mean(np.square(block)))

    with sd.InputStream(channels=channels, samplerate=rate, blocksize=chunk_size, dtype='float32', device=device) as stream:
        while True:
            data, _ = stream.read(chunk_size)
            snd_block = data[:, 0] if channels > 1 else data
            amplitude = get_rms(snd_block)

            if amplitude > threshold:
                print("* Recording with amplitude:", amplitude)
                frames = [data]  # Start with the current chunk

                for _ in range(1, int(rate / chunk_size * record_seconds)):
                    data, _ = stream.read(chunk_size)
                    frames.append(data)
                return np.concatenate(frames, axis=0)
                
def play_sample(filepath=None, data=None, samplerate=None):
    """Plays the audio sample at the specified file path or the provided audio data."""
    if filepath:
        # Load the audio file
        data, samplerate = sf.read(filepath)
    # Play the audio data
    sd.play(data, samplerate)
    # Use sd.wait() to block execution until audio is finished playing
    sd.wait()


def list_devices():
    print(sd.query_devices())

def main(args):
    directory = args.directory
    filename = args.filename
    duration = args.duration
    num_recordings = args.num_recordings
    no_listening_mode=args.no_listening_mode
    if args.device:
        device = args.device
    else:
        device = find_mic_index(sd)
    chunk_size = 128
    sample_rate = args.sample_rate
    treshold = args.treshold

    if args.list_devices:
        list_devices()
        return

    if not filename.endswith('.wav'):
        filename += '.wav'
        
    if not os.path.exists(directory):
        os.makedirs(directory)
    print(f"Recording initialized, press Ctrl+C to stop recording")
    for i in range(num_recordings):
        if num_recordings > 1:
            print(f"Recording {i+1} of {num_recordings}")
        if no_listening_mode:
            record = record_regular(record_seconds = duration, channels=1, rate=sample_rate, chunk_size=chunk_size, device=device)
        else:
            record = record_auto(threshold = treshold, record_seconds = duration, channels=1, rate=sample_rate, chunk_size=chunk_size, device=device)
            
        base_name, ext = os.path.splitext(filename)
        current_filename = os.path.join(directory, f"{base_name}_{i+1}{ext}" if num_recordings > 1 else filename)
        if args.playback:
            play_sample(
            data=record, samplerate=SAMPLE_RATE
            )  
        sf.write(current_filename, record, SAMPLE_RATE)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record audio from the command line.")
    parser.add_argument("-f", "--filename", default="sample", help="Base name for the file (default: 'sample').")
    parser.add_argument("-d", "--duration", type=float, default=1, help="Duration of recording in seconds (default: 1).")
    parser.add_argument("-t", "--treshold", type=float, default=0.1, help="Treshold to start recording (default: 0.2).")
    parser.add_argument("-dir", "--directory", default=".", help="Directory to save the recordings (default: current directory).")
    parser.add_argument("-n", "--num_recordings", type=int, default=1, help="Number of recordings to make (default: 1).")
    parser.add_argument("-dev", "--device", type=int, default=None, help="Choose a specific device for recording. Use --list_devices to view available devices.")
    parser.add_argument("--list_devices", action="store_true", help="List available recording devices and exit.")
    parser.add_argument(
    "--no_listening_mode",
    action="store_true",
    help="Listens for incomming audio and records when there is input. (default is on).",
    )
    parser.add_argument(
    "--playback",
    action="store_true",
    help="Flag to indicate playback or specify a file for playback.",
    )
    parser.add_argument(
        "--sample_rate", type=int, default=16000, help="Sampling Rate (16000 default)."
    )


    args = parser.parse_args()

    main(args)
