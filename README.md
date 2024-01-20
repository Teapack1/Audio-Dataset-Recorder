# Audio Dataset Recording & Augmentation Tools

- <b>dr.py</b>
Record complete dataset with metadata, directories, augmentation, etc.
- <b>drLite.py</b>
Record quickly single or multiple audio samples.
- <b>analyze_samples.py</b>
Analyze and compare sample directories based on average RMS intensity.

Dataset recorder, with its primary script `dr.py`, helps to swiftly produce complete audio datasets, especially for machine learning applications. It's complemented by `drLite.py`, a more lightweight script for quick and easy recording of single or multiple audio samples.

Both scripts are user-friendly, saving recordings in .wav format directly to the relative root directory, ensuring easy access and management.


## Dataset Recorder
`dr.py`

This Python-based utility enables efficient audio recording with options for categorization, augmentation, and metadata generation. It's tailored for collecting data for various audio-related machine learning tasks.


### Features

- **Multiple Recording Modes**: Two recording variants - A for a common folder for all samples, and B for class-specific folders, enhancing data organization.
- **Audio Augmentation**: Inbuilt audio augmentation features: mixing with ambient noises, pitch shifting, time stretching, syntetic noise addition and decibel adjustment. (ambient noises provided by the user).
- **Metadata Generation**: Automates generation of metadata CSV files post-recording, with paths and labels for every audio sample.
- **Automatic Recording**: Samples are recorded automatically after the sctipt runs, no need to press any buttons.
- **Catches up**: In partly recorded dataset finds the last files and do not rewrite any existing files. Can also perform augmentation or metadata only, when placed next to the audio_samples folder.
- **Other Utilities**: Immediate audio playback, Peak-normalization, Trim-padding, Device selection.

### Requirements

- Python 3.10.*
- `sounddevice`
- `librosa`
- `pydub`
- `soundfile`
- `pandas`
- `numpy`

To install the required packages, run:

`
pip install sounddevice librosa pydub soundfile pandas numpy
`

### Usage

1. Record using variant B (default - samples segregated into class-specific folders) 20 samples of each class, and classes are: "one", "two", "three":
`
python dr.py --num_samples 20 --classes one two three
`

2. Record using variant A (all samples in a single folder), no listening mode - samples are recorded after set duration, does not smart record, samples are 2sec long :
`
python dr.py --method A --no_listening_mode --num_samples 10 --duration 2 --classes yes no
`

3. Record variant B with classes "yes", "no" and duration of 1 sec. Gathers 10 samples of each class, produces metadata, trimpads silent ends in the sample, playbacks sample after every recording, augments every record with 25 samples, amplitude treshold to triger new recording is lowered to 0.1, sample rate increased to 48000.
`
python dr.py --classes yes no --duration 1 --num_samples 10 --metadata --trim_pad --playback --augment --num_augmented 25 --treshold 0.1 --sample_rate 48000
`

### Command Line Arguments

- `--classes`: Specify classes for the recordings, default "audio". (eg.:`--classes one two three`) -> mandatory
- `--method`: Recording method. `A` saves all samples in one folder while `B` (default) saves samples to separate folders for each class.
- `--augment`: flag, include to indicate if augmentation is required otherwise samples are not augmented.
- `--num_augmented`: Number of augmented samples for every original sample.
- `--num_samples`: Number of samples in every class.
- `--duration`: Duration of one sample in seconds.
- `--treshold:` Define amplitude threshold to start recording.
- `--metadata`: flag, include in command to produce metadata after recording or produce metadata of already recorded samples.
- `--normalize`: flag, scales the entire audio signal such that the loudest peak in the audio will become 1.0.
- `--trim_pad`: flag, trim silence parts and pad it back with zeros.
- `--playback`: flag, indicate instant playback or specify local file for playback.
- `--device`: flag, Choose a specific device for recording. Lists available devices, do not include for auto-selection.
- `--sample_rate`: Sampling Rate (16000 default).


<br>
<br>


## Dataset Recorder lite
`drLite.py`

This is a simple Python script to record audio samples from your microphone directly from the command line. It records one or several samples in one go. The recordings are saved as `.wav` files.


### Requirements

- Python
- `sounddevice`
- `soundfile`
- `numpy`

To install the required packages, run:

`
pip install sounddevice soundfile numpy
`

### Features

- Use command line arguments for customization.
- Record audio with a specified duration.
- Use immediate playback.
- Save multiple recordings with incremental filenames.
- Automatic recording
   
### Usage

1. Record one 1-second audio and save it as sample.wav in the current directory:
`
python drLite.py
`
2. Record three 10-second audios and save them in a folder named "recordings":
`
python drLite.py -f recording -d 10 -dir recordings -n 3
`

### Command Line Arguments

- `-f` or `--filename`: Base name for the file (default is 'sample').
- `-d` or `--duration`: Duration of recording in seconds (default is 1 second).
- `-dir` or `--directory`: Directory to save the recordings (default is the current directory).
- `-n` or `--num_recordings`: Number of recordings to make (default is 1).
- `-t` or `--treshold:` Define the amplitude threshold for starting the recording.
- `-p` or `--playback:` Flag to indicate playback or specify a file for playback.
- `-sr` or `--sample_rate:`Sampling Rate (16000 default).
- `--device`: Choose a specific device for recording. Otherwise default mic is selected. Use --list_devices to view available devices.
- `--list_devices`: List available recording devices and exit.
- `--no_listening_mode:` Listens for incomming audio and records when there is input. (default is on).
