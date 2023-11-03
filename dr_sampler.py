import sounddevice as sd
import argparse
import os
import time
import librosa
import random
import numpy as np
from pydub import AudioSegment
import soundfile as sf
import pandas as pd
import keyboard

class AudioRecorder:
    def __init__(
        self,
        sample_rate=16000,
        duration=1,
        classes=None,
        samples_dir="audio_samples",
        aug_samples_dir="augmented_audio_samples",
        sample_count=20,
        device_index=1,
        channels=1,
    ):
        self.SAMPLE_RATE = sample_rate
        self.DURATION = duration
        self.CLASSES = classes
        self.SAMPLES_DIR = samples_dir
        self.SAMPLE_COUNT = sample_count
        self.DEVICE_INDEX = device_index
        self.CHANNELS = channels
        self.AUG_SAMPLES_DIR = aug_samples_dir

        if not os.path.exists(self.SAMPLES_DIR):
            os.mkdir(self.SAMPLES_DIR)

    def apply_audio_editing(self, trim_pad_flag, normalize_flag):
        # For every directory, subdir, and file found in SAMPLES_DIR
        for dirpath, dirnames, filenames in os.walk(self.SAMPLES_DIR):
            for f in filenames:
                if f.endswith(".wav"):  # Ensure it's an audio file
                    audio_file = os.path.join(dirpath, f)

                    # Determine if we are using variant A or B
                    if os.path.dirname(audio_file) == self.SAMPLES_DIR:  # Variant A
                        edit_class_dir = (
                            self.SAMPLES_DIR
                        )  # Save directly to SAMPLES_DIR
                    else:  # Variant B
                        class_name = os.path.basename(os.path.dirname(audio_file))
                        edit_class_dir = os.path.join(
                            self.SAMPLES_DIR, class_name
                        )  # Save to the subdirectory

                    y, sr = librosa.load(audio_file, sr=None)

                    if trim_pad_flag:
                        y = self.trim_pad(y)
                        print(f"Trimmed {audio_file}")

                    if normalize_flag:
                        y = self.normalize(y)
                        print(f"Normalized {audio_file}")

                    sf.write(
                        audio_file, y, sr
                    )  # Overwrite the original file with edited data
    

    def trim_pad(self, audio):
        """Trim silence and pad audio to ensure consistent length."""
        # Trim silence
        trimmed, _ = librosa.effects.trim(audio, top_db=15)
        # Pad to ensure consistent length
        desired_length = int(self.SAMPLE_RATE * self.DURATION)
        if len(trimmed) < desired_length:
            padding = desired_length - len(trimmed)
            padded_audio = np.pad(trimmed, (0, padding), "constant")
        else:
            padded_audio = trimmed[:desired_length]
        return padded_audio

    def play_sample(self, filepath=None, data=None, samplerate=None):
        """Plays the audio sample at the specified file path or the provided audio data."""
        if filepath:
            # Load the audio file
            data, samplerate = sf.read(filepath)
        # Play the audio data
        sd.play(data, samplerate)
        # Use sd.wait() to block execution until audio is finished playing
        sd.wait()

    def normalize(self, audio):
        audio_max = np.max(np.abs(audio))
        if audio_max > 0:
            scaling_factor = 1.0 / audio_max
            audio = audio * scaling_factor
        return audio

    def augment_samples(self, num_augmented):
        class_counts = {}

        os.makedirs(self.AUG_SAMPLES_DIR, exist_ok=True)

        for i, (dirpath, dirnames, filenames) in enumerate(os.walk(self.SAMPLES_DIR)):
            for f in filenames:
                file_path = os.path.join(dirpath, f)
                self._augment_file(file_path, num_augmented, class_counts)

    def _augment_file(self, audio_file, num_augmented, class_counts):
        print(f"Augmenting {audio_file}")

        y, sr = librosa.load(audio_file, sr=None)

        # Determine if we are using variant A or B
        if os.path.dirname(audio_file) == self.SAMPLES_DIR:  # Variant A
            class_name = os.path.basename(audio_file).split("_")[0]
            original_prefix = class_name
            augmented_class_dir = (
                self.AUG_SAMPLES_DIR
            )  # Save directly to AUG_SAMPLES_DIR
        else:  # Variant B
            class_name = os.path.basename(os.path.dirname(audio_file))
            original_prefix = os.path.basename(audio_file).split("_")[0]
            augmented_class_dir = os.path.join(
                self.AUG_SAMPLES_DIR, class_name
            )  # Save to subdirectory
            os.makedirs(
                augmented_class_dir, exist_ok=True
            )  # Create subdirectory if it doesn't exist

        if class_name not in class_counts:
            class_counts[class_name] = 0

        for i in range(num_augmented):
            method = random.choice(["pitch", "stretch", "noise", "db"])

            if method == "pitch":
                steps = random.randint(-1, 1)
                augmented = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)

            elif method == "stretch":
                rate = random.uniform(0.9, 1.1)
                augmented = librosa.effects.time_stretch(y, rate=rate)

            elif method == "noise":
                noise = np.random.normal(0, 0.01, len(y))
                augmented = y + noise

            elif method == "db":
                audio_segment = AudioSegment.from_wav(audio_file)
                db_change = random.randint(-10, 10)
                augmented_segment = audio_segment + db_change
                augmented = np.array(augmented_segment.get_array_of_samples())

            class_counts[class_name] += 1

            new_file = os.path.join(
                augmented_class_dir,
                f"{original_prefix}_aug_{class_counts[class_name]}.wav",
            )
            sf.write(new_file, augmented, self.SAMPLE_RATE)

    def get_highest_index(self, cls, variant):
        """
        Get the highest index of the already recorded samples for a given class.
        """
        highest_index = -1

        if variant == "A":
            path = self.SAMPLES_DIR
            files = [
                f
                for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f)) and f.startswith(cls)
            ]
        elif variant == "B":
            path = os.path.join(self.SAMPLES_DIR, cls)
            files = [
                f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))
            ]

        for f in files:
            try:
                index = int(
                    f.split("_")[-1].split(".")[0]
                )  # Extracting index from the filename
                highest_index = max(highest_index, index)
            except ValueError:
                continue

        return highest_index

    def record_audio_variant_A(self, playback=False):
        for cls in self.CLASSES:
            highest_index = self.get_highest_index(cls, "A")
            input(f"Press Enter to start recording for class '{cls}' ...")
            #time.sleep(0.5)

            for sample in range(highest_index + 1, highest_index + 1 + self.SAMPLE_COUNT):
                print(f"Recording sample '{cls}': {sample-highest_index} / {self.SAMPLE_COUNT}")
                record = sd.rec(
                    int(self.DURATION * self.SAMPLE_RATE),
                    samplerate=self.SAMPLE_RATE,
                    channels=self.CHANNELS,
                    device=self.DEVICE_INDEX,
                    dtype="int16",
                )
                sd.wait()

                if record.shape[1] > 1:
                    record = np.mean(record, axis=1)

                filename = os.path.join(self.SAMPLES_DIR, f"{cls}_{sample}.wav")
                sf.write(filename, record, self.SAMPLE_RATE)
                print("Saved at: ", filename)

                if playback:
                    self.play_sample(
                        data=record, samplerate=self.SAMPLE_RATE
                    )  

        print("Finished recording.")


    def record_audio_variant_B(self, playback=False):
        for cls in self.CLASSES:
            highest_index = self.get_highest_index(cls, "B")  # Fixed to use "B"
            input(f"Press Enter to start recording for class '{cls}' ...")

            dir = os.path.join(self.SAMPLES_DIR, cls)
            if not os.path.exists(dir):
                os.mkdir(dir)
            #time.sleep(0.5)

            for sample in range(highest_index + 1, highest_index + self.SAMPLE_COUNT + 1):
                print(f"Recording sample '{cls}': {sample-highest_index} / {self.SAMPLE_COUNT}")
                record = sd.rec(
                    int(self.DURATION * self.SAMPLE_RATE),
                    samplerate=self.SAMPLE_RATE,
                    channels=self.CHANNELS,
                    device=self.DEVICE_INDEX,
                    dtype="int16",
                )
                sd.wait()

                if record.shape[1] > 1:
                    record = np.mean(record, axis=1)

                filename = os.path.join(dir, f"{cls}_{sample}.wav")
                sf.write(filename, record, self.SAMPLE_RATE)
                print("Saved at: ", filename)

                if playback:
                    self.play_sample(data=record, samplerate=self.SAMPLE_RATE) 

        print("Finished recording.")


    def produce_metadata(self):
        metadata = {"filepath": [], "label": [], "class_num": []}

        # 1) Check classes from the argument
        if self.CLASSES:
            classes = self.CLASSES
        # 2) Check AUG_SAMPLES_DIR
        elif os.path.exists(self.AUG_SAMPLES_DIR):
            direct_files = [
                f
                for f in os.listdir(self.AUG_SAMPLES_DIR)
                if os.path.isfile(os.path.join(self.AUG_SAMPLES_DIR, f))
            ]
            classes_from_files = set([f.split("_")[0] for f in direct_files])
            classes_from_dirs = set(
                [
                    d
                    for d in os.listdir(self.AUG_SAMPLES_DIR)
                    if os.path.isdir(os.path.join(self.AUG_SAMPLES_DIR, d))
                ]
            )
            classes = list(classes_from_files.union(classes_from_dirs))
        # 3) Check SAMPLES_DIR
        elif os.path.exists(self.SAMPLES_DIR):
            direct_files = [
                f
                for f in os.listdir(self.SAMPLES_DIR)
                if os.path.isfile(os.path.join(self.SAMPLES_DIR, f))
            ]
            classes_from_files = set([f.split("_")[0] for f in direct_files])
            classes_from_dirs = set(
                [
                    d
                    for d in os.listdir(self.SAMPLES_DIR)
                    if os.path.isdir(os.path.join(self.SAMPLES_DIR, d))
                ]
            )
            classes = list(classes_from_files.union(classes_from_dirs))
        # 4) Default classes
        else:
            classes = ["yes", "no", "hi"]

        # Mapping from class names to numerical values
        class_to_num = {cls: idx for idx, cls in enumerate(classes)}
        print("Class to num: ", class_to_num)

        target_dir = (
            self.AUG_SAMPLES_DIR
            if os.path.exists(self.AUG_SAMPLES_DIR)
            else self.SAMPLES_DIR
        )

        for i, (dirpath, dirnames, filenames) in enumerate(os.walk(target_dir)):
            dir_label = os.path.basename(dirpath) if dirpath != target_dir else None

            for f in filenames:
                label = f.split("_")[0] if dir_label is None else dir_label
                file_path = os.path.join(dirpath, f)
                metadata["filepath"].append(file_path)
                metadata["label"].append(label)
                metadata["class_num"].append(
                    class_to_num.get(label, -1)
                )  # -1 as default if label is not found

        df = pd.DataFrame(metadata)

        df.to_csv("metadata.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Record audio samples for different classes."
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["A", "B"],
        help="Recording variant A save all samples in one folder. B saves samples to separate folders for each class.",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Flag to indicate if augmentation is required",
    )
    parser.add_argument(
        "--num_augmented",
        type=int,
        default=25,
        help="Number of augmented samples from every original sample",
    )
    parser.add_argument(
        "--classes", nargs="+", default=None, help="Specify classes for the recordings."
    )
    parser.add_argument(
        "--num_samples", type=int, default=20, help="Number of samples in every class."
    )
    parser.add_argument(
        "--duration", type=int, default=1, help="Duration of one sample in seconds."
    )
    parser.add_argument(
        "--device_index", type=int, default=1, help="Specify microphone device index."
    )
    parser.add_argument(
        "--check_devices",
        action="store_true",
        help="Flag to check available input devices and exit.",
    )
    parser.add_argument(
        "--metadata",
        action="store_true",
        help="Produce metadata after recording (default is on).",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize audio samples to bring the loudest peak to a target level.",
    )
    parser.add_argument(
        "--trim_pad",
        action="store_true",
        help="Trim silence parts and pad it back with zeros to ensure consistent length.",
    )
    parser.add_argument(
        "--playback",
        nargs="?",
        const=True,
        default=False,
        help="Flag to indicate playback or specify a file for playback.",
    )

    args = parser.parse_args()

    if args.check_devices:
        print(sd.query_devices())
        exit()

    recorder = AudioRecorder(
        classes=args.classes,
        sample_count=args.num_samples,
        duration=args.duration,
        device_index=args.device_index,
    )

    if args.method == "A":
        recorder.record_audio_variant_A(playback=args.playback)
    elif args.method == "B":
        recorder.record_audio_variant_B(playback=args.playback)
    elif args.playback not in (True, False):
        recorder.play_sample(filepath=args.playback)
        exit()

    if args.trim_pad or args.normalize:
        recorder.apply_audio_editing(args.trim_pad, args.normalize)

    if args.augment:
        recorder.augment_samples(args.num_augmented)

    if args.metadata:
        recorder.produce_metadata()
