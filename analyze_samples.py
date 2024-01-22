import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import argparse


def get_rms(audio):
    return np.sqrt(np.mean(np.square(audio)))


def analyze_microphones(base_dir):
    # Analyze the audio samples for each microphone to get average loudness and SNR if noise samples are present
    mic_results = {}
    audio_extensions = (".wav", ".aiff", ".flac", ".mp3")

    for dirname, _, filenames in os.walk(base_dir):
        print(filenames)
        mic_name = os.path.basename(dirname)
        if not filenames:
            continue  # Skip directories without files

        loudness = []
        noise_levels = []

        for filename in filenames:
            if filename.endswith(audio_extensions):
                file_path = os.path.join(dirname, filename)
                audio, sr = librosa.load(file_path, sr=None)
                rms_value = get_rms(audio)

                if filename.startswith("noise"):
                    noise_levels.append(rms_value)
                else:
                    loudness.append(rms_value)

        # Convert the average RMS values to dBFS for the signal
        if loudness:
            average_signal_rms = np.mean(loudness)
            mic_results[mic_name] = {
                "loudness": 20 * np.log10(average_signal_rms)
                if average_signal_rms > 0
                else -100
            }

        # Calculate the SNR if noise samples are present
        if noise_levels:
            average_noise_rms = np.mean(noise_levels)
            # Use RMS values directly to calculate SNR in dB
            snr = (
                20 * np.log10(average_signal_rms / average_noise_rms)
                if average_noise_rms > 0
                else float("inf")
            )
            mic_results[mic_name]["snr"] = snr

    return mic_results


def plot_results(mic_results):
    names = list(mic_results.keys())
    loudness_dbfs = [result["loudness"] for result in mic_results.values()]
    snr_values = [
        result.get("snr", -100) for result in mic_results.values()
    ]  # default to -100 dB for no SNR

    # Set the minimum loudness value to the lowest dBFS value found -3
    min_loudness_value = min(loudness_dbfs) - 3

    x = np.arange(len(names))
    width = 0.35

    fig, ax1 = plt.subplots()

    bars = ax1.bar(
        x,
        [value - min_loudness_value for value in loudness_dbfs],
        bottom=min_loudness_value,
        width=width,
        color="tab:blue",
        label="Loudness",
    )
    ax1.set_xlabel("Microphone")
    ax1.set_ylabel("Loudness (dBFS)", color="tab:blue")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_ylim(bottom=min_loudness_value)

    # Plot the SNR values on a second y-axis with proper scaling
    ax2 = ax1.twinx()
    (snr_line,) = ax2.plot(
        x,
        snr_values,
        color="tab:red",
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=6,
        label="SNR",
    )
    ax2.set_ylabel("SNR (dB)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_ylim(
        [min(snr_values) - 3, max(snr_values) + 3]
    )  # Adjust the SNR scale to fit the data

    # Add a legend to the plot
    plt.legend([bars, snr_line], ["Loudness", "SNR"], loc="upper left")

    # Add explanation text for SNR levels and Loudness
    explanation_text = (
        "Loudness Levels:\n"
        "-8 dBFS: Very Loud\n"
        "-16 dBFS: Good\n"
        "-20 dBFS: Standard\n\n"
        "SNR Levels:\n"
        "SNR < 20 dB: Poor\n"
        "SNR 20-40 dB: Acceptable\n"
        "SNR 40-60 dB: Very Good\n"
        "SNR > 60 dB: Excellent"
    )

    # Adjust figure size to accommodate the text
    fig.set_size_inches(10, 6)

    # Position the text on the graph
    plt.gcf().text(
        0.74,
        0.28,
        explanation_text,
        fontsize=9,
        va="center",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.4),
    )

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze audio from the command line.")
    parser.add_argument(
        "-dir",
        "--directory",
        default=".",
        help="Directory containing the microphone recordings (default: current directory).",
    )
    args = parser.parse_args()

    base_dir = args.directory
    mic_results = analyze_microphones(base_dir)
    plot_results(mic_results)

    # Determine the best microphone based on the highest average loudness
    best_mic = max(mic_results, key=lambda mic: mic_results[mic]["loudness"])
    print(f"The best microphone based on the average loudness is: {best_mic}")
