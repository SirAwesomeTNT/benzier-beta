import subprocess
import numpy as np

def extractAndChunkSamples(filePath, amount):
    """
    Extract left and right channel samples from the audio file and return a chunk of samples.

    Args:
    - filePath (str): Path to the audio file.
    - amount (int): Number of samples to extract.

    Returns:
    - lChunk (numpy.ndarray): Left channel samples as a NumPy array.
    - rChunk (numpy.ndarray): Right channel samples as a NumPy array.
    """

    # Read the file path
    with open(filePath, "r") as file:
        songPath = file.read().strip()

    try:
        # Execute FFmpeg command to extract raw PCM audio samples
        command = ['ffmpeg', '-i', songPath, '-vn', '-f', 'f32le', '-']
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        raw_audio_data = result.stdout
        
        # Convert raw audio data to NumPy array of float32 type
        audio_np = np.frombuffer(raw_audio_data, dtype='float32')
        
        # Separate left and right channel samples
        l_samples = audio_np[::2]  # Every other sample starting from index 0 (left channel)
        r_samples = audio_np[1::2]  # Every other sample starting from index 1 (right channel)

        # Extract the specified amount of samples from left and right channels
        lChunk, rChunk = l_samples[:amount], r_samples[:amount]
        
        return lChunk, rChunk
    except subprocess.CalledProcessError as e:
        # Handle errors during subprocess execution
        print("Error:", e.stderr)
        return None, None
    
import numpy as np
import subprocess

class AudioGenerator:
    def __init__(self):
        pass

    def generate_audio_from_samples(self, samples, output_file):
        """
        Generate a FLAC audio file from samples.

        Args:
        - samples: A 2 by n array where each column represents a pair of left and right channel samples.
        - output_file: The path to save the generated audio file.
        """
        # Normalize and convert to int16
        normalized_samples = (samples * (2 ** 15 - 1)).astype(np.int16)

        # Convert to bytes
        audio_data = normalized_samples.tobytes()

        # FFmpeg command to convert int16 raw audio data to FLAC
        ffmpeg_command = [
            'ffmpeg',
            '-y', 
            '-f', 's16le',       # Input format: 16-bit signed little-endian
            '-ar', '44100',      # Sample rate: 44100 Hz
            '-ac', '2',          # Stereo audio
            '-i', '-',           # Read from stdin
            output_file          # Output file path
        ]

        # Run FFmpeg command and pass audio data
        with subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE) as ffmpeg_process:
            ffmpeg_process.communicate(input=audio_data)


# Example usage:
generator = AudioGenerator()
# Generate random sample values
random_samples = np.random.uniform(-1.0, 1.0, size=(2, 10000))
# Output file path
output_file = "random_samples_stereo.flac"
# Generate audio file
generator.generate_audio_from_samples(random_samples, output_file)
