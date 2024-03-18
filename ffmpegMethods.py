import subprocess
import numpy as np

def openSongAtFileLocation(filePath):
    """
    Read the location of the audio file from a text file.
    
    Args:
    - filePath: Path to the text file containing the location of the audio file
    
    Returns:
    - songPath: Path to the audio file
    """
    with open(filePath, "r") as file:
        songPath = file.read().strip()

    return songPath

def extractSamples(file_path):
    """
    Extract left and right channel samples from the audio file.
    
    Args:
    - file_path: Path to the audio file
    
    Returns:
    - l_samples: Left channel samples as a NumPy array
    - r_samples: Right channel samples as a NumPy array
    """
    try:
        # Execute FFmpeg command to extract raw PCM audio samples
        command = ['ffmpeg', '-i', file_path, '-vn', '-f', 'f32le', '-']
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        raw_audio_data = result.stdout
        
        # Convert raw audio data to NumPy array of float32 type
        audio_np = np.frombuffer(raw_audio_data, dtype='float32')
        
        # Separate left and right channel samples
        l_samples = audio_np[::2]  # Every other sample starting from index 0 (left channel)
        r_samples = audio_np[1::2]  # Every other sample starting from index 1 (right channel)
        
        return l_samples, r_samples
    except subprocess.CalledProcessError as e:
        # Handle errors during subprocess execution
        print("Error:", e.stderr)
        return None, None