import numpy as np
import librosa
import soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pydub import AudioSegment
import tempfile
import os

class AudioSegmenter:
    """Class to segment audio using logistic regression based on audio characteristics."""
    
    def __init__(self, frame_length=2048, hop_length=512):
        """Initialize with frame parameters for feature extraction."""
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.model = LogisticRegression(max_iter=1000)
        self.scaler = StandardScaler()
        
    def extract_features(self, audio, sr):
        """Extract audio features for segmentation."""
        # Extract various features
        # 1. Spectral centroid - brightness of sound
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, 
                                                   n_fft=self.frame_length, 
                                                   hop_length=self.hop_length).T
        
        # 2. Spectral contrast - difference between peaks and valleys
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr,
                                                   n_fft=self.frame_length,
                                                   hop_length=self.hop_length).T
        
        # 3. RMS energy - frame energy
        rms = librosa.feature.rms(y=audio, frame_length=self.frame_length, 
                                hop_length=self.hop_length).T
        
        # 4. Zero crossing rate - frequency of sign changes
        zcr = librosa.feature.zero_crossing_rate(audio, frame_length=self.frame_length, 
                                               hop_length=self.hop_length).T
        
        # 5. Onset strength - likelihood of onset
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr, 
                                               hop_length=self.hop_length)
        onset_env = onset_env.reshape(-1, 1)
        
        # Stack all features
        features = np.hstack([centroid, contrast, rms, zcr, onset_env[:len(centroid)]])
        return features
    
    def train_on_labeled_data(self, audio_files, labels_files):
        """Train the segmentation model on labeled audio files."""
        all_features = []
        all_labels = []
        
        for audio_file, label_file in zip(audio_files, labels_files):
            # Load audio
            audio, sr = librosa.load(audio_file, sr=None)
            
            # Extract features
            features = self.extract_features(audio, sr)
            
            # Load labels (assuming text file with 0/1 per frame)
            with open(label_file, 'r') as f:
                labels = [int(line.strip()) for line in f]
            
            # Make sure labels match feature length
            labels = labels[:len(features)]
            if len(labels) < len(features):
                labels = labels + [0] * (len(features) - len(labels))
            
            all_features.append(features)
            all_labels.extend(labels)
        
        # Combine all features and scale
        X = np.vstack(all_features)
        y = np.array(all_labels)
        
        # Fit scaler and transform
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        print("Model trained successfully!")
        
    def segment_audio(self, audio_file, threshold=0.5):
        """Segment audio file into regions using the trained model."""
        # Load audio
        audio, sr = librosa.load(audio_file, sr=None)
        
        # Extract features
        features = self.extract_features(audio, sr)
        
        # Check if model is trained
        try:
            # Scale features and use model prediction
            X_scaled = self.scaler.transform(features)
            probs = self.model.predict_proba(X_scaled)[:, 1]
        except:
            # If not trained, fall back to a simple onset detection approach
            print("Model not trained, falling back to onset detection for segmentation")
            onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=self.hop_length)
            onset_env = onset_env / np.max(onset_env)  # Normalize
            probs = onset_env  # Use onset strength as probability
        
        # Find segments using threshold
        segments = []
        current_start = 0
        
        for i, prob in enumerate(probs):
            frame_time = i * self.hop_length / sr
            if prob > threshold:
                if i > current_start:  # Avoid empty segments
                    segments.append((current_start * self.hop_length / sr, frame_time))
                current_start = i
        
        # Add the last segment
        if current_start * self.hop_length / sr < len(audio) / sr:
            segments.append((current_start * self.hop_length / sr, len(audio) / sr))
            
        return segments, audio, sr

    def extract_segments(self, segments, audio, sr):
        """Extract audio segments based on time boundaries."""
        audio_segments = []
        
        for start_time, end_time in segments:
            # Convert time to samples
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            
            # Extract segment
            segment = audio[start_sample:end_sample]
            audio_segments.append(segment)
            
        return audio_segments


class AudioTransformer:
    """Class to apply simple audio transformations."""
    
    def pitch_shift(self, audio, sr, n_steps):
        """Shift the pitch of audio by n_steps semitones."""
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    
    def time_stretch(self, audio, rate):
        """Time-stretch audio by rate. rate>1 speeds up, rate<1 slows down."""
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def apply_gain(self, audio, gain_db):
        """Apply gain in decibels to audio."""
        return audio * (10 ** (gain_db / 20))
    
    def apply_low_pass(self, audio, sr, cutoff_freq):
        """Apply a simple low-pass filter."""
        from scipy.signal import butter, filtfilt
        
        # Design filter
        nyquist = 0.5 * sr
        normal_cutoff = cutoff_freq / nyquist
        b, a = butter(4, normal_cutoff, btype='low', analog=False)
        
        # Apply filter
        filtered_audio = filtfilt(b, a, audio)
        return filtered_audio
    
    def apply_high_pass(self, audio, sr, cutoff_freq):
        """Apply a simple high-pass filter."""
        from scipy.signal import butter, filtfilt
        
        # Design filter
        nyquist = 0.5 * sr
        normal_cutoff = cutoff_freq / nyquist
        b, a = butter(4, normal_cutoff, btype='high', analog=False)
        
        # Apply filter
        filtered_audio = filtfilt(b, a, audio)
        return filtered_audio
    
    def apply_reverb(self, audio, sr, reverb_time=1.0):
        """Apply a simple reverb effect."""
        # Create a simple impulse response for reverb
        ir_len = int(reverb_time * sr)
        impulse_response = np.exp(-np.linspace(0, 10, ir_len))
        impulse_response = impulse_response / np.sum(impulse_response)  # Normalize
        
        # Convolve audio with impulse response
        from scipy.signal import convolve
        reverb_audio = convolve(audio, impulse_response, mode='full')
        
        # Trim to original length
        reverb_audio = reverb_audio[:len(audio)]
        
        return reverb_audio


class MusicCreator:
    """Class to create music from segmented and transformed audio."""
    
    def __init__(self, output_sr=44100):
        """Initialize with output sample rate."""
        self.output_sr = output_sr
        self.segmenter = AudioSegmenter()
        self.transformer = AudioTransformer()
        
    def _convert_mp3_to_wav(self, mp3_path):
        path = os.path.dirname(mp3_path)
        ffmpeg_exe = os.path.join(path, 'ffmpeg.exe')
        ffprobe_exe = os.path.join(path, 'ffprobe.exe')
        # Configure pydub
        AudioSegment.converter = ffmpeg_exe
        AudioSegment.ffmpeg = ffmpeg_exe
        AudioSegment.ffprobe = ffprobe_exe
        """Convert MP3 file to WAV for processing."""
        audio = AudioSegment.from_mp3(mp3_path)
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        audio.export(temp_wav.name, format="wav")
        return temp_wav.name
        
    def create_rhythm_from_segments(self, segments, sr, pattern, output_file,
                                   transformations=None):
        """Create a rhythmic pattern from audio segments.
        
        Args:
            segments: List of audio segments
            sr: Sample rate
            pattern: List of segment indices to arrange in sequence
            output_file: Path to save the output
            transformations: List of dictionaries with transformation parameters
        """
        if transformations is None:
            transformations = [{}] * len(pattern)
        
        # Calculate total duration based on pattern
        total_samples = 0
        segment_lengths = [len(segment) for segment in segments]
        
        for idx in pattern:
            if idx < len(segments):
                total_samples += segment_lengths[idx]
        
        # Create output array
        output = np.zeros(total_samples)
        current_position = 0
        
        # Fill output array according to pattern
        for i, idx in enumerate(pattern):
            if idx < len(segments):
                segment = segments[idx].copy()
                
                # Apply transformations if specified
                if i < len(transformations) and transformations[i]:
                    if 'pitch' in transformations[i]:
                        segment = self.transformer.pitch_shift(
                            segment, sr, transformations[i]['pitch'])
                    
                    if 'stretch' in transformations[i]:
                        segment = self.transformer.time_stretch(
                            segment, transformations[i]['stretch'])
                        
                    if 'gain' in transformations[i]:
                        segment = self.transformer.apply_gain(
                            segment, transformations[i]['gain'])
                    
                    if 'low_pass' in transformations[i]:
                        segment = self.transformer.apply_low_pass(
                            segment, sr, transformations[i]['low_pass'])
                    
                    if 'high_pass' in transformations[i]:
                        segment = self.transformer.apply_high_pass(
                            segment, sr, transformations[i]['high_pass'])
                    
                    if 'reverb' in transformations[i]:
                        segment = self.transformer.apply_reverb(
                            segment, sr, transformations[i]['reverb'])
                
                # Add to output
                end_pos = current_position + len(segment)
                if end_pos > len(output):
                    # Adjust segment length if it would exceed output length
                    segment = segment[:len(output) - current_position]
                    end_pos = len(output)
                
                output[current_position:end_pos] += segment
                current_position = end_pos
        
        # Normalize to prevent clipping
        if np.max(np.abs(output)) > 1.0:
            output = output / np.max(np.abs(output))
        
        # Save output
        sf.write(output_file, output, sr)
        print(f"Created rhythmic pattern and saved to {output_file}")
        
    def process_file_to_music(self, input_file, output_file, 
                              segment_threshold=0.5,
                              pattern=None, transformations=None):
        """Process a single audio file into music."""
        # Check if input is MP3 and convert if needed
        if input_file.lower().endswith('.mp3'):
            wav_file = self._convert_mp3_to_wav(input_file)
            input_file = wav_file
            
        # Train a simple model for segmentation (in real use, you'd train on labeled data)
        # For this example, we'll use onset detection as a proxy for segmentation
        segments, audio, sr = self.segmenter.segment_audio(input_file, threshold=segment_threshold)
        
        # Extract the actual audio segments
        audio_segments = self.segmenter.extract_segments(segments, audio, sr)
        
        # Create a default pattern if none provided
        if pattern is None:
            # Simple repeating pattern using available segments
            num_segments = len(audio_segments)
            pattern = list(range(min(16, num_segments)))
        
        # Create default transformations if none provided
        if transformations is None:
            transformations = []
            for i in range(len(pattern)):
                # Apply some varied transformations
                transform = {}
                if i % 3 == 0:  # Every third segment
                    transform['pitch'] = 2  # Up a whole step
                if i % 4 == 0:  # Every fourth segment
                    transform['reverb'] = 0.8
                if i % 5 == 0:  # Every fifth segment
                    transform['gain'] = 3  # Boost volume
                transformations.append(transform)
        
        # Create the music
        self.create_rhythm_from_segments(audio_segments, sr, pattern, output_file, transformations)
        return segments, audio_segments


# Example usage
if __name__ == "__main__":
    # Example parameters
    input_file = "796874__kevp888__009a_091104_0019_maur_sunny_morning_in_mauritius.mp3"  # Replace with path to your audio file
    output_file = "transformed_rhythm.wav"
    
    # Create the music creator
    creator = MusicCreator()
    
    # Process the file
    segments, audio_segments = creator.process_file_to_music(
        input_file, output_file, 
        segment_threshold=0.3,  # Lower threshold for more segments
        pattern=[0, 1, 2, 0, 3, 1, 2, 3, 0, 2, 1, 3, 2, 0, 1, 3],  # Custom pattern
        transformations=[
            {'pitch': 0, 'gain': 3},           # Segment 0: Boost volume
            {'pitch': -2, 'low_pass': 1000},   # Segment 1: Lower pitch, apply low-pass
            {'pitch': 4, 'reverb': 0.5},       # Segment 2: Higher pitch, add reverb
            {'stretch': 0.8, 'high_pass': 500} # Segment 3: Speed up, apply high-pass
        ]
    )
    
    print(f"Created music track with {len(segments)} segments")
    print(f"Segment time boundaries:")
    for i, (start, end) in enumerate(segments):
        print(f"  Segment {i}: {start:.2f}s - {end:.2f}s (duration: {end-start:.2f}s)")