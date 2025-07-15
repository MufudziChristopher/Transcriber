#!/usr/bin/env python3
import os
import time
import logging
import numpy as np
from pydub import AudioSegment
import tempfile
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
import soundfile as sf
import pyloudnorm
import noisereduce as nr
import torch
import warnings
import whisper

class RobustSermonTranscriber:
    """Optimized for reliability with African accents"""
    
    def __init__(self, model_size: str = "base.en", verbose: bool = True):
        self.verbose = verbose
        self._setup_logging()
        self._load_model(model_size)
        self.temp_dir = tempfile.mkdtemp(prefix="transcriber_")
        
    def _setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO if self.verbose else logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        self.log = logging.getLogger(__name__)
        
    def _load_model(self, model_size: str):
        """Safe model loading"""
        warnings.filterwarnings(
            "ignore",
            message="FP16 is not supported on CPU",
            category=UserWarning
        )
        
        self.log.info(f"Loading {model_size} model...")
        try:
            self.model = whisper.load_model(model_size)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(self.device)
            self.log.info(f"Model loaded on {self.device}")
        except Exception as e:
            self.log.error(f"Model load failed: {e}")
            raise

    def _safe_preprocess(self, audio_path: str) -> Optional[str]:
        """Fully validated audio processing"""
        try:
            # Load with validation
            audio = AudioSegment.from_file(audio_path)
            if len(audio) < 500:  # At least 500ms
                raise ValueError("Audio too short")
                
            # Convert to numpy with validation
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            if len(samples) < 8000:  # At least 0.5s at 16kHz
                raise ValueError("Insufficient audio samples")
                
            # Noise reduction
            reduced = nr.reduce_noise(
                y=samples,
                sr=audio.frame_rate,
                stationary=True,
                prop_decrease=0.8
            )
            
            # Skip normalization if audio is very quiet
            if np.max(np.abs(reduced)) > 0.02:
                meter = pyloudnorm.Meter(audio.frame_rate)
                loudness = meter.integrated_loudness(reduced)
                normalized = pyloudnorm.normalize.loudness(
                    reduced,
                    loudness,
                    -20.0,
                    True
                )
            else:
                normalized = reduced
                
            return normalized
        except Exception as e:
            self.log.error(f"Preprocessing failed: {e}")
            return None

    def transcribe(self, input_path: str, output_path: str) -> bool:
        """Robust transcription pipeline"""
        try:
            self.log.info(f"Starting transcription of {input_path}")
            
            # Preprocess entire file first
            processed = self._safe_preprocess(input_path)
            if processed is None:
                raise ValueError("Audio preprocessing failed")
                
            # Export to temp file
            temp_path = os.path.join(self.temp_dir, "processed.wav")
            sf.write(temp_path, processed, self.target_sample_rate)
            
            # Transcribe in one go (more reliable than chunks)
            result = self.model.transcribe(
                temp_path,
                language="en",
                initial_prompt="This is a Christian sermon with Zimbabwean accent",
                verbose=self.verbose
            )
            
            with open(output_path, 'w') as f:
                f.write(result['text'])
                
            self.log.info(f"Successfully saved transcript to {output_path}")
            return True
            
        except Exception as e:
            self.log.error(f"Transcription failed: {e}")
            return False
            
        finally:
            # Cleanup
            for f in os.listdir(self.temp_dir):
                os.remove(os.path.join(self.temp_dir, f))
            os.rmdir(self.temp_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("-o", "--output", default="transcript.txt", help="Output file")
    parser.add_argument("-m", "--model", default="base.en", help="Whisper model size")
    args = parser.parse_args()
    
    transcriber = RobustSermonTranscriber(model_size=args.model)
    transcriber.transcribe(args.input, args.output)
