class RobustSermonTranscriber:
    """Optimized for reliability with African accents"""
    
    def __init__(self, model_size: str = "base.en", verbose: bool = True):
        self.verbose = verbose
        self.target_sample_rate = 16000
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

    def _safe_preprocess(self, audio_path: str) -> Optional[np.ndarray]:
        """Corrected audio preprocessing"""
        try:
            # Load audio file
            audio = AudioSegment.from_file(audio_path)
            
            # Basic validation
            if len(audio) < 500:  # At least 500ms
                raise ValueError("Audio too short")
                
            # Standardize format
            audio = audio.set_frame_rate(self.target_sample_rate).set_channels(1)
            
            # Convert to numpy array
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            samples /= np.iinfo(np.int16).max  # Normalize to [-1, 1]
            
            # Noise reduction
            reduced = nr.reduce_noise(
                y=samples,
                sr=audio.frame_rate,
                stationary=True,
                prop_decrease=0.8
            )
            
            # Loudness normalization - CORRECTED
            meter = pyloudnorm.Meter(audio.frame_rate)
            loudness = meter.integrated_loudness(reduced)
            normalized = pyloudnorm.normalize.loudness(reduced, loudness, -20.0)
            
            # Soft clipping to prevent distortion
            normalized = np.tanh(normalized * 0.95)
            
            return normalized
            
        except Exception as e:
            self.log.error(f"Preprocessing failed: {e}")
            return None

    def transcribe(self, input_path: str, output_path: str) -> bool:
        """Robust transcription pipeline"""
        try:
            self.log.info(f"Starting transcription of {input_path}")
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Preprocess audio
            processed = self._safe_preprocess(input_path)
            if processed is None:
                raise ValueError("Audio preprocessing failed")
                
            # Save processed audio to temp file
            temp_path = os.path.join(self.temp_dir, "processed.wav")
            sf.write(temp_path, processed, self.target_sample_rate)
            
            # Transcribe with Zimbabwean accent context
            result = self.model.transcribe(
                temp_path,
                language="en",
                initial_prompt=(
                    "This is a Christian sermon with Zimbabwean English accent. "
                    "Common features: 'bible' pronounced as 'bhibheri', "
                    "'prayer' as 'praya', 'amen' as 'amhen'."
                ),
                verbose=self.verbose
            )
            
            # Save transcript
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result['text'])
                
            self.log.info(f"Successfully saved transcript to {output_path}")
            return True
            
        except Exception as e:
            self.log.error(f"Transcription failed: {e}")
            return False
            
        finally:
            # Cleanup temp files
            for f in os.listdir(self.temp_dir):
                os.remove(os.path.join(self.temp_dir, f))
            os.rmdir(self.temp_dir)
