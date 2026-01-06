import numpy as np
from pydub import AudioSegment
import python_speech_features as psf
import io

def extract_mfcc_from_bytes(audio_bytes):
    """
    Extrait les MFCC normalisés sans silence depuis un fichier audio en bytes.
    """
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")
    audio = audio.set_frame_rate(16000).set_channels(1)
    
    samples = np.array(audio.get_array_of_samples())
    sr = audio.frame_rate
    
    mfccs = psf.mfcc(samples, sr, numcep=13, nfft=512, winfunc=np.hamming, appendEnergy=False)
    deltas = psf.delta(mfccs, 2)
    energies = np.sum(np.square(mfccs), axis=1).reshape(-1, 1)
    
    features = np.hstack((mfccs, deltas))
    
    # Normalisation CMVN
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    features_norm = (features - mean) / (std + 1e-8)
    
    # Suppression du silence
    threshold = 0.1 * np.max(energies)
    voiced_frames = energies.ravel() > threshold
    features_voiced_norm = features_norm[voiced_frames, :]
    
    return features_voiced_norm
