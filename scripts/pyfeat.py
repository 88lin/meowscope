import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import librosa
from train import SR, N_MFCC

path = sys.argv[1]
y, _ = librosa.load(path, sr=SR)
if len(y) < SR // 4:
    y = np.pad(y, (0, SR // 4 - len(y)))
mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
d1 = librosa.feature.delta(mfcc)
cent = librosa.feature.spectral_centroid(y=y, sr=SR)
rolloff = librosa.feature.spectral_rolloff(y=y, sr=SR)
zcr = librosa.feature.zero_crossing_rate(y)
rms = librosa.feature.rms(y=y)
feat = np.concatenate([
    mfcc.mean(1), mfcc.std(1), d1.mean(1),
    [cent.mean(), cent.std(), rolloff.mean(), zcr.mean(), zcr.std(), rms.mean(), rms.max()],
])
print(json.dumps({"feat": feat.tolist()}))
