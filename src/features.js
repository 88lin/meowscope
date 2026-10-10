// features.js — replicate librosa MFCC feature extraction in pure JS.
// Matches train.py: sr=22050, n_mfcc=20, n_fft=2048, hop=512 (librosa defaults),
// slaney mel filterbank, DCT-II orthogonal, power=2.
const SR = 22050, N_FFT = 2048, HOP = 512, N_MELS = 128, N_MFCC = 20;

function hzToMel(f, htk = false) {
  if (htk) return 2595 * Math.log10(1 + f / 700);
  const fSp = 200 / 3, mMin = 1000 / fSp, logstep = Math.log(6.4) / 27;
  return f < 1000 ? f / fSp : mMin + Math.log(f / 1000) / logstep;
}
function melToHz(m, htk = false) {
  if (htk) return 700 * (Math.pow(10, m / 2595) - 1);
  const fSp = 200 / 3, mMin = 1000 / fSp, logstep = Math.log(6.4) / 27;
  return m < mMin ? m * fSp : 1000 * Math.exp(logstep * (m - mMin));
}

// librosa.filters.mel(sr, n_fft, n_mels=128, fmin=0, fmax=null, htk=false, norm='slaney')
function melFilterbank() {
  const fmax = SR / 2;
  const nFreqs = 1 + N_FFT / 2;
  const fftFreqs = Array.from({ length: nFreqs }, (_, i) => i * SR / N_FFT);
  const melF = (f) => hzToMel(f);
  const minMel = melF(0), maxMel = melF(fmax);
  const mels = Array.from({ length: N_MELS + 2 }, (_, i) => minMel + i * (maxMel - minMel) / (N_MELS + 1));
  const melsHz = mels.map((m) => melToHz(m));
  // slaney norm: each triangular filter normalized by its bandwidth (in mel? librosa: norm='slaney' divides by f_diff in Hz)
  const fdiff = [];
  for (let i = 0; i < N_MELS + 1; i++) fdiff.push(melsHz[i + 1] - melsHz[i]);
  const fb = [];
  for (let m = 0; m < N_MELS; m++) {
    const row = new Float64Array(nFreqs);
    const lows = melsHz[m], centers = melsHz[m + 1], highs = melsHz[m + 2];
    // ramp = (fftfreqs - low) / (center - low), clipped both sides
    const enorm = 2 / (melsHz[m + 2] - melsHz[m]);
    for (let k = 0; k < nFreqs; k++) {
      const f = fftFreqs[k];
      let down = (f - lows) / (centers - lows);
      if (down < 0) continue;
      let up = (highs - f) / (highs - centers);
      const w = Math.max(0, Math.min(down, up));
      row[k] = w * enorm;
    }
    fb.push(row);
  }
  return fb;
}

// naive but correct DFT for 1025 bins (n_fft=2048) — cached basis
let basisCache = null;
function dftBasis() {
  if (basisCache) return basisCache;
  const n = N_FFT, half = n / 2 + 1;
  const cos = new Float64Array(half * n), sin = new Float64Array(half * n);
  for (let k = 0; k < half; k++)
    for (let t = 0; t < n; t++) {
      const a = 2 * Math.PI * k * t / n;
      cos[k * n + t] = Math.cos(a); sin[k * n + t] = Math.sin(a);
    }
  basisCache = { cos, sin, half };
  return basisCache;
}

function frameSignal(y) {
  const nFrames = 1 + Math.floor((y.length - N_FFT) / HOP);
  if (nFrames <= 0) return null;
  const frames = new Float64Array(nFrames * N_FFT);
  for (let i = 0; i < nFrames; i++)
    for (let t = 0; t < N_FFT; t++)
      frames[i * N_FFT + t] = y[i * HOP + t] * (0.5 - 0.5 * Math.cos(2 * Math.PI * t / N_FTT_SAFE));
  return frames;
}
const N_FTT_SAFE = N_FFT; // hann window periodic

function stftPower(y) {
  const { cos, sin, half } = dftBasis();
  const nFrames = 1 + Math.floor((y.length - N_FFT) / HOP);
  const S = new Float64Array(nFrames * half);
  const buf = new Float64Array(N_FFT);
  for (let i = 0; i < nFrames; i++) {
    for (let t = 0; t < N_FFT; t++)
      buf[t] = y[i * HOP + t] * (0.5 - 0.5 * Math.cos(2 * Math.PI * t / N_FFT));
    for (let k = 0; k < half; k++) {
      let re = 0, im = 0;
      const off = k * N_FFT;
      for (let t = 0; t < N_FFT; t++) {
        const x = buf[t];
        re += cos[off + t] * x; im -= sin[off + t] * x;
      }
      S[i * half + k] = re * re + im * im;
    }
  }
  return { S, nFrames, half };
}

function melSpectrogram(y) {
  const { S, nFrames, half } = stftPower(y);
  const fb = melFilterbank();
  const mel = new Float64Array(nFrames * N_MELS);
  for (let i = 0; i < nFrames; i++)
    for (let m = 0; m < N_MELS; m++) {
      let sum = 0; const row = fb[m], off = i * half;
      for (let k = 0; k < half; k++) sum += row[k] * S[off + k];
      // librosa power_to_db: 10*log10(max(S, amin)) - ref, amin=1e-10, top_db=80
      mel[m * nFrames + i] = 10 * Math.log10(Math.max(sum, 1e-10));
    }
  // top_db 80 relative to max
  let maxV = -Infinity;
  for (let i = 0; i < mel.length; i++) if (mel[i] > maxV) maxV = mel[i];
  for (let i = 0; i < mel.length; i++) mel[i] = Math.max(mel[i], maxV - 80);
  return { mel, nFrames };
}

// librosa.feature.mfcc: DCT-II, norm='ortho' applied on (n_mfcc, n_frames) of log-mel
function dct2(matrix, rows, cols) { // rows=n_mfcc out, cols=nFrames, input n_mels x frames
  const out = new Float64Array(rows * cols);
  for (let c = 0; c < cols; c++)
    for (let r = 0; r < rows; r++) {
      let sum = 0;
      for (let m = 0; m < N_MELS; m++)
        sum += matrix[m * cols + c] * Math.cos(Math.PI * r * (2 * m + 1) / (2 * N_MELS));
      const norm = r === 0 ? Math.sqrt(1 / (4 * N_MELS)) : Math.sqrt(1 / (2 * N_MELS));
      out[r * cols + c] = sum * 2 * norm;
    }
  return out;
}

function centroidRolloffZcrRms(y, nFrames, half, S) {
  const freqs = Array.from({ length: half }, (_, i) => i * SR / N_FFT);
  const cent = new Float64Array(nFrames), rolloff = new Float64Array(nFrames), zcr = new Float64Array(nFrames), rms = new Float64Array(nFrames);
  for (let i = 0; i < nFrames; i++) {
    let num = 0, den = 0;
    for (let k = 1; k < half; k++) { const p = S[i * half + k]; num += freqs[k] * p; den += p; }
    cent[i] = den > 0 ? num / den : 0;
    // rolloff 0.85
    let total = 0; for (let k = 0; k < half; k++) total += S[i * half + k];
    let acc = 0, rk = half - 1;
    for (let k = 0; k < half; k++) { acc += S[i * half + k]; if (acc >= 0.85 * total) { rk = k; break; } }
    rolloff[i] = freqs[rk];
    let z = 0;
    const start = i * HOP;
    for (let t = 1; t <= N_FFT && start + t < y.length; t++)
      z += ((y[start + t] * y[start + t - 1]) < 0) ? 1 : 0;
    zcr[i] = z / N_FFT;
    let e = 0;
    for (let t = 0; t < N_FFT; t++) { const v = y[start + t]; e += v * v; }
    rms[i] = Math.sqrt(e / N_FFT);
  }
  return { cent, rolloff, zcr, rms };
}

function meanStd(arr) {
  let m = 0; for (const v of arr) m += v; m /= arr.length;
  let s = 0; for (const v of arr) s += (v - m) * (v - m);
  return [m, Math.sqrt(s / arr.length)];
}

export function extractFeatures(y) {
  // pad to at least 0.25s like train.py
  if (y.length < SR / 4) {
    const p = new Float64Array(Math.ceil(SR / 4)); p.set(y); y = p;
  }
  const { S, nFrames, half } = stftPower(y);
  const { mel } = melSpectrogram(y);
  const mfcc = dct2(mel, N_MFCC, nFrames); // r*cols+c, mel is m*cols+c
  // delta1: librosa.feature.delta, mode='interp', width=9
  const d1 = new Float64Array(N_MFCC * nFrames);
  const W = 4; // width 9 -> offsets -4..4, edge fill with nearest (mode interp uses edge padding then Savitzky-Golay; approximate with simple central diff of padded)
  for (let r = 0; r < N_MFCC; r++) {
    for (let c = 0; c < nFrames; c++) {
      const at = (i) => mfcc[r * nFrames + Math.max(0, Math.min(nFrames - 1, i))];
      let num = 0;
      for (let d = 1; d <= W; d++) num += d * (at(c + d) - at(c - d));
      const norm = 2 * (W * (W + 1) * (2 * W + 1) / 6);
      d1[r * nFrames + c] = num / norm;
    }
  }
  const feat = [];
  for (let r = 0; r < N_MFCC; r++) feat.push(meanStd(col(mfcc, r, nFrames))[0]);
  for (let r = 0; r < N_MFCC; r++) feat.push(meanStd(col(mfcc, r, nFrames))[1]);
  for (let r = 0; r < N_MFCC; r++) feat.push(meanStd(col(d1, r, nFrames))[0]);
  const { cent, rolloff, zcr, rms } = centroidRolloffZcrRms(y, nFrames, half, S);
  feat.push(meanStd(cent)[0], meanStd(cent)[1], meanStd(rolloff)[0], meanStd(zcr)[0], meanStd(zcr)[1], meanStd(rms)[0], Math.max(...rms));
  return feat;
}

function col(mat, r, nFrames) {
  const out = new Float64Array(nFrames);
  for (let c = 0; c < nFrames; c++) out[c] = mat[r * nFrames + c];
  return out;
}
