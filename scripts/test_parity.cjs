// parity test: compare JS extractFeatures vs Python librosa features
const fs = require("fs");
const { execSync } = require("child_process");
// Node 24 can't import ESM .js from CJS easily; use dynamic import
(async () => {
  // 注意：features.js 在 public/ 下，这里必须回到上层目录去找
  const { extractFeatures, extractF0Stats } = await import("../public/features.js");
  // read wav via minimal parser (16-bit PCM mono)
  function readWav(path) {
    const b = fs.readFileSync(path);
    const dataIdx = b.indexOf("data");
    const fmt = b.indexOf("fmt ");
    const bits = b.readUInt16LE(fmt + 22);
    const channels = b.readUInt16LE(fmt + 10);
    const rate = b.readUInt32LE(fmt + 12);
    // chunk size field can be wrong (PAD chunk); trust file length
    let off = dataIdx + 8;
    const len = b.length - off;
    const n = Math.floor(len / (bits / 8) / channels);
    const y = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let v = 0;
      for (let c = 0; c < channels; c++) {
        const o = off + (i * channels + c) * (bits / 8);
        v += bits === 16 ? b.readInt16LE(o) / 32768 : b.readUInt8(o) / 128 - 1;
      }
      y[i] = v / channels;
    }
    return { y, rate };
  }
  const path = process.argv[2];
  let { y, rate } = readWav(path);
  if (rate !== 22050) {
    // linear resample to 22050
    const n = Math.floor(y.length * 22050 / rate);
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const p = i * rate / 22050, i0 = Math.floor(p);
      out[i] = y[i0] + (y[Math.min(i0 + 1, y.length - 1)] - y[i0]) * (p - i0);
    }
    y = out;
  }
  const t0 = Date.now();
  const feat = extractFeatures(y);
  const f0 = extractF0Stats(y);
  console.log(JSON.stringify({ feat, f0, ms: Date.now() - t0 }));
})();
