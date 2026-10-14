/**
 * 闸门：端到端判定必须与 Python(train.py + joblib) 一致。
 *
 * 为什么需要它（它抓到过一个真 bug）：
 *   parity_check 只验证「每个特征算得对不对」（89 维 + 6 维分别比对），
 *   但没验证「喂给森林的向量是不是模型期望的那一个」。曾经 extractAll 把
 *   6 维音高单独返回、调用处忘记拼回去，485 个用到音高的决策节点拿到
 *   undefined 全部走右分支——特征本身全对，判定却是错的，旧闸门全绿。
 *   所以这里直接比最终概率：浏览器判出来的三类概率必须等于 Python。
 *
 * 用法：
 *   (cd public && python3 -m http.server 8931 &) ; sleep 2
 *   node scripts/judgement_check.cjs
 */
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

const URL = process.env.MEOWSCOPE_URL || 'http://127.0.0.1:8931/index.html';
const EXP = JSON.parse(fs.readFileSync(path.resolve(__dirname, '../tests/fixture_expected.json'), 'utf8'));
const FIXTURE = path.resolve(__dirname, '../tests/fixture.wav');
const TOL_P = 0.005;    // 概率容差：float32/float64 与树遍历顺序差异的量级
const TOL_F = 2e-4;     // 特征容差
const fails = [];
const ok = (c, m) => { if (!c) fails.push(m); };

(async () => {
  const browser = await chromium.launch({
    executablePath: process.env.CHROMIUM_PATH || undefined, args: ['--no-sandbox'],
  });
  const page = await browser.newPage();
  const errs = [];
  page.on('pageerror', (e) => errs.push('pageerror: ' + e.message));
  page.on('console', (m) => { if (m.type() === 'error') errs.push('console: ' + m.text()); });
  await page.goto(URL, { waitUntil: 'load' });
  await page.waitForTimeout(1000);

  // 真实用户路径：走文件输入 → 完整 analyze 流程（而不是直接调内部函数）
  await page.setInputFiles('#file', FIXTURE);
  await page.waitForTimeout(2500);

  const got = await page.evaluate(async () => {
    await window.__ms.loadForest ? window.__ms.loadForest() : null;
    return { preds: window.__ms.preds ? window.__ms.preds.slice() : null };
  });
  ok(Array.isArray(got.preds) && got.preds.length === 3, `浏览器未产出三类概率：${JSON.stringify(got.preds)}`);

  if (Array.isArray(got.preds)) {
    const byCls = {};
    got.preds.forEach((p) => { byCls[p.c] = p.p; });
    // 期望文件的 classes 是 ['0','1','2']，对应 brushing / waiting_for_food / isolation
    const map = { 0: 'brushing', 1: 'waiting_for_food', 2: 'isolation' };
    const rows = [];
    for (const k of Object.keys(EXP.proba)) {
      const name = map[k];
      const want = EXP.proba[k];
      const have = byCls[name];
      rows.push({ name, want, have });
      ok(have !== undefined, `浏览器缺少类别 ${name} 的概率`);
      if (have !== undefined) {
        ok(Math.abs(have - want) <= TOL_P,
          `${name} 概率不一致：浏览器 ${have.toFixed(4)} vs Python ${want.toFixed(4)}（容差 ${TOL_P}）`);
      }
    }
    console.log('端到端判定对拍（fixture.wav，1.6s）：');
    rows.forEach((r) => console.log(`  ${r.name.padEnd(16)} 浏览器 ${(r.have ?? NaN).toFixed(4)}  Python ${r.want.toFixed(4)}  差 ${Math.abs((r.have ?? NaN) - r.want).toFixed(5)}`));
  }

  // 顺带断言：喂给森林的向量维度 == 模型 n_features（直接堵住漏拼这类问题）
  const dim = await page.evaluate(() => {
    const f = window.__ms.extractAll(new Float32Array(22050 * 2));
    return { feat: f.feat.length, f0: f.f0.length, joined: f.feat.concat(f.f0).length };
  });
  console.log(`特征维度：feat=${dim.feat} + f0=${dim.f0} = ${dim.joined}（模型期望 ${EXP.n_features}）`);
  ok(dim.joined === EXP.n_features,
    `拼好的特征维度 ${dim.joined} != 模型期望 ${EXP.n_features}（漏拼/多拼会静默走错分支）`);

  // ---- 长音频自动截取：音频在页面内合成，不依赖外部文件 ----
  // 训练数据全部落在 1–4 秒，超过 4 秒必须截取到发声段，否则整段统计的特征会被
  // 静音稀释、退化到多数类。这里合成「6s 底噪 + 2s 强音 + 6s 底噪」验证定位。
  const trim = await page.evaluate(async () => {
    const SR = 22050, sec = 14, n = SR * sec;
    const d = new Float32Array(n);
    for (let i = 0; i < n; i++) d[i] = (Math.random() - 0.5) * 0.001;      // 底噪
    for (let i = SR * 6; i < SR * 8; i++) {                                // 6–8s 强音
      const t = (i - SR * 6) / SR;
      d[i] = 0.35 * Math.sin(2 * Math.PI * 700 * t) + 0.15 * Math.sin(2 * Math.PI * 1400 * t);
    }
    const bytes = new Uint8Array(44 + n * 2), dv = new DataView(bytes.buffer);
    const wr = (o, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i)); };
    wr(0, 'RIFF'); dv.setUint32(4, 36 + n * 2, true); wr(8, 'WAVEfmt ');
    dv.setUint32(16, 16, true); dv.setUint16(20, 1, true); dv.setUint16(22, 1, true);
    dv.setUint32(24, SR, true); dv.setUint32(28, SR * 2, true);
    dv.setUint16(32, 2, true); dv.setUint16(34, 16, true); wr(36, 'data');
    dv.setUint32(40, n * 2, true);
    for (let i = 0; i < n; i++) dv.setInt16(44 + i * 2, Math.max(-1, Math.min(1, d[i])) * 32767, true);
    const dt = new DataTransfer();
    dt.items.add(new File([bytes], 'long.wav', { type: 'audio/wav' }));
    const inp = document.getElementById('file');
    inp.files = dt.files;
    inp.dispatchEvent(new Event('change', { bubbles: true }));
    await new Promise((r) => setTimeout(r, 2500));
    const t = window.__ms.lastTrim;
    const note = document.getElementById('wave-note');
    return { trim: t ? { a: t.a, b: t.b, total: t.total } : null,
             noteShown: note ? !note.hidden : false,
             noteText: note ? note.textContent.slice(0, 40) : '' };
  });
  console.log(`长音频截取：${trim.trim ? `${trim.trim.a.toFixed(2)}–${trim.trim.b.toFixed(2)}s / 共 ${trim.trim.total.toFixed(1)}s` : '未触发'}，说明可见=${trim.noteShown}`);
  ok(trim.trim !== null, '14s 音频应触发截取');
  if (trim.trim) {
    const len = trim.trim.b - trim.trim.a;
    ok(len >= 1.7 && len <= 4.05, `截取窗口应在 1.8–4.0s，实际 ${len.toFixed(2)}s`);
    const ov = Math.max(0, Math.min(trim.trim.b, 8.0) - Math.max(trim.trim.a, 6.0));
    ok(ov >= 1.5, `窗口应与发声段(6–8s)重叠 ≥1.5s，实际 ${ov.toFixed(2)}s`);
  }
  ok(trim.noteShown, '长音频应显示截取说明（告诉用户分析的是哪一段）');

  await browser.close();
  if (errs.length) { console.error('页面错误:', errs.join(' | ')); fails.push('页面有 JS 错误'); }
  if (fails.length) {
    console.error('\nFAIL 端到端判定对拍未通过：');
    fails.forEach((f) => console.error('  - ' + f));
    process.exit(1);
  }
  console.log('\nPASS 端到端判定与 Python 一致（概率 + 喂模型的向量维度）');
})();
