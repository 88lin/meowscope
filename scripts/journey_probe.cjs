/**
 * 用户之旅深度巡检（只读，不改页面）：走一遍真实用户可能遇到的路径，
 * 报告可见问题。与 ui_check.cjs（关卡式 PASS/FAIL）不同，这里是排查用的
 * 宽口径扫描——把「没崩但不对劲」的东西找出来。
 *
 * 用法：
 *   CHROMIUM_PATH=<chromium> node scripts/journey_probe.cjs
 */
const { chromium } = require('playwright');

const URL = process.env.MEOWSCOPE_URL || 'http://127.0.0.1:8931/index.html';
const notes = [];
const note = (s) => { notes.push(s); console.log(s); };

(async () => {
  const browser = await chromium.launch({
    executablePath: process.env.CHROMIUM_PATH || undefined,
    args: ['--no-sandbox', '--use-fake-device-for-media-stream',
           '--use-fake-ui-for-media-stream', '--autoplay-policy=no-user-gesture-required'],
  });
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const errs = [];
  page.on('pageerror', (e) => errs.push('pageerror: ' + e.message));
  page.on('console', (m) => { if (m.type() === 'error') errs.push('console: ' + m.text()); });
  await page.goto(URL, { waitUntil: 'load' });
  await page.waitForTimeout(1000);

  // ---- 1. 用真实猫叫样本走一遍（upfile 走 upload input）----
  await page.setInputFiles('#file', require('path').resolve(__dirname, '../tests/fixture.wav'));
  await page.waitForTimeout(2500);
  const res = await page.evaluate(() => {
    const vis = (id) => { const el = document.getElementById(id); return el && getComputedStyle(el).display !== 'none'; };
    return {
      resultVisible: vis('result'),
      main: document.getElementById('v-main')?.textContent || '',
      pct: document.getElementById('v-pct')?.textContent || '',
      meta: document.getElementById('v-meta')?.textContent || '',
      featF0: document.getElementById('feat-f0')?.textContent || '',
      featJit: document.getElementById('feat-jit')?.textContent || '',
      featDur: document.getElementById('feat-dur')?.textContent || '',
      melPainted: (() => {
        const c = document.getElementById('mel'); if (!c) return 'no-canvas';
        const g = c.getContext('2d'); const d = g.getImageData(0, 0, c.width, c.height).data;
        let n = 0; for (let i = 3; i < d.length; i += 4) if (d[i] > 0) n++;
        return Math.round(n / (d.length / 4) * 100) + '%';
      })(),
      shareVisible: vis('btn-share'), saveVisible: vis('btn-save'),
      speakVisible: vis('btn-speak'), compareVisible: vis('btn-compare'),
      historyCount: document.querySelectorAll('#history .h-item, #history [data-hist]').length,
      state: document.getElementById('state')?.textContent || '',
    };
  });
  note(`[1] 真实样本分析结果：${res.main} | 置信度 ${res.pct} | ${res.meta}`);
  note(`    频谱图绘制 ${res.melPainted} | 特征行 ${res.featF0}/${res.featJit}/${res.featDur}`);
  note(`    按钮可见：分享=${res.shareVisible} 保存=${res.saveVisible} 朗读=${res.speakVisible} 对照=${res.compareVisible}`);
  note(`    历史条数=${res.historyCount} | 状态行="${res.state}"`);

  // ---- 2. 语言切换后扫残留（找硬编码/漏译）----
  await page.click('#lang-btn'); await page.waitForTimeout(600);
  const langCheck = await page.evaluate(() => {
    const pick = (sel) => [...document.querySelectorAll(sel)]
      .filter((el) => el.offsetParent !== null)
      .map((el) => el.textContent.trim()).filter(Boolean);
    // 英文界面下仍含中日韩字符的可见文本 = 漏译
    const cjk = [...document.querySelectorAll('h1,h2,h3,p,button,label,span,small,div')]
      .filter((el) => el.offsetParent !== null && el.children.length === 0)
      .map((el) => el.textContent.trim())
      .filter((s) => s && /[一-鿿぀-ヿ]/.test(s));
    return { lang: window.__i18n.LANG, cjk: [...new Set(cjk)].slice(0, 12),
             main: document.getElementById('v-main')?.textContent || '' };
  });
  note(`[2] 切到 ${langCheck.lang} 后，英文界面残留中文文本：${langCheck.cjk.length ? JSON.stringify(langCheck.cjk) : '无'}`);
  await page.click('#lang-btn'); await page.waitForTimeout(400);

  // ---- 3. 边界：极短/静音/超长 ----
  const mk = async (seconds, kind) => {
    await page.evaluate(([sec, kind]) => {
      const sr = 22050, n = Math.floor(sr * sec);
      const buf = new AudioBuffer({ length: n, sampleRate: sr, numberOfChannels: 1 });
      const d = buf.getChannelData(0);
      for (let i = 0; i < n; i++) d[i] = kind === 'tone' ? 0.5 * Math.sin(2 * Math.PI * 700 * i / sr) : 0;
      // 直接手写 wav 头（22050Hz 16-bit 单声道），不依赖 AudioContext 的采样率
      const bytes = new Uint8Array(44 + n * 2);
      const dv = new DataView(bytes.buffer);
      const wr = (o, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i)); };
      wr(0, 'RIFF'); dv.setUint32(4, 36 + n * 2, true); wr(8, 'WAVEfmt ');
      dv.setUint32(16, 16, true); dv.setUint16(20, 1, true); dv.setUint16(22, 1, true);
      dv.setUint32(24, sr, true); dv.setUint32(28, sr * 2, true);
      dv.setUint16(32, 2, true); dv.setUint16(34, 16, true); wr(36, 'data');
      dv.setUint32(40, n * 2, true);
      for (let i = 0; i < n; i++) dv.setInt16(44 + i * 2, Math.max(-1, Math.min(1, d[i])) * 32767, true);
      const blob = new Blob([bytes], { type: 'audio/wav' });
      const dt = new DataTransfer();
      dt.items.add(new File([blob], 'x.wav', { type: 'audio/wav' }));
      const inp = document.getElementById('file');
      inp.files = dt.files;
      inp.dispatchEvent(new Event('change', { bubbles: true }));
    }, [seconds, kind]);
    await page.waitForTimeout(1800);
    return await page.evaluate(() => ({
      state: document.getElementById('state')?.textContent || '',
      main: document.getElementById('v-main')?.textContent || '',
      resultVisible: getComputedStyle(document.getElementById('result')).display !== 'none',
    }));
  };
  const short = await mk(0.1, 'tone'); note(`[3] 0.1s 超短音频 → 状态="${short.state}" 结果区可见=${short.resultVisible}`);
  const silent = await mk(2, 'silence'); note(`    2s 静音 → 状态="${silent.state}" main="${silent.main}"`);
  const long = await mk(30, 'tone'); note(`    30s 长音频 → 状态="${long.state}" main="${long.main}"`);

  // ---- 4. 非猫叫输入（700Hz 纯音，模拟人声基频外的信号）----
  const tone = await page.evaluate(() => ({
    main: document.getElementById('v-main')?.textContent || '',
    pct: document.getElementById('v-pct')?.textContent || '',
  }));
  note(`[4] 纯音输入判定：${tone.main} 置信度 ${tone.pct}（观察是否有非猫叫提示）`);

  // ---- 5. 键盘 / 无障碍 ----
  const a11y = await page.evaluate(() => {
    const btn = document.getElementById('btn-record');
    return {
      recordAria: btn?.getAttribute('aria-label') || '(无)',
      resultAria: document.getElementById('result')?.getAttribute('aria-label') || '(无)',
      langBtnAria: document.getElementById('lang-btn')?.getAttribute('aria-label') || '(无)',
      imgsNoAlt: [...document.querySelectorAll('img')].filter((i) => !i.alt).length,
      btnNoLabel: [...document.querySelectorAll('button')]
        .filter((b) => b.offsetParent !== null && !b.textContent.trim() && !b.getAttribute('aria-label')).length,
    };
  });
  note(`[5] 无障碍：录音按钮 aria="${a11y.recordAria}" 结果区 aria="${a11y.resultAria}"`);
  note(`    无 alt 图片=${a11y.imgsNoAlt} 无标签按钮=${a11y.btnNoLabel}`);

  // ---- 6. 移动端 375 关键区域溢出 ----
  await page.setViewportSize({ width: 375, height: 780 });
  await page.waitForTimeout(500);
  const mob = await page.evaluate(() => {
    const bad = [];
    document.querySelectorAll('section, .personal-box, .c-form, .h-item, .feat-row, #contribute').forEach((el) => {
      const r = el.getBoundingClientRect();
      if (r.right > window.innerWidth + 1 || r.left < -1) bad.push(`${el.tagName}#${el.id || el.className} ${Math.round(r.left)}..${Math.round(r.right)}`);
    });
    return { overflow: bad, scrollW: document.documentElement.scrollWidth, vw: window.innerWidth };
  });
  note(`[6] 375px 溢出：${mob.overflow.length ? JSON.stringify(mob.overflow) : '无'}（scrollW=${mob.scrollW} vw=${mob.vw}）`);

  note(`\n页面错误：${errs.length ? errs.join(' | ') : '无'}`);
  await browser.close();
})();
