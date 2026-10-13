/**
 * UI 校验（需要 Playwright + Chromium）。
 *
 * 存在的理由：只做「文字对比度」「几何溢出」这类检查抓不到
 * 「元素压根没画出来」的问题——比如进度条填充是 <span>，行内元素
 * 忽略 width/height，结果是 0×0，页面上什么都不显示却没有任何报错。
 * 所以这里用真实浏览器量渲染结果。
 *
 * 用法：
 *   npm install && npx playwright install chromium
 *   (cd public && python3 -m http.server 8931 &) ; sleep 2
 *   node scripts/ui_check.cjs
 */
const { chromium } = require('playwright');

const URL = process.env.MEOWSCOPE_URL || 'http://127.0.0.1:8931/index.html';
const fails = [];
const ok = (cond, msg) => { if (!cond) fails.push(msg); };

(async () => {
  // CI 里由 `npx playwright install chromium` 提供浏览器；
  // 本地可设 CHROMIUM_PATH 指向系统已装的 Chromium。
  const browser = await chromium.launch({
    executablePath: process.env.CHROMIUM_PATH || undefined,
    args: ['--no-sandbox'],
  });

  // ---------- 1) 结果区：进度条必须真的画出来 ----------
  {
    const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
    await page.goto(URL, { waitUntil: 'load' });
    await page.waitForTimeout(800);
    // 点第一个示例触发一次分析
    await page.click('.demo-btn[data-i="0"]');
    await page.waitForTimeout(3500);

    const bars = await page.evaluate(() => [...document.querySelectorAll('.bar-row')].map((r) => {
      const t = r.querySelector('.bar-track').getBoundingClientRect();
      const f = r.querySelector('.bar-fill').getBoundingClientRect();
      return {
        name: r.querySelector('.name').textContent.trim(),
        pct: parseFloat(r.querySelector('.pct').textContent),
        tw: t.width, th: t.height, fw: f.width, fh: f.height,
      };
    }));
    ok(bars.length === 3, `进度条应有 3 行，实际 ${bars.length}`);
    for (const b of bars) {
      ok(b.th >= 6, `「${b.name}」轨道高度异常：${b.th}px`);
      ok(b.fw > 1, `「${b.name}」填充宽度为 ${b.fw}px —— 进度条没画出来（多半是 display:inline 导致）`);
      ok(b.fh > 1, `「${b.name}」填充高度为 ${b.fh}px —— 进度条没画出来`);
      if (b.tw > 0) {
        const got = (b.fw / b.tw) * 100;
        ok(Math.abs(got - b.pct) < 2.5,
          `「${b.name}」填充占比 ${got.toFixed(1)}% 与标注的 ${b.pct}% 不符`);
      }
    }
    console.log(`进度条：${bars.map((b) => `${b.name} ${b.pct}%→${(b.fw / b.tw * 100).toFixed(0)}%`).join(' / ')}`);

    // 置信度环
    const ring = await page.evaluate(() => {
      const el = document.getElementById('ring-fill');
      const box = el.getBoundingClientRect();
      return { w: box.width, h: box.height, offset: parseFloat(el.style.strokeDashoffset || '207.3') };
    });
    ok(ring.w > 10 && ring.h > 10, `置信度环渲染异常：${ring.w}x${ring.h}`);
    ok(ring.offset < 207.3, `置信度环没有随概率变化（offset=${ring.offset}）`);
    console.log(`置信度环：${ring.w}x${ring.h}，offset=${ring.offset.toFixed(1)}`);

    // 梅尔频谱图：必须真的画出内容（空画布 = 又一个"摆设"）
    const mel = await page.evaluate(() => {
      const cv = document.getElementById('mel');
      if (!cv) return null;
      const d = cv.getContext('2d').getImageData(0, 0, cv.width, cv.height).data;
      let painted = 0;
      for (let i = 3; i < d.length; i += 4) if (d[i] > 0) painted++;
      return { w: cv.width, h: cv.height, painted, total: d.length / 4 };
    });
    ok(mel && mel.w >= 400, `梅尔频谱图画布异常：${mel && mel.w}x${mel && mel.h}`);
    ok(mel && mel.painted > mel.total * 0.2,
      `梅尔频谱图没有绘制内容（${mel && mel.painted}/${mel && mel.total} 像素）`);
    console.log(`梅尔频谱图：${mel.w}x${mel.h}，已绘制 ${(mel.painted / mel.total * 100).toFixed(0)}%`);

    // 声学特征行：三个值都应是具体数字，不能停留在占位符 —
    const feats = await page.evaluate(() => ({
      f0: document.getElementById('feat-f0').textContent.trim(),
      jit: document.getElementById('feat-jit').textContent.trim(),
      dur: document.getElementById('feat-dur').textContent.trim(),
    }));
    ok(/\d+\s*Hz/.test(feats.f0), `音高特征没有填充：${feats.f0}`);
    ok(/\d+\s*Hz/.test(feats.jit), `抖动特征没有填充：${feats.jit}`);
    ok(/\d+(\.\d+)?\s*s/.test(feats.dur), `时长特征没有填充：${feats.dur}`);
    console.log(`声学特征：音高 ${feats.f0} · 抖动 ${feats.jit} · 时长 ${feats.dur}`);

    // 语言切换：aria-label 之类的属性也要跟随（data-i18n-aria）
    const ariaBefore = await page.evaluate(() => document.getElementById('result').getAttribute('aria-label'));
    await page.click('#lang-btn');
    await page.waitForTimeout(400);
    const ariaAfter = await page.evaluate(() => document.getElementById('result').getAttribute('aria-label'));
    ok(!!ariaBefore && ariaBefore !== ariaAfter,
      `aria-label 未随语言更新（${ariaBefore} → ${ariaAfter}）`);
    console.log(`aria-label 语言切换：${ariaBefore} → ${ariaAfter}`);
    await page.close();
  }

  // ---------- 2) 移动端：不能横向溢出 ----------
  for (const w of [375, 320]) {
    const page = await browser.newPage({ viewport: { width: w, height: 780 } });
    await page.goto(URL, { waitUntil: 'load' });
    await page.waitForTimeout(600);
    const over = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    ok(over <= 1, `${w}px 宽度下横向溢出 ${over}px`);
    console.log(`${w}px：横向溢出 ${over}px`);
    await page.close();
  }

  // ---------- 3) 明暗两模式：抽查关键文字对比度 ----------
  for (const scheme of ['light', 'dark']) {
    const page = await browser.newPage({ viewport: { width: 1280, height: 900 }, colorScheme: scheme });
    await page.goto(URL, { waitUntil: 'load' });
    await page.waitForTimeout(700);
    const bad = await page.evaluate(() => {
      const lum = (r, g, b) => {
        const f = (c) => { c /= 255; return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); };
        return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b);
      };
      const parse = (s) => { const m = s.match(/rgba?\(([^)]+)\)/); if (!m) return null;
        const v = m[1].split(',').map(Number); return { r: v[0], g: v[1], b: v[2], a: v.length > 3 ? v[3] : 1 }; };
      const bgOf = (el) => { let n = el; while (n) { const c = parse(getComputedStyle(n).backgroundColor);
        if (c && c.a > 0.99) return c; n = n.parentElement; } return { r: 255, g: 255, b: 255, a: 1 }; };
      const out = [];
      for (const sel of ['h1', '.hero p.lead', '.hint', 'section h2', '.card p', 'footer', '.bar-row .name']) {
        const el = document.querySelector(sel); if (!el) continue;
        const fg = parse(getComputedStyle(el).color), bg = bgOf(el);
        const cr = (Math.max(lum(fg.r, fg.g, fg.b), lum(bg.r, bg.g, bg.b)) + 0.05) /
                   (Math.min(lum(fg.r, fg.g, fg.b), lum(bg.r, bg.g, bg.b)) + 0.05);
        if (cr < 4.5) out.push({ sel, cr: +cr.toFixed(2) });
      }
      return out;
    });
    ok(bad.length === 0, `${scheme} 模式对比度不足：${JSON.stringify(bad)}`);
    console.log(`${scheme} 模式：抽查对比度 ${bad.length === 0 ? '全部达标' : JSON.stringify(bad)}`);
    await page.close();
  }

  await browser.close();

  if (fails.length) {
    console.error('\nFAIL UI 校验未通过：');
    fails.forEach((f) => console.error('  - ' + f));
    process.exit(1);
  }
  console.log('\nPASS UI 校验通过（进度条 / 置信度环 / 移动端溢出 / 明暗对比度）');
})();
