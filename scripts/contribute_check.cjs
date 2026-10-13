/**
 * 数据众筹链路校验（需要 Playwright + Chromium，用假麦克风）。
 *
 * 存在的理由：「打包下载」是纯本地功能，一旦 zip 写错（CRC、目录偏移、
 * central directory），用户拿到打不开的文件，而且没有任何报错——页面
 * 只会看到一个成功的下载。所以这里用系统 unzip 真解真校验。
 * 同时验证隐私设计：录音样本不许进 localStorage。
 *
 * 用法：
 *   npm install && npx playwright install chromium
 *   (cd public && python3 -m http.server 8931 &) ; sleep 2
 *   node scripts/contribute_check.cjs
 */
const fs = require('fs');
const { execSync } = require('child_process');
const { chromium } = require('playwright');

const URL = process.env.MEOWSCOPE_URL || 'http://127.0.0.1:8931/index.html';
const fails = [];
const ok = (cond, msg) => { if (!cond) fails.push(msg); };

(async () => {
  const browser = await chromium.launch({
    executablePath: process.env.CHROMIUM_PATH || undefined,
    args: ['--no-sandbox', '--use-fake-device-for-media-stream',
           '--use-fake-ui-for-media-stream', '--autoplay-policy=no-user-gesture-required'],
  });
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
  await page.goto(URL, { waitUntil: 'load' });
  await page.waitForTimeout(800);

  // 中文界面，滚到众筹区
  if (await page.evaluate(() => window.__i18n.LANG) !== 'zh') {
    await page.click('#lang-btn'); await page.waitForTimeout(300);
  }
  await page.evaluate(() => document.getElementById('contribute').scrollIntoView({ block: 'center' }));
  await page.fill('#c-breed', '狸花猫');
  await page.fill('#c-age', '2 岁');

  // 第一段：等饭，5 秒自动停
  await page.selectOption('#c-label', 'waiting_for_food');
  await page.click('#c-record');
  await page.waitForTimeout(5600);
  let n = await page.evaluate(() => window.__ms.cSamples().length);
  ok(n === 1, `录音后应有 1 段样本，实际 ${n}`);
  const firstTag = await page.evaluate(() => {
    const t = document.querySelector('#c-list .c-tag');
    return t ? t.textContent : '';
  });
  ok(firstTag.includes('等饭'), `样本标签应为「等饭」，实际「${firstTag}」`);
  console.log(`录音：${n} 段，标签「${firstTag}」`);

  // 第二段：隔离，手动停
  await page.selectOption('#c-label', 'isolation');
  await page.click('#c-record');
  await page.waitForTimeout(1800);
  await page.click('#c-record');
  await page.waitForTimeout(500);
  n = await page.evaluate(() => window.__ms.cSamples().length);
  ok(n === 2, `手动停止后应有 2 段样本，实际 ${n}`);

  // 打包下载并用系统 unzip 真校验
  const [download] = await Promise.all([
    page.waitForEvent('download'),
    page.click('#c-export'),
  ]);
  const zipPath = await download.path();
  const dest = '/tmp/meowscope_contrib_check.zip';
  fs.copyFileSync(zipPath, dest);
  ok(/^meowscope-samples-\d{8}-\d{4}\.zip$/.test(download.suggestedFilename()),
    `zip 文件名不规范：${download.suggestedFilename()}`);
  let test = '';
  try {
    test = execSync(`unzip -t ${dest} 2>&1`).toString();
  } catch (e) {
    test = (e.stdout || '') + (e.stderr || '');
  }
  ok(test.includes('No errors detected'), `zip 解压校验失败：\n${test}`);
  console.log(`zip：${download.suggestedFilename()}，unzip 校验 ${test.includes('No errors detected') ? '通过' : '失败'}`);

  execSync(`rm -rf /tmp/meowscope_contrib_x && mkdir -p /tmp/meowscope_contrib_x && cd /tmp/meowscope_contrib_x && unzip -q ${dest}`);
  let manifest = {};
  try {
    manifest = JSON.parse(fs.readFileSync('/tmp/meowscope_contrib_x/manifest.json', 'utf8'));
  } catch (e) { /* 下面统一报错 */ }
  ok(manifest.tool === 'meowscope-contribution', 'manifest.tool 不正确');
  ok(manifest.license === 'CC0-1.0', 'manifest.license 应为 CC0-1.0');
  ok(manifest.cat && manifest.cat.breed === '狸花猫', 'manifest.cat.breed 未写入');
  ok(Array.isArray(manifest.samples) && manifest.samples.length === 2,
    `manifest.samples 应有 2 条，实际 ${manifest.samples && manifest.samples.length}`);
  ok(manifest.samples && manifest.samples.every((s) => s.duration_s > 0.2 && s.file && s.label),
    'manifest.samples 字段缺失');
  ok(fs.existsSync('/tmp/meowscope_contrib_x/README.txt'), 'zip 内缺少 README.txt');
  console.log(`manifest：tool=${manifest.tool} license=${manifest.license} samples=${manifest.samples ? manifest.samples.length : 0}`);

  // 隐私：音频样本不许落 localStorage
  const lsKeys = await page.evaluate(() => Object.keys(localStorage));
  ok(!lsKeys.some((k) => k.includes('meowscope_contrib') || k.includes('sample')),
    `localStorage 出现疑似音频样本键：${lsKeys.join(',')}`);
  console.log(`localStorage 键：${lsKeys.join(', ') || '（空）'}`);

  // 删除 / 清空
  await page.click('#c-list li:first-child .c-acts button:last-child');
  await page.waitForTimeout(200);
  let afterDel = await page.evaluate(() => window.__ms.cSamples().length);
  await page.click('#c-clear');
  await page.waitForTimeout(200);
  let afterClear = await page.evaluate(() => window.__ms.cSamples().length);
  ok(afterDel === 1 && afterClear === 0, `删除/清空异常：del=${afterDel} clear=${afterClear}`);

  await browser.close();
  if (errors.length) { console.error('页面错误:'); errors.forEach((e) => console.error(' - ' + e)); fails.push('页面有 JS 错误'); }

  if (fails.length) {
    console.error('\nFAIL 数据众筹校验未通过：');
    fails.forEach((f) => console.error('  - ' + f));
    process.exit(1);
  }
  console.log('\nPASS 数据众筹校验通过（录音 / 列表 / zip 真解压 / manifest / 隐私 / 删除清空）');
})()
