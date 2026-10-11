// 检查 public/index.html 里引用的静态资源都真实存在（防止改完忘记同步文件）。
const fs = require('fs');
const path = require('path');

const root = path.join(__dirname, '..', 'public');
const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');

const refs = new Set();
for (const m of html.matchAll(/(?:src|href)="(\.\/[^"]+)"/g)) refs.add(m[1]);
// 脚本里 fetch / import 的资源
for (const m of html.matchAll(/(?:fetch|import)\(\s*'\.\/([^']+)'/g)) refs.add('./' + m[1]);
for (const m of html.matchAll(/import\s+[^']*from\s+'\.\/([^']+)'/g)) refs.add('./' + m[1]);
// manifest 里声明的图标
try {
  const mf = JSON.parse(fs.readFileSync(path.join(root, 'manifest.webmanifest'), 'utf8'));
  for (const ic of mf.icons || []) refs.add('./' + ic.src);
} catch (e) { /* manifest 缺失由下面统一报错 */ }
// service worker 预缓存清单
try {
  const sw = fs.readFileSync(path.join(root, 'sw.js'), 'utf8');
  for (const m of sw.matchAll(/'\.\/([^']+)'/g)) refs.add('./' + m[1]);
} catch (e) {}

const missing = [...refs].filter(r => !fs.existsSync(path.join(root, r.replace(/^\.\//, ''))));
const unused = fs.readdirSync(root)
  .filter((f) => !['index.html'].includes(f))
  .filter((f) => !refs.has('./' + f));

if (missing.length) {
  console.error('FAIL 引用了不存在的文件：', missing.join(', '));
  process.exit(1);
}
console.log(`PASS 引用的 ${refs.size} 个资源全部存在`);
if (unused.length) console.log('提示：未被 index.html/manifest/sw 引用的文件：', unused.join(', '));
