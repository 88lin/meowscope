# 喵语通 MeowScope 🐱

![License: MIT](https://img.shields.io/badge/License-MIT-green) ![Zero dependency](https://img.shields.io/badge/dependencies-0-blue) ![Privacy](https://img.shields.io/badge/uploads-0-orange)

**在浏览器里听懂猫在说什么。** 录音或上传猫叫，本地 AI 实时判断它在哪种情境下发声（被刷毛 / 等饭 / 被隔离）——所有计算都在你的浏览器完成，**音频不会上传到任何服务器**。

在线体验：https://88lin.github.io/meowscope/ （或直接打开 `public/index.html` 本地跑）

> 分享到微信 / X / Telegram 时的链接预览图：
> ![MeowScope 分享预览图](public/og-image.png)

分析结果可以一键生成 1200×630 的分享卡片（含波形、三条概率、置信度），直接保存或复制：

![分享卡片示例](docs/share-card-example.png)

## 它能听懂什么

| 情境 | 场景 | 特点 |
|---|---|---|
| 🪮 被刷毛 brushing | 主人梳毛时 | 短促、音调平缓 |
| 🍚 等饭 waiting_for_food | 站在食盆前 | 急促重复、最有辨识度 |
| 🚪 被隔离 isolation | 被独自留在陌生环境 | 鸣叫更长、音更高 |

先说实话：**喵语没有百科全书**。这个模型只做情境分类，不做"翻译句子"。同一声喵在不同猫、不同场合含义可以完全不同（学界称为 *semantic aliasing*，语义混叠）。

## 诚实的成绩单

训练/测试**按猫划分**（模型从未听过测试集里那只猫的声音），不是作弊的随机划分：

| 验证方式 | 准确率 |
|---|---|
| 按猫分组 5 折交叉验证 | **60.9%** |
| 留一猫验证 (LOSO) | **58.9%** |
| 随机猜 | 33.3% |

细节见 [`src/metrics.json`](src/metrics.json)。参照系：Nicastro & Owren (2003) 与 Prato-Previde et al. (2020) 测得**人类**分辨陌生猫发声情境仅 26-41% 正确——本模型已高于陌生人类，但仍远低于"养猫人听自家猫"。很多宠物翻译产品宣传 90%+，那是"同一只猫的录音再喂回去"的成绩。我们选择公布严格数字。

## 工作原理

```
音频 → 22.05kHz 单声道 → 20 维 MFCC + 一阶差分 + 谱统计（67 维特征）
     → 随机森林 150 棵（depth 6）→ 三类概率
```

- **特征**：MFCC 用纯 JS 实现了 librosa 的完整管线（slaney mel 滤波器组、periodic Hann、DCT-II ortho、power→dB top_db=80），与 Python 逐位对拍（误差 < 1e-6），测试在 `scripts/test_parity.cjs` + `scripts/pyfeat.py`。STFT 用 radix-2 FFT 实现，与改造前的朴素 DFT 逐位对拍（**最大相对差 ~2e-12**，三段示例的预测概率完全一致）。
- **模型**：随机森林——在 440 样本的小数据上比深度网络更稳、且无需张量库。150 棵树导出为 320KB 的 JSON 决策树（gzip 后 ~50KB），浏览器直接执行。
- **速度**（2 秒音频实测，Chromium）：特征提取 ~40ms + YIN 音高 ~120ms + 森林推理 ~1ms，合计约 0.2 秒。改用 FFT 之前，朴素 DFT 让特征提取要 1.1–1.4 秒。
- **重采样一致性**：训练加载器（`train.py:_load_linear`）与网页 wav 解析路径使用完全相同的线性插值，保证训练/推理看到同一段波形。浏览器录音（webm/mp3）走 Web Audio 解码 + 同样的降采样。

## 复现

```bash
# 1. 数据：CatMeows (Zenodo, CC-BY-NC)
curl -L -o dataset.zip https://zenodo.org/api/records/4008297/files/dataset.zip/content
python3 -c "import zipfile; zipfile.ZipFile('dataset.zip').extractall('data/')"
# 解压后 wav 在 data/dataset/，文件名前缀 B/F/I 即类别

# 2. 训练
pip install librosa scikit-learn joblib numpy scipy
python scripts/train.py          # 输出 metrics.json + model.joblib

# 3. 导出为浏览器模型（可选深度：改脚本里的 max_depth）
python scripts/export_js.py      # 输出 forest.json

# 4. 特征对拍验证（Node >= 18）
node scripts/test_parity.cjs data/dataset/<某个wav>
```

## 分享图片

项目里有两张图，用途不同：

| 图片 | 用途 | 尺寸 |
|---|---|---|
| `public/og-image.png` | **站点分享预览**：把链接发到微信 / X / Telegram / Slack 时自动显示的大图 | 1200×630 |
| 结果卡片（运行时生成） | **单次分析结果**：点「📷 分享 / ⬇️ 保存图片」现画，含波形、三类概率、置信度、日期 | 1200×630（2x 超采样） |

- OG 图由 [`scripts/make_og_image.py`](scripts/make_og_image.py) 生成（需要一个 CJK 字体，用 `MEOWSCOPE_FONT_BOLD` / `MEOWSCOPE_FONT_REGULAR` 指定）：
  ```bash
  python3 scripts/make_og_image.py     # → public/og-image.png
  ```
- 结果卡片在浏览器里用 Canvas 现画，中英文随页面语言；优先调用系统分享，桌面端退化为复制到剪贴板（可直接粘进微信），再退化为下载 PNG。文件名形如 `meowscope-waiting_for_food-67-20261011.png`。

## 目录

```
public/              # 网站本体（单页应用，零依赖零构建）
  index.html         # 页面 + 交互 + 推理 UI
  features.js        # 纯 JS MFCC 特征提取（与 librosa 对拍）
  forest.json        # 随机森林模型（JSON 决策树，按需 fetch）
  metrics.json       # 训练指标，页面「诚实的数据」一节自动渲染
  demos.js           # 三段示例猫叫（base64，点击时才加载）
  i18n.js            # 中/英文案
  og-image.png       # 社交分享预览图（1200×630）
  favicon.svg / icon-192.png / icon-512.png
  llms.txt           # 给生成式引擎看的站点速览（GEO）
  robots.txt / sitemap.xml
  manifest.webmanifest / sw.js   # PWA，可离线安装
scripts/
  train.py           # 训练 + 按猫分组交叉验证
  export_js.py       # sklearn 森林 → JSON
  pyfeat.py          # Python 参考特征（对拍用）
  test_parity.cjs    # Node 对拍测试
  make_og_image.py   # 生成 OG 分享图
docs/
  share-card-example.png
src/                 # 与 public/ 同源的训练产物副本（forest.json / features.js / metrics.json）
```

> `src/` 与 `public/` 下的 `forest.json`、`features.js`、`metrics.json` 内容完全相同，是训练流水线输出的两份拷贝。改动时记得两边同步，或者干脆只保留 `public/` 一份。

## 给生成式引擎看的事实（GEO）

为了让 AI 搜索引擎引用到准确数字，站点提供了三样东西：

- `public/llms.txt`：站点速览 + 关键事实（准确率、数据集、技术栈、许可）+ 链接。
- `index.html` 内嵌 JSON-LD：一个 `WebApplication`（功能、免费、仓库、数据集）和一个 `FAQPage`（4 条问答，含准确率与隐私）。
- `robots.txt` 与 `sitemap.xml`。

切换页面语言时，`<title>`、`meta description`、`og:*`、`twitter:*` 会同步更新，分享预览不会出现「中文页面配英文描述」。

## 引用 / 致谢

- 数据集：**CatMeows** — Schork & Petry 等，Zenodo [4008297](https://zenodo.org/records/4008297)（CC-BY-NC，仅供研究）。
- 相关工作：Ntalampiras（猫叫情境识别）、Pandeya & Lee（猫叫分类）、Meow-Omni 1（多模态动物意图理解，2025）。

## 教它认识你家猫（个性化）

除了通用模型，页面支持**纯本地个性化训练**：三个情境各录 3 段自家猫叫声，浏览器内用 softmax 回归训练专属分类头（复用同一条 70 维特征管线），存 localStorage，零上传。训练后可一键在「通用 / 专属」模型间切换。

## License

代码 MIT。数据集版权归 CatMeows 作者（CC-BY-NC）。
