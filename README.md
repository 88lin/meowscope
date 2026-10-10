# 喵语通 MeowScope 🐱

**在浏览器里听懂猫在说什么。** 录音或上传猫叫，本地 AI 实时判断它在哪种情境下发声（被刷毛 / 等饭 / 被隔离）——所有计算都在你的浏览器完成，**音频不会上传到任何服务器**。

在线体验：https://88lin.github.io/meowscope/ （或直接打开 `public/index.html` 本地跑）

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
| 按猫分组 5 折交叉验证 | **59.4%** |
| 留一猫验证 (LOSO) | **56.6%** |
| 随机猜 | 33.3% |

细节见 [`src/metrics.json`](src/metrics.json)。很多宠物翻译产品宣传 90%+，那是"同一只猫的录音再喂回去"的成绩——对你家第一次叫的猫没有参考价值。我们选择公布严格数字。

## 工作原理

```
音频 → 22.05kHz 单声道 → 20 维 MFCC + 一阶差分 + 谱统计（67 维特征）
     → 随机森林 150 棵（depth 6）→ 三类概率
```

- **特征**：MFCC 用纯 JS 实现了 librosa 的完整管线（slaney mel 滤波器组、periodic Hann、DCT-II ortho、power→dB top_db=80），与 Python 逐位对拍（误差 < 1e-6），测试在 `scripts/test_parity.cjs` + `scripts/pyfeat.py`。
- **模型**：随机森林——在 440 样本的小数据上比深度网络更稳、且无需张量库。150 棵树导出为 266KB 的 JSON 决策树，浏览器直接执行，单次推理 < 500ms。
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

网页本体在 [`public/`](public/)，纯静态，任何静态托管都能跑（`index.html` + 3 个资源文件）。

## 目录

```
public/          # 网站本体（单页应用，零依赖零构建）
  index.html     # 页面 + 交互 + 推理 UI
  features.js    # 纯 JS MFCC 特征提取（与 librosa 对拍）
  forest.json    # 随机森林模型（JSON 决策树）
  metrics.json   # 训练指标，页面"诚实的数据"一节自动渲染
scripts/
  train.py       # 训练 + 按猫分组交叉验证
  export_js.py   # sklearn 森林 → JSON
  pyfeat.py      # Python 参考特征（对拍用）
  test_parity.cjs# Node 对拍测试
```

## 引用 / 致谢

- 数据集：**CatMeows** — Schork & Petry 等，Zenodo [4008297](https://zenodo.org/records/4008297)（CC-BY-NC，仅供研究）。
- 相关工作：Ntalampiras（猫叫情境识别）、Pandeya & Lee（猫叫分类）、Meow-Omni 1（多模态动物意图理解，2025）。

## License

代码 MIT。数据集版权归 CatMeows 作者（CC-BY-NC）。
