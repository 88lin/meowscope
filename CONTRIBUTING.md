# 参与 MeowScope

有三条参与路径，都欢迎：**贡献代码**、**贡献数据**、**报 bug / 提建议**。

## 1. 贡献代码

站点本体是零依赖零构建的纯静态页面（`public/`），任何改动直接改文件即可，不需要 npm install。

```bash
git clone https://github.com/88lin/meowscope.git
cd meowscope
# 开发期校验才需要依赖（playwright）
npm install
(cd public && python3 -m http.server 8931 &) ; sleep 2
npm run check        # assets + ui 两个闸门
python3 scripts/parity_check.py   # 特征对拍（需要 numpy/librosa）
```

提交前必跑（CI 同样会跑，不过不过不让合）：

| 闸门 | 命令 | 查什么 |
|---|---|---|
| 特征对拍 | `python3 scripts/parity_check.py` | JS 特征提取 ↔ Python librosa 逐位一致 |
| 资源自洽 | `node scripts/check_assets.cjs` | 引用文件存在、中英文案 key 一致 |
| UI 渲染 | `node scripts/ui_check.cjs` | 真实浏览器里量：进度条/频谱图画没画出来、移动端溢出、对比度 |

约定：
- 训练/推理一致性是红线：改 `public/features.js` 必须重新对拍
- 新增界面文案必须中英文成对加进 `public/i18n.js`（assets 闸门会查 key 集合）
- Commit 用 Conventional Commits（`feat:` / `fix:` / `docs:` …）
- 数据集是 CC BY-NC：代码 MIT，但不要把它商用化

## 2. 贡献数据（最缺的一环）

模型只学过 21 只猫（缅因猫、欧洲短毛猫），对狸花猫、橘猫、布偶……完全没验证过。
**贡献数据比贡献代码更有价值。**

1. 打开 https://88lin.github.io/meowscope/ ，进「帮它听更多猫的叫声」一节
2. 选情境、录 1–5 秒（可多录几段，最多 5 秒/段），可填品种和年龄
3. 点「打包下载」得到一个 zip（音频 + 标签 + manifest，**本地生成，不上传**）
4. 开一个 [data contribution issue](../../issues/new?template=data_contribution.yml)，把 zip 拖进评论区

贡献的音频以 **CC0-1.0** 开放。请不要录进清晰的第三方人声对话。

## 3. 报 bug / 提建议

直接开 issue。复现步骤、浏览器版本、期望行为三件套写清楚即可。
涉及准确率的反馈尤其欢迎——请附上音频（同样走数据贡献流程）和当时的判定截图。

## 行为准则

友善、就事论事、不贴未标注来源的「准确率」数字。这个项目存在的理由就是把猫叫分类的
真实水位诚实地摆出来，请一起维护它。
