# Guizang PPT Skill (网页 PPT) 使用说明书

> **安装路径**：[guizang-ppt-skill](file:///home/lirui/.gemini/config/skills/guizang-ppt-skill/)
> **适用环境**：Antigravity IDE Agent / 任何具备 shell 读写和运行环境的 AI 助手
> **输出格式**：单文件 HTML 横向翻页 PPT（内嵌 WebGL 背景、Lucide 图标、Motion One 动效）

---

## 📋 核心特色

该技能用于生成**单文件 HTML 格式的学术与演讲 PPT**、PPT 配图和多平台封面。它不依赖外部重型构建框架，浏览器双击可直接离线演示。内置两套极高美学水准的视觉系统：

1. **Style A：电子杂志 × 电子墨水（暖色调）**
   * **设计美学**：模仿 *Monocle* 杂志的排版风格。等高线/流体 WebGL 炫彩背景，衬线体标题（Playfair Display & Noto Serif SC）与非衬线正文对比，配有等宽字体元数据。
   * **适用场景**：人文观察、观点表达、学术汇报、故事分享。
2. **Style B：瑞士国际主义（Swiss Style - 高反差）**
   * **设计美学**：网格至上，全程无衬线体（Inter & Helvetica & Noto Sans SC），极致字号大小对比，直角纯色（无圆角、无渐变、无阴影）。使用单一高饱和度功能高亮色（克莱因蓝 IKB、柠檬黄、柠檬绿、安全橙四选一）。
   * **适用场景**：科技产品发布、数据图表汇报、严谨方法论、工程分析。

---

## 🔧 快速启动：如何让 AI 帮您制作 PPT？

由于该技能已在您的 Antigravity 技能库中部署成功，您可以直接在对话框中**用自然语言向我（AI）下达指令**。以下是几个推荐的指令模板：

### 模版 1：文献/长文转 PPT（推荐 Style B 瑞士风）
> “帮我基于刚才导出的 [赵明慧_毕业论文笔记.md](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/赵明慧_毕业论文笔记.pdf) 制作一份学术汇报 PPT。使用 **Style B 瑞士风**（克莱因蓝配色），控制在 8 页左右，结构要包含封面、三层架构、以及用于性能对比的双列对比版式。”

### 模版 2：个人学术观点分享（推荐 Style A 杂志风）
> “帮我把我的 [周记汇总与科研工作进展.md](file:///home/lirui/note/lr_project/周记/周记汇总与科研工作进展.md) 制作一份演讲 Slides。使用 **Style A 电子杂志风**（默认墨水经典主题），要求有大金句展示页和两列流水线（Pipeline）版式。”

### 模版 3：提取论文生成封面
> “基于我的项目大纲，生成一张公众号 21:9 比例的封面大图，要符合瑞士国际主义风格的字号和锚点色设计。”

---

## 🏛️ 标准工作流 (Workflow)

当您召唤 AI 启动 PPT 制作时，AI 会在后台自动执行以下标准工作流：

```
1. 需求澄清 (对齐7问) ➔ 2. 拷贝模板 ➔ 3. 规划主题节奏与版式 ➔ 4. 填充内容 ➔ 5. 校验与本地预览
```

### 1. 需求澄清（7问清单）
如果您只给出了一个模糊的主题，AI 会主动与您确认以下 7 个问题：
1. **风格选择**：Style A (杂志风) 还是 Style B (瑞士风)？
2. **受众与场景**：用于组会、学术发表还是 Demo Show？
3. **分享时长**：以此估算 PPT 页数（一般 15分钟约 10 页）。
4. **原始素材**：是否提供特定文献、代码文档或大纲？
5. **图片/截图处理**：是否有配图，如何排版？
6. **主题色预设**：
   * 杂志风 5 套：*墨水经典、靛蓝瓷、森林墨、牛皮纸、沙丘*。
   * 瑞士风 4 套：*克莱因蓝 (IKB)、柠檬黄、柠檬绿、安全橙*。
7. **硬性约束**：是否有不可遗漏的数据或公式？

### 2. 拷贝模板与选色
AI 会在您的项目目录下自动创建 `images/` 文件夹和 `index.html` 目标文件，并拷贝对应的模板文件：
* 杂志风种子：[template.html](file:///home/lirui/.gemini/config/skills/guizang-ppt-skill/assets/template.html)
* 瑞士风种子：[template-swiss.html](file:///home/lirui/.gemini/config/skills/guizang-ppt-skill/assets/template-swiss.html)

### 3. 主题节奏规划
为避免视觉疲劳，页面背景色必须冷暖/深浅交替（强制不允许连续 3 页采用同种底色）：
* `hero dark` (封面/过渡幕封) ➔ `light` (数据/正文) ➔ `dark` (金句/大引用) ➔ `light` (图文)

### 4. 锁定版式填充 (Style B 瑞士风核心)
瑞士风启用 **Swiss Locked Mode**（严格锁版限制）。每一页正文必须被标记为原始的 22 种特定版式之一（写有 `data-layout="Sxx"` 属性）：
* **S01** (Cover) / **S02** (时间线) / **S03** (左右分栏论点) / **S06** (KPI塔柱数据) / **S08** (双列对比) / **S11** (流程图) / **S14** (闭环流图) / **S17** (系统架构图) / **S22** (21:9大图英雄页) 等。
* **字号分档与阶梯**：中文标题严禁粗暴照搬英文大字号（以防重叠撑破），中文 1 行 ≤ 8 字使用 `6.4vw`，2 行使用 `5.2vw`；正文小字绝对不低于 `16px` 以保障投屏演示清晰度。

### 5. 质量校验与预览
* **代码静态校验**：脚本会运行 [validate-swiss-deck.mjs](file:///home/lirui/.gemini/config/skills/guizang-ppt-skill/scripts/validate-swiss-deck.mjs)，检查是否有未登记的版式、溢底问题、错位圆角或在 SVG 内部硬编码文本等 P0 级设计事故。
* **本地打开**：直接在您的 Linux 浏览器中双击打开项目中的 `index.html` 文件，即可完美交互。键盘 `←` `→` 键翻页，按 `ESC` 键可查看全局幻灯片索引大图，按 `B` 键可直接进入**低功耗静态模式**（关闭 RAF Canvas 背景动画并瞬间呈现所有进入动效，防止旧设备风扇狂飙）。

---

## 📂 资源清单导览

如果您想深入修改样式或研究其实现原理，可参考以下目录资产：
* **组件设计字典**：[components.md](file:///home/lirui/.gemini/config/skills/guizang-ppt-skill/references/components.md) (按钮、图表线、卡片标记样式定义)
* **杂志风布局集**：[layouts.md](file:///home/lirui/.gemini/config/skills/guizang-ppt-skill/references/layouts.md) (杂志风 10 种页面代码原型)
* **瑞士风版式锁**：[swiss-layout-lock.md](file:///home/lirui/.gemini/config/skills/guizang-ppt-skill/references/swiss-layout-lock.md) (22 个锁定版式的参数规范)
* **瑞士风布局集**：[layouts-swiss.md](file:///home/lirui/.gemini/config/skills/guizang-ppt-skill/references/layouts-swiss.md) (22 种瑞士风页面代码原型)
* **地图连线组件**：[swiss-map-component.md](file:///home/lirui/.gemini/config/skills/guizang-ppt-skill/references/swiss-map-component.md) (用于 S08 的点线卡片地图组件)
* **AI 绘图与配图提示词**：[image-prompts.md](file:///home/lirui/.gemini/config/skills/guizang-ppt-skill/references/image-prompts.md) (配合 Midjourney 制作插图的标准范式)
* **质量自检红线**：[checklist.md](file:///home/lirui/.gemini/config/skills/guizang-ppt-skill/references/checklist.md) (P0/P1/P2 级体验把关清单)
