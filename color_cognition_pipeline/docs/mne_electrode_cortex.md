# 跨被试 MNI 空间电极皮层映射 (右半脑专属视角)

根据您的要求，这里只展示了 **切半的右脑 (Right Hemisphere)**。皮层不透明度已降低至 30% (透明度增加)，以让您可以直接透过皮层看到内部电极的精确位置！

> [!NOTE]
> 🔴 **红色大圆点 (Color Selective)**: 直接在每个被试的 `ieegloc.xlsx` 文件的 `AAL3` 列中标记为 `Color_with_sti` 的电极位点。
> 🔵 **蓝色小圆点 (Target Area)**: 其他只通过 50ms 筛选的纯 `type1` 重点区电极。

---

## 📸 重新生成的皮层映射视图
这里是基于 `AAL3` 中 `Color_with_sti` 标签重新生成并定制的三个视角（向右滑动切换，**包含已修正的背面仰视 20 度图**）：

````carousel
![右半脑切半-背面仰视20度 (Posterior Up 20 View)](/home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/mne_cortex_posterior_up20.png)
<!-- slide -->
![右半脑切半-内侧面 (Medial View)](/home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/mne_cortex_rh_medial.png)
<!-- slide -->
![右半脑切半-外侧面 (Lateral View)](/home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/mne_cortex_rh_lateral.png)
````

---

## 📊 绘制在皮层上的所有电极表格 (共 33 个去重电极)

| Subject | Channel | Group | MNI_X | MNI_Y | MNI_Z | Desikan_Killiany | DKT | AAL3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test001 | B5 | type1 | 46.902 | -54.307 | 1.414 | White R | White R | Temporal_Mid R |
| test001 | C10 | type1 | 68.494 | -38.477 | 4.939 | Superior temporal R | Superior temporal R | Temporal_Mid R |
| test001 | C8 | type1 | 61.763 | -38.616 | 2.22 | Banks sts R | Superior temporal R | Temporal_Mid R |
| test001 | D4 | colorwithsti | 39.982 | -45.26 | -9.577 | White R | White R | Fusiform R |
| test001 | D5 | colorwithsti | 39.518 | -49.125 | -8.145 | White R | White R | Fusiform R |
| test001 | D6 | colorwithsti | 39.053 | -52.99 | -6.714 | White R | White R | Temporal_Inf R |
| test001 | F9 | type1 | 48.167 | -14.144 | -20.967 | White R | White R | Temporal_Inf R |
| test001 | G10 | type1 | 42.377 | -62.037 | -5.666 | White R | White R | Temporal_Inf R |
| test001 | G11 | type1 | 44.406 | -65.262 | -4.685 | White R | White R | Temporal_Inf R |
| test001 | G5 | colorwithsti | 32.234 | -45.909 | -10.575 | White R | White R | Fusiform R |
| test001 | G6 | colorwithsti | 34.263 | -49.135 | -9.593 | White R | White R | Fusiform R |
| test001 | G7 | colorwithsti | 36.291 | -52.36 | -8.612 | White R | White R | Fusiform R |
| test001 | H9 | type1 | 60.495 | -56.396 | 5.054 | Inferior parietal R | Inferior parietal R | Temporal_Mid R |
| test002 | A3 | type1 | 38.665 | -64.805 | -12.983 | Fusiform R | Fusiform R | Occipital_Inf R |
| test002 | B1 | type1 | 27.594 | -72.62 | -13.955 | Fusiform R | Fusiform R | Fusiform R |
| test002 | B2 | colorwithsti | 31.056 | -74.27 | -13.639 | Lateral occipital R | Lateral occipital R | Fusiform R |
| test002 | D1 | colorwithsti | 2.772 | -90.166 | 17.365 | Cuneus R | Cuneus R | V2 |
| test002 | D2 | colorwithsti | 5.584 | -92.901 | 17.572 | Cuneus R | Cuneus R | V2 |
| test002 | D3 | colorwithsti | 8.396 | -95.637 | 17.778 | Cuneus R | Cuneus R | V2 |
| test002 | F3 | type1 | 51.922 | -4.786 | -17.761 | Superior temporal R | Superior temporal R | Temporal_Mid R |
| test002 | F5 | type1 | 59.517 | -4.791 | -18.177 | Superior temporal R | Superior temporal R | Temporal_Mid R |
| test003 | A12 | type1 | 56.464 | -4.83 | -25.039 | White R | White R | Temporal_Mid R |
| test003 | A7 | type1 | 39.281 | -7.108 | -22.9 | White R | White R | Hippocampus R |
| test003 | G11 | type1 | 54.704 | -58.681 | 6.179 | White R | White R | Temporal_Mid R |
| test003 | G12 | type1 | 56.973 | -60.898 | 7.736 | White R | White R | Temporal_Mid R |
| test003 | G3 | colorwithsti | 36.122 | -40.433 | -5.631 | White R | White R | ParaHippocampal R |
| test003 | G4 | colorwithsti | 38.389 | -42.849 | -4.337 | White R | White R | ParaHippocampal R |
| test003 | H11 | type1/colorwithsti | 42.053 | -68.848 | 13.191 | White R | White R | Temporal_Mid R |
| test003 | H12 | type1/colorwithsti | 43.62 | -71.546 | 14.844 | Lateral occipital R | Lateral occipital R | Temporal_Mid R |
| test003 | H2 | colorwithsti | 27.237 | -44.012 | -1.26 | White R | White R | Hippocampus R |
| test003 | H3 | colorwithsti | 28.747 | -46.893 | 0.316 | Ventricle lat R | Ventricle lat R | Precuneus R |
| test003 | H4 | colorwithsti | 30.336 | -49.697 | 1.906 | Ventricle lat R | Ventricle lat R | Precuneus R |
| test003 | H5 | colorwithsti | 31.981 | -52.457 | 3.55 | White R | White R | Calcarine R |
