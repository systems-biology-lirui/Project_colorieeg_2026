0.信号的预处理：
erp与subbandhighgamma的获得
    （1）0.1-200hz的滤波，50，100，150的陷波，
    （2）重参考
    （3）坏段剔除（先暂时取消）
    （4）试次对齐
    （5）提取特征
    
1.电极的筛选
参考文献：
    1.color-biased regions of the ventral visual pathway lie between face- and place-selective regions in humans, as in macaques
    猕猴的从枕叶颞叶交界到颞极差不多有5个分散的patch，人都是在颞下回的三个patch
    2.A shared code for perceiving and imagining objects in human ventral1
    这也是一个颞下回的电极
所在的ROI决定：
    将区域的限制放大到如下范围，重新进行寻找：
    枕叶：Calcarine (早期视觉), Occipital_Inf, Occipital_Mid, Lingual (颜色与早期特征加工)
    颞叶后/下部：Fusiform, Temporal_Inf (高级视觉、颜色斑块与形状整合)
    颞叶前/上部：Temporal_Mid, Temporal_Pole (语义知识与记忆匹配)
四种策略：
    1.混合类别的color/gray显著在一个100-400ms窗口的平均值上显著
    2.在混合类别下存在50ms以上窗口的color-gray差异显著
    3.在单一类别下，100-400ms平均值显著
    4.存在某一类别的50ms以上窗口的colorgray差异显著。
    ![image alt text](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/electrode_selection_comparison.png)
    当使用最灵敏且动态的**策略 4（单一条件连续 50ms 显著 + 限扩大靶区）**时，各被试筛选到的具体电极名单如下：

    1. 被试 test001
    ERP（共 7 个，新增 G11、H1，移除了 F5 等非靶区电极）： B5, C8, C10, F9, G11, H1, H9
    High Gamma（共 7 个，保持稳定）： B5, C10, E8, G4, H6, H9, H10
    2. 被试 test002
    ERP（共 6 个，新增 C7、C8、F3，移除了 D 系列顶叶电极）： A3, B1, C7, C8, F3, F5
    High Gamma（共 15 个，显著召回了位于枕中回/枕下回的电极如 B6、C9 等）： A2, A3, A4, A8, A9, B6, C9, C10, F4, F6, G5, G6, G7, H5, H8
    3. 被试 test003
    ERP（共 3 个）： A12, G11, G12
    High Gamma（共 5 个）： D9, G9, G10, G11, G14
计算1：需要重新统计一下获得的电极并确定。
    /home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/interactive_brain/erp_3d_brain.html
计算1_1加入了附近的电极的选取（暂时不对他们进行分析）

计算2：重新绘制nilearn的电极分布图，colorpatch的mask实在不行就使用半透明的球来表示，然后使用侧视图
计算3：对找出的电极绘制颜色选择性差异，从后到前（比较难绷，好像刚好和世界做出来的结果是相反的，当然这个可能不是主要的结果，有可能是窗口的原因）

2.颜色知识的decoding
目前使用的是策略4进行，分别在erp和highgamma上进行
单电极的信号差异：
计算3：单电极的不同颜色信号的灰色图片差异，需要逐个时间点的显著性差异和100-400ms窗口平均的差异显著性


单电极的decoding：
    在test001上有一个非常好的电极，G11（temporal_inf），test003的三个电极的效果也不错
    ![test001_single_electrode_decoding](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/test001/decoding/single_electrode/single_electrode_decoding.png)
    ![test002_single_electrode_decoding](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/test002/decoding/single_electrode/single_electrode_decoding.png)
    ![test003_single_electrode_decoding](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/test003/decoding/single_electrode/single_electrode_decoding.png)
这一步是在多电极的decoding之后进行的，目的其实是想看看三个被试的峰值时间有明显的先后差异是不是由于每个被试选取的电极位于整个信号通路的前后导致的。
计算4：需要对每个电极进行decoding，并找出现显著的时间点的峰值

多电极的decoding：
    erp的效果很不错，而且使用glmm进行显著性校验的效果也不错
    ![alt text](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/group_average_erp_strategy4_decoding_significant.png)
    ![alt text](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/group_glmm_erp_strategy4_decoding.png)
计算5：要看一下这里的glmm有没有用对,其次，对于decoding进行的优化内容有哪些

cluster的多电极的decoding
计算6：根据前后分为不同的电极组合来分开decoding，看时间上是否有先后，目前的一个小猜测是枕叶上的可能不一定能解码出来。这样test002就会慢一些。

3.记忆颜色和表达颜色
使用了两种方式，
    纯色色块训练解码记忆颜色
        但是由于颜色解码本身效果不太好，所以在记忆颜色上的decoding效果也不太好
    记忆颜色训练解码纯色色块
        效果也一般
    如果二者都不行，可能是erp和highgamma之间的某种连接？这个还得思考一下。
    计算7：纯色色块的影响应该很小，为什么效果很差呢，erp和highgamma都看一下，看看能不能进行一下代码优化什么的
    计算8：对于这一cross解码，使用世间泛化的decoding

4.真假的decoding
挑选出color patch和颞极的位点做
    单电极的信号差异
    单电极的decoding
    多电极的decoding

5.别的位点电极的认知颜色的，比如amygdala，hippocampus等
    





额外可以做的：
1.物体特异性的colorpatch和非特异性的colorpatch，是否有分布的差异，即存在主要角色，但是别的角色可以进一步的提高decoding的效果。
2.颜色的decoding时间

当下存在的问题：
1.多种用于寻找color patch电极的策略，找到的电极不一致
1）color-gray
2）电刺激
3）highgamma和erp

step2_1
将这些找到的差异电极定义为memory_color电极
[memory_color](file;file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/result/select_channel/memory_color) 这下面的两个子文件夹中的电极，我希望能够将电极分在不同的被试子文件夹下面


对于两种电极的组合(colorwithsti/type1）都要进行一次。
然后使用erp使用task3的红绿色训练，对于task2中的西瓜和猕猴桃进行测试，做一个时间泛化的decoding。
最后使用task2进行真假的decoding，也是采用配对的方式，除了使用我们找出的两个类比的电极进行，使用temporal_pole的电极进行。

step3_1
为什么[erp_group_decoding_estp.png](file;file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/result/select_channel/decoding/single_electrode/erp_group_decoding_estp.png) 中的合并曲线decoding的效果这么奇怪，正确率像是有一个阶梯变化，这是为什么
