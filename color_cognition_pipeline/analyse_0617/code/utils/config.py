import os

BASE_DIR = '/home/lirui/liulab_project/ieeg/Project_colorieeg_2026'
PIPELINE_DIR = os.path.join(BASE_DIR, 'color_cognition_pipeline')
ANALYSE_DIR = os.path.join(PIPELINE_DIR, 'analyse_0617')
FEATURE_DIR = os.path.join(ANALYSE_DIR, 'feature')
DOC_DIR = os.path.join(ANALYSE_DIR, 'doc')
RESULT_DIR = os.path.join(ANALYSE_DIR, 'result')

SUBJECTS = ['test001', 'test002', 'test003']
TASKS = [1, 2, 3]

# Task 2 灰色水果触发器 (用于记忆颜色解码)
R1_TRIGS = ['Trigger-In:123'] # 灰色草莓
R2_TRIGS = ['Trigger-In:133'] # 灰色西瓜
G1_TRIGS = ['Trigger-In:103'] # 灰色卷心菜
G2_TRIGS = ['Trigger-In:113'] # 灰色猕猴桃

# Task 2 真假颜色触发器
STRAWBERRY_TRIGS = ['Trigger-In:121', 'Trigger-In:122'] # 真红, 假绿
WATERMELON_TRIGS = ['Trigger-In:131', 'Trigger-In:132'] # 真红, 假绿
CABBAGE_TRIGS = ['Trigger-In:102', 'Trigger-In:101']    # 假红, 真绿
KIWI_TRIGS = ['Trigger-In:112', 'Trigger-In:111']       # 假红, 真绿

# 区分红绿标签 (红=0, 绿=1)
RED_LABELS = ['Trigger-In:121', 'Trigger-In:131', 'Trigger-In:102', 'Trigger-In:112']
GREEN_LABELS = ['Trigger-In:122', 'Trigger-In:132', 'Trigger-In:101', 'Trigger-In:111']

# Task 3 纯色触发器
RED_COLOR_TRIGS = ['Trigger-In:51']
GREEN_COLOR_TRIGS = ['Trigger-In:54']
