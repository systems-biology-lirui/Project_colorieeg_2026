import re
import pandas as pd

def get_roi_category(label):
    """
    根据解剖标签判断其所属的 ROI 大类。
    返回: '枕叶', '颞叶后/下部', '颞叶前/上部' 或 None
    """
    if not isinstance(label, str) or label.strip() == '':
        return None
    label_lower = label.lower()
    
    # 枕叶: Calcarine, Occipital_Inf, Occipital_Mid, Lingual
    if any(kw in label_lower for kw in ['calcarine', 'occipital_inf', 'occipital_mid', 'lingual']):
        return '枕叶'
    # 颞叶后/下部: Fusiform, Temporal_Inf
    elif any(kw in label_lower for kw in ['fusiform', 'temporal_inf']):
        return '颞叶后/下部'
    # 颞叶前/上部: Temporal_Mid, Temporal_Pole
    elif any(kw in label_lower for kw in ['temporal_mid', 'temporal_pole']):
        return '颞叶前/上部'
    return None

def is_neighbor_valid_label(label):
    """
    判断邻近电极的解剖标注是否属于 unknown, N/A 或旁海马
    """
    if pd.isna(label) or not isinstance(label, str) or label.strip() == '':
        return True
    label_lower = label.lower().strip()
    if 'unknown' in label_lower or 'n/a' in label_lower or label_lower == 'nan':
        return True
    if 'parahippocampal' in label_lower or 'parahippocampus' in label_lower:
        return True
    return False

def parse_channel_name(ch_name):
    """
    解析例如 G11 -> ('G', 11), FP2 -> ('FP', 2)
    """
    if not isinstance(ch_name, str):
        return None, None
    match = re.match(r'^([a-zA-Z]+)(\d+)$', ch_name.strip())
    if match:
        prefix = match.group(1).upper()
        num = int(match.group(2))
        return prefix, num
    return None, None

def find_neighbors(target_ch, all_channels):
    """
    在所有可用通道中，寻找 target_ch 的物理邻近电极（同一轴且序号相差为 +/- 1）
    """
    target_prefix, target_num = parse_channel_name(target_ch)
    if target_prefix is None:
        return []
    
    neighbors = []
    for ch in all_channels:
        if ch == target_ch:
            continue
        pref, num = parse_channel_name(ch)
        if pref == target_prefix and abs(num - target_num) == 1:
            neighbors.append(ch)
    return neighbors
