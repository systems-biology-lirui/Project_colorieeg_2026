---
name: wsl2_freesurfer_vcxsrv_setup
description: Comprehensive technical guide for installing Ubuntu 22.04 LTS on WSL2, deploying FreeSurfer 7.4.1, resolving Freeview 3D OpenGL rendering issues via VcXsrv X Server, and configuring auto-start workflows.
---

# WSL2 环境下 FreeSurfer 7.4.1 部署与 VcXsrv 3D 渲染配置指南

本指南记录了从 WSL2 环境构建、Ubuntu 22.04 LTS 部署、FreeSurfer 7.4.1 自动化安装、许可文件配置，到解决 WSLg 离屏缩死并接入 VcXsrv (XLaunch) 3D OpenGL 图形渲染的全流程及开机自动运行配置。

---

## 1. WSL2 与 Ubuntu 22.04 LTS 基础设施部署

### 1.1 安装与环境验证
```powershell
# 1. 检查 WSL 状态及在线分发版
wsl --list --online

# 2. 安装 Ubuntu 22.04 LTS（FreeSurfer 7.4.1 官方推荐版本）
wsl --install -d Ubuntu-22.04
```

### 1.2 Linux 基础依赖包安装
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y bc tar wget curl libgomp1 libqt5gui5 libqt5widgets5 libqt5core5a libgfortran5 libxss1 libxft2 tcsh
```

---

## 2. FreeSurfer 7.4.1 下载与部署

### 2.1 软件本体安装 (.deb)
```bash
# 下载官方 Ubuntu 22.04 编译包 (6.5 GB)
wget https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/7.4.1/freesurfer_ubuntu22-7.4.1_amd64.deb -O /tmp/freesurfer.deb

# 执行安装 (默认路径 /usr/local/freesurfer/7.4.1)
sudo apt install -y /tmp/freesurfer.deb
rm -f /tmp/freesurfer.deb
```

### 2.2 环境变量注入 (`~/.bashrc`)
在 `~/.bashrc` 尾部添加（务必防止变量提前展开）：
```bash
export FREESURFER_HOME=/usr/local/freesurfer/7.4.1
source $FREESURFER_HOME/SetUpFreeSurfer.sh
export DISPLAY=$(ip route show | grep default | awk '{print $3}'):0.0
```

### 2.3 许可文件配置 (`license.txt`)
将申请获取的密钥保存至 `/usr/local/freesurfer/7.4.1/license.txt`：
```bash
sudo cat << 'EOF' > /usr/local/freesurfer/7.4.1/license.txt
your_email@domain.com
12345
 *CKgLcVovhAb6
 FSwSQlnHVX/Us
 tvoW9NQQ5uqBuOGRhFcgk87ugCiFzmx7tmr02u3Mb68=
EOF
```

---

## 3. Freeview 3D OpenGL 渲染与 VcXsrv 踩坑排查

### 3.1 现象分析（WSLg 任务栏无法点开）
在 Windows 11 默认的 WSLg 环境下单独运行 `freeview` 时，系统会生成一个 0 像素或离屏的模态窗口并自动最小化至任务栏，导致点击无响应。

### 3.2 解决方案：接入 VcXsrv Windows X Server
1. 安装 VcXsrv (XLaunch)：[SourceForge 下载](https://sourceforge.net/projects/vcxsrv/)
2. 运行 XLaunch：
   - Select `Multiple windows`, Display number = `0`.
   - Select `Start no client`.
   - 勾选 **`Disable access control`**（禁用访问控制）。

### 3.3 踩坑 1：报错 "Cannot establish any listening sockets - Make sure an X server isn't already running"
- **原因**：VcXsrv (`vcxsrv.exe`) 已经在 Windows 后台或右下角系统托盘中正常运行，重复双击打开时端口 6000 被占用。
- **解法**：点击弹窗中的【确定】关闭提示即可，无需重复启动。可以在任务管理器或右下角托盘图标确认其后台状态。

### 3.4 自动化开机静默启动配置（无需每次手动打开）
1. 在 XLaunch 完成界面点击 **"Save configuration"**，将配置文件另存为 `config.xlaunch`。
2. 按 `Win + R` 输入 `shell:startup` 打开 Windows 开机启动文件夹。
3. 将 `config.xlaunch` 文件放入该文件夹中。
4. **效果**：Windows 每次开机后，VcXsrv 将自动在后台静默运行，无需每次手动配置或点击 XLaunch。

---

## 4. 常用运行与测试指令

```bash
# 检查核心版本与许可验证
recon-all --version
mri_convert --version

# 启动 3D 脑影像可视化界面 (带示例数据)
freeview -v $FREESURFER_HOME/subjects/bert/mri/orig.mgz &
```
