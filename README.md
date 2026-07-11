# 7.10-12Hackthon-Disc-Throwing-Program

## (1) 环境配置（Windows + WSL）

Panthera-HT SDK 当前推荐运行环境为 **Ubuntu 22.04**。Windows 用户推荐通过 **WSL2 + Ubuntu 22.04** 方式使用，不建议在纯 Windows 原生 Python 环境下直接安装和运行。

### 1. 安装 WSL2 + Ubuntu 22.04

在管理员 PowerShell 执行：

```powershell
wsl --install
```

如需指定 Ubuntu 22.04：

```powershell
wsl --install -d Ubuntu-22.04
```

安装完成后**必须重启**电脑。首次打开 Ubuntu 时设置用户名和密码，然后更新系统：

```bash
sudo apt update && sudo apt upgrade -y
```

### 2. 安装 Conda（推荐 Miniconda）

```bash
# 下载
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 安装（全部默认 + 输入 yes）
bash Miniconda3-latest-Linux-x86_64.sh

# 初始化
source ~/.bashrc

# 验证
conda --version
```

创建项目环境：

```bash
conda create -n panthera python=3.10 -y
conda activate panthera
```

### 3. 安装 Panthera SDK

```bash
# 克隆仓库
cd ~
git clone https://github.com/HighTorque-Robotics/Panthera-HT_SDK.git
cd Panthera-HT_SDK/panthera_python

# 安装依赖（Ubuntu 22 推荐）
pip install motor_whl/hightorque_robot-1.2.0-cp310-cp310-linux_x86_64.whl
pip install -r requirements.txt

# 验证安装
python -c "import hightorque_robot"
python -c "import pinocchio"
```

### 4. USB 串口透传（机械臂连接）

**Windows 侧安装 usbipd：**

```powershell
winget install --interactive --exact dorssel.usbipd-win
```

**查看设备：**

```powershell
usbipd list
```

找到目标设备（形如 `USB Serial Device ... Not shared`，如无法确定可插拔一下机械臂观察）。假设设备 busid 为 `4-2`：

```powershell
# 共享设备
usbipd bind --busid 4-2

# 挂载到 WSL
usbipd attach --wsl --busid 4-2
```

**WSL 中检查：**

```bash
ls /dev/ttyACM*
```

**权限设置：**

```bash
sudo chmod 666 /dev/ttyACM*
```

> ⚠️ **重要说明**
> - 一个机械臂可能对应多个 `/dev/ttyACM*`
> - SDK 会自动扫描 `/dev/ttyACM*`

### 5. 常见问题

**❌ 没接设备 → Segmentation fault**

- 原因：SDK 在扫描串口时未做空设备保护
- 解决：必须连接机械臂后再运行

### 6. VS Code 连接 WSL

1. 安装 VS Code
2. 安装插件：`Remote - WSL`
3. 连接 WSL：`Ctrl + Shift + P` → 输入 `WSL: Connect to WSL`

### 7. 运行示例

```bash
conda activate panthera
cd panthera_python/scripts
python 0_robot_get_state.py
```

## (2) 代码架构