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

### 整体目录结构

```
Frisbee Throw/PythonScripts/
├── panthera_cpp/              ← C++ 底层代码（高级用户用，负责和电机 CAN 总线通信）
├── panthera_python/           ← Python 控制代码（日常开发主要在这里）
├── .gitignore
├── LICENSE
└── README.md
```

### panthera_python/ — Python 控制目录（重点）

```
panthera_python/
├── images/                    ← 文档图片
├── motor_whl/                 ← 预编译电机驱动包（.whl，按 Python 版本/架构区分，pip 直接安装）
├── Panthera-HT_description/   ← 机械臂 URDF 3D 模型，供运动学/动力学计算，SDK 自动加载
├── robot_param/                ← 机械臂配置文件（关节限位、最大力矩、电机 ID、CAN 端口等）
├── scripts/                   ← 本次 hackathon 项目脚本主目录（见下一节展开）
├── src/                       ← C++ ↔ Python 的 pybind11 绑定源码，从源码编译时才需要关心
├── .gitignore
├── CMakeLists.txt             ← 编译配置（从源码安装时用）
├── pyproject.toml             ← Python 项目打包配置
├── README.md                  ← Python SDK 详细使用说明
├── requirements.txt           ← Python 依赖列表（pyyaml、pinocchio、scipy 等）
└── setup.py                   ← Python 包安装脚本
```

### scripts/ — 项目脚本目录

```
scripts/
├── __pycache__/                        ← Python 自动生成的字节码缓存，可忽略
├── motor_example/                      ← 底层电机控制示例（位置/速度/力矩/电压/电流/MIT 五参数控制）
├── Panthera_lib/                       ← 核心高层控制库（运动学、动力学、夹爪控制、轨迹录制/回放工具）
│
├── 0_robot_get_state.py                ← 获取当前电机状态，可用在配完环境连接机械臂后运行检验是否连接成功
├── 0_robot_set_zero.py                 ← 设置机械臂零位置
├── 1_record.py                         ← 示教过程记录机械臂轨迹，生成一份json文件，需要在replay类文件中读取。本次record示教都不记录夹爪运动
├── 1_record_PartiallyFixed.py          ← 示教过程，与上一个文件的不同之处在于按空格可以限制住除了肩关节和手腕（电机1和5）其它部分的运动，再按空格解除。更适配扔飞盘
├── 2_replay_SetGripper.py              ← 示教复现，用键盘“前进/后退”键手动驱动机械臂运动。本程序的核心功能是肉眼观看慢放的示教复现过程，人工确定夹爪释放的关键帧
├── 2_replay.py                         ← 示教复现过程，复现record类代码时走过的运动路径。可以自行调整加速倍数、夹爪释放关键帧。同时需要调整json文件名
├── 3_hand_eye_calibration.py           ← 手眼标定，用于明确摄像机坐标系与机械臂末端坐标系坐标变换矩阵，实现机械臂视觉
├── arm_teach_20260711_005433.jsonl     ← 本次demo所用示教数据记录文件
├── hand_eye_calibration.json           ← 本次所用手眼标定的数据文件
├── red_disc_detection.py               ← 识别红色飞盘
├── red_frisbee_FollowGrasp.py          ← 跟随并抓取红色飞盘
└── Demo.py                             ← Demo整合代码
```

## （3）硬件要求

**高擎Panthera-HT机械臂**
**Intel RealSenseD405深度摄像机**
**合适形状的夹爪**：本次所用为自主设计3D打印的夹爪：[上夹爪设计文件](./redesign_gripper/redesign_upper_gripper.3mf)、[下夹爪设计文件](./redesign_gripper/redesign_lower_gripper.3mf)。高擎Panthera-HT机械臂有自带夹爪，如需复现也可以自己设计
**飞盘**：可根据测试情况自行选择。本次采用的是红色儿童软盘，抓取和扔出的效果相对较好。本次测试也尝试了175g Ultimate Frisbee成人比赛用盘，相对效果不如儿童软盘，但也可以以一定自转速度扔出。

## (4) 资料参考

[高擎官方Panthera-HT SDK](https://github.com/HighTorque-Robotics/Panthera-HT_SDK)

[Panthera_HT_SDK_Extensions](https://github.com/HighTorque-Robotics/Panthera_HT_SDK_Extensions)

本项目特别致谢高擎动力的支持。用到的绝大部分代码都参考了高擎SDK的代码中的思路，在此为感兴趣的同学提供相关代码，SDK学习/调用路径。