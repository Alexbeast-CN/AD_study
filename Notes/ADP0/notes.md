#! https://zhuanlan.zhihu.com/p/532507009
# ADP0. Carla 初体验

> ADP 是 Autonomous Driving Practise 自动驾驶实践的缩写。是配合着 AD 理论课程的应用层。这里我们选择 Carla 作为自动驾驶模拟器，因为目前来说 (22 年)，Carla 是学界和工业届主流的自动驾驶模拟器。

## 1. 安装 Carla

### 1.1 安装包安装

安装 Carla 分为两种方式，一种是直接下载安装包，另外一种是在本地编译。如果你不需要修改 Carla 程序本身，或者自己创建地图的话，建议使用 Carla 的安装包，安装过程非常简单，使用起来也很方便。若是要安装包的话，首先需要安装[虚幻引擎4](https://www.unrealengine.com/zh-CN)，在安装之前先将自己的 Github 账号与 Epic 进行绑定以加入 Epic 组织。具体的安装过程可以看这个贴子 [如何在 Ubuntu 上安装 UE4](https://www.addictivetips.com/ubuntu-linux-tips/unreal-engine-4-ubuntu/)，或者看小飞哥的 [教程](https://zhuanlan.zhihu.com/p/338927297) （该教程与后面编译 Carla 的教程是同一个）。之后只需要下载 Carla 的安装包，然后解压即可。以 Carla 0.9.10.1 为例：

```
#!/usr/bin/env bash

# Download and install CARLA
mkdir carla
cd carla
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.10.1.tar.gz # Need to change
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/AdditionalMaps_0.9.10.1.tar.gz # Need to change
tar -xf CARLA_0.9.10.1.tar.gz
tar -xf AdditionalMaps_0.9.10.1.tar.gz
rm CARLA_0.9.10.1.tar.gz
rm AdditionalMaps_0.9.10.1.tar.gz
cd ..
```

> 注意这里的下载地址 `wget https://` 那里的 url 需要更改一下。由于我在欧洲，所以默认给我了一个欧洲的下载地址，其他地区的同学需要自行在 `Carla Release` 处获取自己所需的 Carla 下载地址。

### 1.2 编译安装

但是如果你是 Carla 的重度使用者，如需要修改 Carla 地图，甚至修改 Carla 的程序的话，则推荐本地编译 Carla。具体的编译的过程 [@叶小飞](https://www.zhihu.com/people/xie-xiao-fei-78-24) 已经写了非常详细的过程了，大家跟着 [史上最全Carla教程 |（二）Carla安装](https://zhuanlan.zhihu.com/p/338927297) 进行安装就可以了。安装过程可能会遇到很多的问题，这里我对一些可能遇到的问题做简短的说明：

1. CARLA 版本选择。这里我们选择的是 CARLA 0.9.10.1，选择这个版本的主要原因是 [Carla Challenge](https://carlachallenge.org/) 所选用的版本是 0.9.10.1。因此绝大多数的学术研究也是使用该版本的 Carla。
2. Ubuntu 版本选择。与该版本 CARLA 对应的最佳选择是 Ubuntu 18.04。
3. 在安装 UE4 之前需要先下载显卡驱动，Linux 并没有自带驱动。在 [Nvidia 的网站](https://www.nvidia.com/content/DriverDownload-March2009/confirmation.php?url=/XFree86/Linux-x86_64/470.129.06/NVIDIA-Linux-x86_64-470.129.06.run&lang=us&type=TITAN) 上有提供 Vulkan 驱动的安装包。
4. 在安装 CARLA 之前请先卸载 Anaconda，否在在 make 的过程中会出现奇奇怪怪的问题导致失败。
5. 在 make 过程中报错请查看小飞哥文章的评论区。
 
## 2. 运行示例

安装完 Carla 之后再安装 Anacoda 以及其他你需要的库和软件，然后开始下一步，运行并查看 Carla 的示例代码，以理解 carla 的 api。这里小飞哥也做了很多笔记，大家可以查看文章 [基础API的使用](https://zhuanlan.zhihu.com/p/340031078)，或者大家也可以看 youtuber [sentdex](https://www.youtube.com/c/sentdex) 的视频 [Controlling the Car and getting Camera Sensor Data](https://www.youtube.com/watch?v=2hM44nr7Wms&list=PLQVvvaa0QuDeI12McNQdnTlWz9XlCa0uo&index=2)。

这里我写一个我个人认为比较重要的点，新手在运行 Carla 程序的时候很有可能会遇到 `import carla` 报错的情况。要解决这个问题只需要将 carla 的蟒蛇蛋 (.egg 文件) 安装到你的 anaconda 环境中即可，如：

```
export CARLA_ROOT = <Your path to Carla>
easy_install ${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
```

## 3. 自动驾驶模型体验

这里选择的模型是 [TransTuser](https://ap229997.github.io/projects/transfuser/)，代码仓库位于 [Github](https://github.com/autonomousvision/transfuser)。选择该模型的主要原因是其代码仓库包含了模型，训练，数据集以及在 Carla Leaderboard 上评估的全流程，并且文档写的很不错，对新手玩家比较友好。

这里暂时不对模型进行讲解，只是介绍一下如何安装，以及如何使用。（默认大家已经安装了 anaconda，和 carla 0.9.10.1）

**克隆仓库并配置环境：**

```
git clone https://github.com/autonomousvision/transfuser
cd transfuser
conda create -n transfuser python=3.7
conda activate transfuser
pip3 install -r requirements.txt
```

**下载数据集:**

该数据集是记录了一个基于规则手工编程的 agent，在 carla 的 8 个地图中的运行记录。输入完指令后，会提示选择数据集 0 或者是 1. 0 是一个小数据集，1 是一个大数据集（406G），我们暂时先下载那个小的就可以了。

```
chmod +x download_data.sh
./download_data.sh
```

该数据集的格式如下：

```
- TownX_{tiny,short,long}: corresponding to different towns and routes files
    - routes_X: contains data for an individual route
        - rgb_{front, left, right, rear}: multi-view camera images at 400x300 resolution
        - seg_{front, left, right, rear}: corresponding segmentation images
        - depth_{front, left, right, rear}: corresponding depth images
        - lidar: 3d point cloud in .npy format
        - topdown: topdown segmentation images required for training LBC
        - 2d_bbs_{front, left, right, rear}: 2d bounding boxes for different agents in the corresponding camera view
        - 3d_bbs: 3d bounding boxes for different agents
        - affordances: different types of affordances
        - measurements: contains ego-agent's position, velocity and other metadata
```

**下载预训练的模型：**

```
mkdir model_ckpt
wget https://s3.eu-central-1.amazonaws.com/avg-projects/transfuser/models.zip -P model_ckpt
unzip model_ckpt/models.zip -d model_ckpt/
rm model_ckpt/models.zip
```

**运行预训练的程序：**

打开 Carla Server：

```
<Path to carla>/CarlaUE4.sh --world-port=2000 -opengl
```

然后另开一个终端：

打开 `run_evaluation.sh` 文件：

```
gedit leaderboard/scripts/run_evaluation.sh
```

并对其进行修改，将等号后面的内容替换为下面的你内容：

```
export TEAM_AGENT=leaderboard/team_code/transfuser_agent.py
export TEAM_CONFIG=model_ckpt/transfuser
export CHECKPOINT_ENDPOINT=results/transfuser_result.json
export SCENARIOS=leaderboard/data/scenarios/town05_all_scenarios.json
```

之后：

```
leaderboard/scripts/run_evaluation.sh
```

程序正常运行后，你就可以看到一个自动驾驶的车辆在 Carla 挑战赛的场景中运行的画面了。