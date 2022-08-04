#! https://zhuanlan.zhihu.com/p/536102804
# ADP1. Carla 排行榜参赛程序梳理

> 不了解 Carla 排行榜的同学可以查看 [@Here-Kin](https://www.zhihu.com/people/kin_zhang) 的文章 [【排行榜】Carla leaderboard 排行榜 运行与参与手把手教学](https://www.cnblogs.com/kin-zhang/p/16395716.html)，本文主要对排行榜的代码格式进行梳理。使用的代码示例来自 [Transfuser](https://github.com/autonomousvision/transfuser)，关于该参考的体验方法可以查看文章 [ADP0. Carla 初体验](https://zhuanlan.zhihu.com/p/532507009).

开始之前，请先进入文件夹：Transfuser

```
cd xxx/transfuser
```

## 1. 起始点: leaderboard/scripts/run_evaluation.sh

**作用：** 为 `run_evaluaion.py` 文件传参数

**参数：**

```sh
- scenarios # 一个指向 '<map_scenarios>.json' 文件的路径，文件中其中包含地图中的所有场景

- routes # 一个指向 '<map_routes>.xml' 文件的路径，文件中包含地图中的所有路线。 路线使用的起点坐标和终点的坐标表示。

- repetitions # 1 表示多次评估运行； 0 用于单次评估运行。 

- track # SENSOR 或者 MAP，即你想要参加的比赛类型。

- checkpoint # 一个指向 '<results>.json' 文件的路径，其中包含用于保存统计信息和测评结果。

- agent # 一个指向模型代理 '<agent>.py' 文件的路径。

- agent-config # 一个指向预训练模型参数 '<best_model>.pth' 文件的路径。

- debug # 是否开启调试模式，开启后可以看到由黑色路径点表示的全局路线。使用 1 或 0 来开启和关闭。

- record # 是否使用 CARLA 录制功能创建场景录制。使用 1 或 0 来开启和关闭。

- resume # 是否从上一个检查点恢复执行？使用 1 或 0 来开启和关闭。

- port # TCP 端口（默认值：2000）。

- trafficManagerPort # TrafficManager 的端口（默认值：8000）。
```

## 2. leaderboard_evaluator.py

**功能：** 设置评估环境（模拟器）并加载代理。

它使用 `<agent>.py` 中的 `get_entry_point()` 函数来查找代理类名称，并使用该类创建参与排行榜评估机制的代理。

## 3. agent.py (以 cilrs_agent.py 为例):

所有代理类都派生自 `autonomous agent.Autonomous Agent()` 类。 在自己创建 `<my_agent>.py` 的过程中大多数部分均可以保留 `cilrs_agent.py` 的写法，仅有少部分如修改传感器之类的内容需要改动。下面是 `cilrs_agent.py` 与 `transfuser_agent.py` 的对比图，同一行的左右两边分别出现红色和绿色则表示有改动。

![agent 代码对比图](./../pics/agent_compare.png)

下面将对程序的主要函数进行介绍：

```py
- setup(): # 与 '__init__()' 写法类似，但主要用于设置模型所需的传感器及储存传感器数据的文件路径，传入和模型网络和其预训练参数文件。

- __init(): # 仅用于将导航模块传入 agent。

- sensors(): # 设置 agent 所使用的传感器类型，位置以及参数。

- tick(): # 对传感器的输入数据进行预处理。

- run_step(): # 1. 将传感器的输入数据格式转变为 tensor 并存入 cuda 和缓存中。
              # 2. 使用 'net.encoder(data)' 将缓存中的数据传入模型的神经网络中。
              # 3. 获取模型输出的动作 'steer', 'throttle', 'brake'，以及其他可能需要的信息。
              # 4. 该函数会将以上动作 return 到 'leaderboard_evaluator.py' 中。
```

## 4. model.py

此部分的内容为 `agent.py` 中 `setup()` 的输入，也就是你的模型部分。此部分是项目的核心内容，不同项目之间大不相同，但是大致可以分为以下的几个部分。以 cilrs 模型的图为例：

![模型](./../pics/model.png)

首先是对于输入量，入传感器数据和测量数据进行处理的神经网络，我们叫做 `encoder`，以及对于处理后的信息进行映射的神经网络，我们叫做 `decoder`。在自动驾驶中，就是由环境以及车辆自身信息（如速度和导航信息）为输入，到车辆动作为输出的映射模型。

具体是如何实现的，我将在以后的文章中进行讲解。

- 上篇：[ADP0. Carla 初体验](https://zhuanlan.zhihu.com/p/532507009)
- 下篇：[]()