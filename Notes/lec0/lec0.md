#! https://zhuanlan.zhihu.com/p/517710302
# AD0. 自动驾驶学习资料汇总

## 1. 课程资料

- 德国蒂宾根大学的自动驾驶课程 (Self-Driving Cars, lectureed by Prof. Andreas Geiger, University of Tübingen)
  - [课程视频 | Youtube](https://www.youtube.com/playlist?list=PL05umP7R6ij321zzKXK6XCQXAaaYjQbzr)
  - [课程主页](https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/autonomous-vision/lectures/self-driving-cars/)
  - [其他课程资料 | 百度网盘](https://pan.baidu.com/s/1OsKPZ9KUTlLO26SHR_-yTg)，提取码：a69d

> 本专栏的主要学习资料即来自于蒂宾根大学的自动驾驶课程，本专栏的 [Github](https://github.com/Alexbeast-CN/AD_study)

- 德国波恩大学的自动驾驶课程 (Techniques for Self-Driving Cars" taught at the University of Bonn)
  - [课程视频 | Youtube](https://www.youtube.com/watch?v=EBFlmHqgezM&list=PLgnQpQtFTOGQo2Z_ogbonywTg8jxCI9pD)
  - [课程主页](https://www.ipb.uni-bonn.de/sdc-2020/)

- MIT 的自动驾驶课程 (Self-Driving Cars: State of the Art (2019), taught by Lex Fridman)
  - [课程视频 | youtube](https://www.youtube.com/playlist?list=PLrAXtmErZgOeY0lkVCIVafdGFOTi45amq)
  - [课程主页](https://deeplearning.mit.edu/)
  - [代码仓库 | Github](https://github.com/lexfridman/mit-deep-learning)

## 2. 比赛

- [CARLA Autonomous Driving Challenge](https://leaderboard.carla.org/)
  - 基于 Carla 模拟器
  - 比赛规模很大。学界和工业届都会参加
  - 提供 RGB、深度、分割、GPS 和 IMU 传感器
  - 基本上参加比赛的队伍都会发表至少一篇论文
  - 随机找的 [比赛论文](https://www.sciencedirect.com/science/article/pii/S2352146521007699)
  - 随机找的 [代码仓库](https://github.com/bradyz/2020_CARLA_challenge)

![Carl Challenge](pics/carla_challenge_2020.jpg)

- [Learn to race challenge](https://www.aicrowd.com/challenges/learn-to-race-autonomous-racing-virtual-challenge)
  - 基本上每年都有比赛
  - 一般都是 CMU 主办
  - 基于 OpenAI gym 的模拟器
  - 提供 RGB、深度、分割、GPS 和 IMU 传感器
  - 使用强化学习
  - 比赛 [论文](https://learn-to-race.org/assets/papers/2103.11575.pdf)
  - 官方提供的 [代码](https://github.com/learn-to-race/l2r)

![Learn to race challenge Overview](pics/image_overview.png)
![System](pics/main_figure.png)

- [DeepRacer | AWS](https://student.deepracer.com/home)
  - 基本上每年都有比赛
  - 比赛需要使用 ASW 的服务器进行训练
  - ASW 开发的模拟器
  - 场地内只有一辆带有摄像头的小车
  - 使用强化学习

![DeepRacer](./pics/deepracer.png)

## 3. 论文

> 读论文是学习自动驾驶的必经之路。

目前自动驾驶的解决方案一般分成两种，一条是工业届目前使用的比较多的 pipeline，即，将自动驾驶任务拆分成多个小块然后一块一块的解决。一般是分成以下五个小块：

![Pipeline](./pics/taks.png)

另外一条路是直接使用强化学习算法由传感器直接到控制，目前这个方法学界研究的比较多，因为需要开发新的算法，难度很大。

![Reinforcement Learning](./pics/rl.png)

- 下篇：[]()
