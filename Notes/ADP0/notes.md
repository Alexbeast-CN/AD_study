# ADP0. Carla 初体验

## 1. 安装 Carla

> ADP 是 Autonomous Driving Practise 自动驾驶实践的缩写。是配合着 AD 理论课程的应用层。这里我们选择 Carla 作为自动驾驶模拟器，因为目前来说 (22 年)，Carla 是学界和工业届主流的自动驾驶模拟器。

安装 Carla 的过程 [@叶小飞](https://www.zhihu.com/people/xie-xiao-fei-78-24) 已经写了非常详细的过程了，大家跟着 [史上最全Carla教程 |（二）Carla安装](https://zhuanlan.zhihu.com/p/338927297) 进行安装就可以了。这里我只写一些简短的说明：

1. CARLA 版本选择。这里我们选择的是 CARLA 0.9.10.1，选择这个版本的主要原因是 [Carla Challenge](https://carlachallenge.org/) 所选用的版本是 0.9.10.1。因此绝大多数的学术研究也是使用该版本的 Carla。
2. Ubuntu 版本选择。与该版本 CARLA 对应的最佳选择是 Ubuntu 18.04。
3. 在安装 UE4 之前需要先下载显卡驱动，Linux 并没有自带驱动。在 [Nvidia 的网站](https://www.nvidia.com/content/DriverDownload-March2009/confirmation.php?url=/XFree86/Linux-x86_64/470.129.06/NVIDIA-Linux-x86_64-470.129.06.run&lang=us&type=TITAN) 上有提供 Vulkan 驱动的安装包。
4. 在安装 CARLA 之前请先卸载 Anaconda，否在在 make 的过程中会出现奇奇怪怪的问题导致失败。
 
## 2. 运行示例

安装完 Carla 之后再安装 Anacoda 以及其他你需要的库和软件，然后开始下一步，看 Carla 的示例代码：

