### 膨胀门卷积-关系抽取
借鉴了苏剑林大佬的项目，详细的介绍都在大佬的仓库里有介绍

基于苏剑林大佬的工作，我将模型进行了微调使其应用于金融领域的文本关系抽取
### 文件目录介绍
#### complete.py

> 必须使用GPU环境的版本，这个bug在苏神原文中并没有点名，对初学者可能会带来影响，苏神的代码中调用了CuDNNGRU这个方法在keras只能以Tensorflow后端运行在GPU上，下面是官方文档截图：

![image-20210326184848728](https://tva1.sinaimg.cn/large/008eGmZEly1goxi1v30w0j30fw0n678c.jpg)

#### complete_cpu.py

> 这个脚本可以直接在cpu上运行

### 环境
python2.7+tensorflow1.5+keras2.2.4
### 引用
> @misc{
  jianlin2019bdkgf,
  title={A Hierarchical Relation Extraction Model with Pointer-Tagging Hybrid Structure},
  author={Jianlin Su},
  year={2019},
  publisher={GitHub},
  howpublished={\url{https://github.com/bojone/kg-2019}},
}