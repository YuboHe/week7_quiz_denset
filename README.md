# 第七周作业

## 作业1

利用slim框架，做一个inceptionv4的迁移训练

### 结果反馈

![Screen Shot 2018-01-11 at 4.22.08 pm](/Users/Yubo/Desktop/Knowledge/AI/CSDN_AI_Eng/csdn_week7/week7_quiz_denset/Screen Shot 2018-01-11 at 4.22.08 pm.png)

在TinyMind上训练选用默认参数，经过7小时训练，TOP1 可以达到0.678

TOP5 可以到达0.869. 之后更改了lr, dropout, 以及batch 试了一下，每个看了半小时

## 作业2

学员自己实现一个densenet的网络，并插入到slim框架中进行训练。按照论文以及老师代码实现，代码在tinymind跑起来了至少，之前失败原因由于没在fullyconnect前面加flatten操作导致报错，询问助教好知道错误并改正，网络成功跑起来了。。拖了有点久

![Screen Shot 2018-01-15 at 11.33.35 am](/Users/Yubo/Desktop/Knowledge/AI/CSDN_AI_Eng/csdn_week7/week7_quiz_denset/Screen Shot 2018-01-15 at 11.33.35 am.png)