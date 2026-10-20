# 1. Recognition of Egyptian hieroglyphic texts through focused generic segmentation and cross-validation voting

## Abstract

- contribution1: 建立更完整的数据集

- contribution2: 组合分割与分类方法
    * segementation: 把符号与背景、相邻符号区分开，便于单独提取
    * classification: 判断提取出的符号属于哪个类别。

- ConvNeXt: CNN + Transformer
  
- cross validation voting: 通过不同训练／验证划分得到多个模型，再合并它们的预测

## Introduction
 Method Based on Region
Segmentation (MBRS) ->  Integrated Generic Segmentation Model (IGSM)

###  two previous approaches:

- classical computer vision technique: not generalize
- Yolo or Detectron2: d require more complex label-
ing of hieroglyphs

### new idea:
用提示引导通用 SAM，再单独分类:
图像处理产生提示 → SAM 提取符号 → 分类模型识别符号。

**Generic**: SAM 已经学习了通用图像分割能力。作者希望利用这种能力，减少为象形文字专门准备精细分割标注、重新训练分割模型的工作。

**Focused**: 直接对整张图运行全局分割，得到的结果不能满足需求。因此，用视觉处理方法产生特定提示，让 SAM 关注局部区域，逐个提取符号

## Proposed approach

### Dataset

**Gardiner's Sign List**

根据象形文字所描绘的对象（如人物、动物、植物、人体部位等）将其划分为不同类别，并为每个符号分配唯一的编号。例如，A1 表示坐着的男性，G17 表示猫头鹰.

在基于深度学习的古埃及象形文字识别任务中，Gardiner 编码通常被用作图像分类的类别标签（Class Label），使模型能够学习识别不同的象形文字符号。

The name of the class is made up of an alphabetic character followed by a number. The Gardiner code is a unique identifier for each hieroglyph class.

**Egyptian Hieroglyphic Dataset**

- 310 different hieroglyph classes and 13,729 image

- The hieroglyphs have been grouped in folders according to the structure and typology defined by the Egyptologist Gardiner
The images obtained have been classified and labeled for each of Gardiner’s classe

- The images obtained have different
backgrounds and sizes -> preprocessing -> all the images have the same dimensions in X or Y. 

### Segmentation

| approach | principal | resultat |
|---|---|---|
| MBRS | 利用边缘和区域提取符号 | 符号与背景差异比较明显 |
|IGSM | 用提示点引导预训练 SAM | 边界较难定义的图像 |

#### MBRS
![exemple MBRS](./note_imgs/MBRS_exemple.png)
原图 → Canny 边缘检测 → 分析独立区域 → 找外接矩形 → 裁剪

优点是流程直接，容易建立传统方法基线。

局限：图像中的边缘不一定都是符号边缘。 裂纹、阴影和背景纹理也可能产生边缘；一个符号的多段笔画，也可能被分成几个区域。

#### IGSM
![exemple IGSM](./note_imgs/IGSM_exemple.png)

1. **Otsu 二值化与清理**: 将图像转换成前景／背景形式，突出刻痕、阴影等区域，再删除过小的区域，减少噪声。
   
2. **寻找连通区域并采样**: 在二值图中，把相互连接的前景像素归为区域，再从这些区域中随机选择提示点。 一个连通区域不一定对应一个完整符号：它可能只是一段笔画，也可能连接了多个部分。
   
3. **用提示点引导SAM**: 将原图与这些点的位置提供给 SAM，得到候选分割结果。作者发现单次分割可能遗漏部分区域，所以使用多个点，并组合分割结果。
   
4. **去重和过滤**: 使用 NMS 减少重复结果，并过滤尺寸不合理的区域，例如可能来自石碑破损的过大区域。

传统图像处理产生候选线索＋SAM 分割＋后处理。

**分割结果怎样送入分类器？**

SAM 掩膜 → 确定符号范围 → 计算外接矩形 → 从原图裁剪带局部背景的小图 → 分类

作者能够去掉背景，只保留符号；但他们的实验发现，保留背景的矩形裁剪图分类更好，因此最终采用这种输入。

**Organizing the hieroglyphs so they can then be read
**

reading from right to left and from top to bottom

not very inmortant for our projet

### classification


#### ConvNeXt
ConvNeXt 是一种卷积神经网络架构，属于 CNN。 它来自 2022 年的论文 A ConvNet for the 2020s。该研究从 ResNet 出发，借鉴现代视觉网络的设计思路，改进卷积网络，形成 ConvNeXt 系列


#### CVV classification
generalizability -> CVV
模型对没有参与训练的新图片是否仍然能够正确分类。

different similar classifiers trained on disjoint validation group

**validation set**

to avoid overfitting

| data | is update parameters | usage |
|---|---|---|
|training set| yes | 计算训练损失，反向传播，更新参数 |
|validation set| no | 观察泛化表现，选择模型状态或决定何时停止 |

**early stopping**

通常监测验证表现，在若干轮没有改善后停止，并保留最佳检查点

**data distribution**

![CVV schema](./note_imgs/cvv_schema.png)

training data set into different training and validation slot

* $\tau$：用于训练和验证的完整开发数据；
* $T_i$: 第 $i$ 个模型的训练集；
* $V_i$: 第 $i$ 个模型的验证集；
* $k$: 模型数量

$$
\tau = \bigcup_{i=1}^{k} V_i
$$
各验证组共同覆盖完整开发数据。

$$
\bigcap_{i=1}^{k} V_i = \varnothing
$$
没有样本同时存在于所有验证组中

$$
[\tau = T_i \cup V_i] \quad \forall i \in [1,k]
$$
对每个模型来说，训练集与验证集合起来，就是完整开发数据

$$
[T_i \cup V_i = \varnothing] \quad \forall i \in [1,k]
$$
同一个模型的训练集和验证集没有交集


**output of all models**

对一张输入图片 $x_i$，第 $i$ 个模型输出：
$$
\mathbf p_i(x)=
[p_{i1}(x),p_{i2}(x),\ldots,p_{in}(x)]
$$

* $i$ 是模型编号；
* $j$ 是类别编号；
* $n$ 是类别总数；
* $p_{ij}(x)$ 是模型 $i$ 认为图片属于类别 $j$ 的概率


**soft voting**

对每个类别，将各模型的概率加权相加，再选择得分最高的类别

$$
q_j(x)=\sum_{i=1}^{k}w_i\,p_{ij}(x)
$$

$$
S(x)=c_{\arg\max_j q_j(x)}
$$

其中 $c_j$ 是第 $j$ 个类别的标签，$\arg\max$ 返回最大得分对应的类别编号。

$$
w_i=\frac{1}{k}
$$


**hard voting**
$$
bij(x) =
\begin{cases}
1,&j\text{ 是模型 }i\text{ 预测概率最大的类别}\\
0,&\text{其他类别}
\end{cases}
$$

把每个模型的概率向量转换成投票向量

$$
H(x)=c_{\arg\max_j\sum_{i=1}^{k}w_i\,b_{ij}(x)}
$$

然后对这些票数加权求和：

**different**
* 硬投票只保留每个模型的第一选择；
* 软投票保留各类别的概率分配。
  
**Random data augmentatio**

* 样本数量低于某个门槛的类别，增加增强样本；
* 样本过多的类别，删除部分图片；
* 每类训练图片的最大数量设为平均每类数量的两倍。

增强与数量调整共同进行。不是把所有类别机械地复制到相同数量，也不是只做随机变换。

| augmentation | setting |
|---|---|
| translation horizontal and vertical | maximum 10 pixels |
| small random rotations | 5 degrees |
|  random scalin | 0.90～1.10 |
| horizontal flipping | 

The total number of images in the
training set was 9359 and, after data augmentation, 18,71