# 理解置信度传播算法及其拓展

## 1、推断和图模型

​		本文将要描述和解释置信度传播算法（Belief propagation，BP），该算法可以用于（近似的）解决“推断”问题。推断问题存在于很多科学领域中，所以一个解决该问题的好方法在不停地被重新提出也并不奇怪。事实上，很多截然不同的算法比如前向后向（Forward-backward）算法、维特比（Viterbi）算法、Gallager code 和 turbocode 的迭代解码（iterative decodeing）算法，Pearl的贝叶斯网络信念传播算法，卡曼滤波（Kalman filter）和 物理中的传递矩阵方法（transfer-martix approach）都是BP算法在不同科学领域的变种。

理解一个方法在多学科问题上的应用将会对该方法产生更加深刻的认识，因此我们从AI领域，机器视觉领域，统计物理领域和digital communications literatures领域中选取简单的推断问题，这将会让我们有机会看到在解决这些问题时对应的不同图模型，第一节是一个对于基础知识的简单复习，但我们希望这些知识能够帮助读者理解这些不同领域中的问题之间的深度相似性。

### 1.1 贝叶斯网络

在人工智能领域的文献中，贝叶斯网络可能是最流行的图模型，贝叶斯网络构建的专家系统涉及了很多领域例如医疗诊断，地图学习，语言理解，启发式搜索等。我们会使用一个医学诊断问题作为例子，加入我们希望能够构建一台自动为患者诊断的机器，对于每一名患者，我们都会得到一些信息（可能不完整），例如病人的症状和检测结果，然后我们想*推断*某个或某些能够造成这些症状的疾病的概率。我们同样假设我们已经知道（来自医学专家的建议）不同症状、检测结果和疾病之间的统计学关联。例如，让我们思考一下Lauritzen 和 Spiegelhalter所虚构的“Asia”例子，如图一。

模型中展现了不同变量之间定性的统计学关联情况，可以用以下文字表述：

1. 患者若近期有亚洲（A）的旅行经历会提高其患肺结核的概率（T）。
2. 吸烟（S）是患肺癌（L）和支气管炎（B）的危险因素。
3. 是否存在肺结核或者肺癌（E）可以经由X光（X）检测，但是仅有X光无法区分二者。
4. 呼吸困难（D）可能由支气管炎或者肺癌和肺结核的存在导致。

图1中的每一个节点都代表了一个拥有离散概率分布的变量，我们用 ![](https://latex.codecogs.com/svg.latex?x_{i}) 来表示变量  ![](https://latex.codecogs.com/svg.latex?i) 的不同可能的状态。除了贝叶斯网络中描述的变量之间的定性相关性之外，还存在定量的统计学关系存在于图中每条边的箭头中，每个箭头都代表了一个条件概率：例如，我们将患者并不吸烟却患上肺癌的概率写为  ![](https://latex.codecogs.com/svg.latex?p(x_{L}|p_{S}) )，对于这种链接，我们称S节点为L节点的父节点（parent），因为L节点的状态是以S节点的状态为条件的。某些节点例如D节点可能会有不止一个父节点，在这种情况下，我们可以根据节点所有的父节点在定义条件概率，例如  ![](https://latex.codecogs.com/svg.latex?P(x_{D}|x_{E},x_{B}) )为呼吸困难的条件概率。

贝叶斯网络定义了一种独立结构：某一节点处在某一种状态的概率仅仅直接取决于其父节点的状态。而对于像A，S一样的节点，我们引入类似 ![](https://latex.codecogs.com/svg.latex?P(x_{S})) 的不以任何父节点为条件的概率。 总的来说，一个贝叶斯网络（或者其他我们提到的图模型）在稀疏时最好用，因为在这时大多数的节点都有一个直接的统计学关联。

在我们的例子中，联合概率即患者具有出多种症状，检测结果和疾病的概率为  ![](https://latex.codecogs.com/svg.latex?p(\{x\})=p(x_A,x_S,x_T,x_L,x_B,x_E,x_D,x_X))  
该联合概率可以被写成概率图中所有父节点（更类似与根节点）和所有条件概率的连乘：

![](https://latex.codecogs.com/svg.latex?p(\{x\})=p(x_A)p(x_S)p(x_T|x_A)p(x_L|x_S)p(x_B|x_S)p(x_E|x_L,x_T)p(x_D|x_B,x_E)p(x_X|x_E))

   更广义的说，一个贝叶斯网络就是一个具有N个随机变量![](https://latex.codecogs.com/svg.latex?x_{i})的有向无环图（directed acyclic graph），其联合概率可以表示为：

![](https://latex.codecogs.com/svg.latex?p(x_1,x_2,...,x_N)=\prod_{i=1}^{N}p(x_i|Par(x_i)))

其中![](https://latex.codecogs.com/svg.latex?Par(x_{i})) 代表节点 ![](https://latex.codecogs.com/svg.latex?i) 的父节点的状态，若节点 ![](https://latex.codecogs.com/svg.latex?i) 没有父节点，则：
![](https://latex.codecogs.com/svg.latex?p(x_i|Par(x_i))=p(x_{i})) 
有向无环图代表着，we mean that the arrows do not loop around in a cycle-it is still possible that the links form a loop when one ignores the arrows.

我们的目标是计算特定变量的边缘概率（marginal probability），例如：我们可能想要计算某病人患有特定疾病的概率。计算这些边缘概率的过程，即为“推断”。从数学上说，对系统中其他节点所代表的变量做积分即可得到边缘概率，例如，如果想要得到最后一个变量p_{N} 的边缘概率，则需要计算：

![](https://latex.codecogs.com/svg.latex?p(x_{N})=\sum_{x_1}\sum_{x_2}...\sum_{x_{N}-1}p(x_1,x_2,...,x_N)) 

我们将我们计算出来的近似的边缘概率成为“信念”（belief），并将节点i的信念表示为 ![](https://latex.codecogs.com/svg.latex?b(x_{i}))。

如果我们有关于一些节点的信息（例如在Asia例子中我们已经知道某患者不吸烟），然后我们固定对应的变量并且在积分时不会将那个节点的未知状态加进去。我们将这样的节点成为已知节点，相反的其他未知节点则被称为隐含节点（hidden），在我们的图模型中，我们将使用实心圆来代表已知节点，空心圆代表未知节点。

对于小型的贝叶斯网络，我们可以轻易地对其做积分来使其边缘化，但不幸的是，需要被加数的数量会随着隐含变量的增加指数上涨，BP算法的优点就是在（近似）计算边缘概率的时候计算复杂度会随着变量的增加线性增加，因此，BP算法在实践中可以被用作大型贝叶斯网络中统计数据的”推断引擎“（误？）。在回到BP算法 之前，我们需要先介绍一些其他的推断问题。


### 1.2 成对马尔科夫随机场

BP算法最近同样被用作低级的计算机视觉问题的”引擎“，人们总是将一些简单的计算机视觉问题例如图像分割和识别物体的解决认为是理所应当，我们人类解决这些问题过于简单所以会惊讶于教机器解决同样的问题的难度，得到的数据仅仅有一系列的二维矩阵，解决机器视觉的关键在于找到在理论上可靠而且能够计算的模型。

成对马尔科夫随机场（Pairwise markov random field）MRF 为计算机视觉问题提供了一个诱人的理论模型，在这类问题中，我们一般想要从已知的数据中推断（a representation of whatever is "really out there"），其中数据最终就是一个由表示像素强度的数值形成的二维矩阵。举一个更加形象的例子，我们希望计算出视野中某个物体的距离，我们所使用的图像是1000*1000的灰度图，我们将问题转化为根据某像素的强度值![](https://latex.codecogs.com/svg.latex?I_{i})（intensity value）推断距离值 ![](https://latex.codecogs.com/svg.latex?d_{i})，其中 i 为一百万像素点中的一个。除了距离之外，我们可能也会需要推断其他的一些关于图像内容的数值，例如图像中遗失的高分辨率细节或者一系列图片中的光流（optical flow）。

总的说来，我们假设已经观察到一些图像点![](https://latex.codecogs.com/svg.latex?y_{i})的数值，并且我们希望能够推断其他的关于底层场景（underlying scene）![](https://latex.codecogs.com/svg.latex?x_{i})的数值，其中序号![](https://latex.codecogs.com/svg.latex?i) 可以代表单个像素点的位置，也可以代表一个小型像素方块的位置。我们进一步假设在![](https://latex.codecogs.com/svg.latex?x_{i})和![](https://latex.codecogs.com/svg.latex?y_{i})之间在位置i处存在某些统计学关联，并且将这种关联写为函数 ![](https://latex.codecogs.com/svg.latex?\phi_{i}(x_{i},y_{i})) ，
函数  ![1](https://latex.codecogs.com/svg.latex?\phi_{i}(x_{i},y_{i})) 经常被称为 ![](https://latex.codecogs.com/svg.latex?x_{i}) 的证据（evidence），最后，为了让我们能够推断场景中的任意事物，![](https://latex.codecogs.com/svg.latex?x_{i}) 之间一定存在某种结构，总的来说，如果我们不对这样一种结构作出假设，计算接视觉问题就是 inherently hopelessly ill-posed。我们假设节点![](https://latex.codecogs.com/svg.latex?i) 被放置在一个二维的网格中，就此而言，单个的节点![](https://latex.codecogs.com/svg.latex?x_{i}) 应当和周围的其他节点![](https://latex.codecogs.com/svg.latex?x_{j})互相“兼容”，可以表示为compatibility function ![](https://latex.codecogs.com/svg.latex?\psi_{i,j}(x_{i},x_{j})) ，其中![](https://latex.codecogs.com/svg.latex?\psi_{i,j}) 仅仅链接网格中相邻的位置。则，我们可以关于![](https://latex.codecogs.com/svg.latex?x_{i})和![](https://latex.codecogs.com/svg.latex?y_{i})的联合概率表示为：

![4](https://latex.codecogs.com/svg.latex?p({x},{y}=\frac{1}{Z}\prod_{(ij)}\psi_{ij}(x_{i},x_{j})\prod_{i}\phi_{i}(x_{i},y_{i})))

其中![](https://latex.codecogs.com/svg.latex?Z)为标准化参数，在![](https://latex.codecogs.com/svg.latex?i,j)上的连乘代表在方格上所有相邻的两个节点之间做连乘。

该模型可以表示为图二，其中实心圆代表已知的图像节点![](https://latex.codecogs.com/svg.latex?y_{i})，空心圆代表隐藏的场景节点![](https://latex.codecogs.com/svg.latex?x_{i}),这种马尔科夫随机场之所以被称为成对的（pairwise），就是因为compatibility function 仅仅和在位置上相邻的两个成对的![](https://latex.codecogs.com/svg.latex?i,j)有关。和贝叶斯网络相反，马尔科夫随机场是一个无向图，因此并没有在贝叶斯网络中某变量![](https://latex.codecogs.com/svg.latex?x_{i})是受其相邻的父节点影响的说法，所以在计算中使用compatibility function ![](https://latex.codecogs.com/svg.latex?\psi_{i,j}(x_{i},x_{j}))而不是条件概率![](https://latex.codecogs.com/svg.latex?p(x_{i}|x_{j}))。但是，二者在做推断的步骤上非常相似，都希望对所有的位置![](https://latex.codecogs.com/svg.latex?i)计算其信念（belief）![](https://latex.codecogs.com/svg.latex?b(x_{i}))，然后可以根据belief来推断出图片中隐藏场景的信息。同样的，直接计算所需的时间代价是指数级的，所以才会需要BP一样的快速算法。We note in passing that for a restricted class of pairwise MRF's relevant to computer vision, fast algorithms based on "graph cuts" can also be used to estimate hidden states.

### 1.3 Potts 和 Ising 模型

在这里有必要稍微跑题一下来展示成对MRF是如何被带入到名为“Potts model” 的物理模型中，首先让我们定义一个作用在相邻节点的变量之间的相互作用函数![](https://latex.codecogs.com/svg.latex?J_{i,j}(x_{i},x_{j})=\ln(\psi_{i,j}(x_{i},x_{j})))

和一个作用在诶一个单独节点的场函数![](https://latex.codecogs.com/svg.latex?h_{i}(x_{i})=\ln\phi_{i,j}(x_{i},y_{i}))(因为![](https://latex.codecogs.com/svg.latex?y_{i})是已知的，所以就从场函数中省略了），因此若我们定义Potts模型的能量（Energy）：

![](https://latex.codecogs.com/svg.latex?E({x})=-\sum_{i,j}J_{i,j}(x_{i},x_{j})-\sum_{i}h_{i}(x_{i}))

然后带入到统计力学中的玻尔兹曼法则中（Boltzmann‘s law）：

![](https://latex.codecogs.com/svg.latex?p({x})=\frac{1}{Z}e^{-E(\{x_{i}\})/T})

我们可以看到成对MRF的联合概率分布公式直接对应Potts模型在![](https://latex.codecogs.com/svg.latex?T=1)的情况下的![](https://latex.codecogs.com/svg.latex?p(x)),其中标准化常数![](https://latex.codecogs.com/svg.latex?Z)在物理学中成为配分函数（partition function），若每个节点可能的状态数为2，该模型则被称为Ising模型，在这种情况下，物理学家通常使用取值为1或-1的![](https://latex.codecogs.com/svg.latex?s_{i}) 替代![](https://latex.codecogs.com/svg.latex?x_{i})，并且进一步约束相互作用函数![](https://latex.codecogs.com/svg.latex?J_{i,j})使其成为对称的形式（？）并能够写成“旋转玻璃”(spin glass)能量函数。

![](https://latex.codecogs.com/svg.latex?E({s})=-\sum_{(i,j)}J_{i,j}s_{i}s_{j}-\sum_{i}h_{i}s_{i})

Ising 模型中，对于belief ![](https://latex.codecogs.com/svg.latex?b(x_{i}))的推断可以对应成物理中局部磁化强度（magnetization）的计算。

![](https://latex.codecogs.com/svg.latex?m_{i}\equiv{b}(s_{i}=1)-b(s_{i}=-1))


### 1.4 Tanner图和factor图

另一个BP算法成功应用的例子是纠错码（error-correcting codes）的迭代解码。BP算法作为很多最佳编码的解码器，其中就包括turbocodes和Gallager codes。对于纠错码的解码是推断问题的一个优雅（elegant）的例子，接收器接收到已经被有噪声的信道影响过的信息并且试图推断最初的信息。

Gallager编码和tuubo编码可以被归类为部分验证编码（parity-check codes），在部分验证编码中我们通常在N比特信息块（block）中包含k比特的信息，其中N大于k的部分为可以被用作修复误差的冗余信息，。让我们试一个简单的例子：一个 N = 6,k = 3 的二元编码，可以表示成图三一样的Tanner图。

Tanner图是纠错码中编码部分验证约束的图形化表现（pictorial represantation of the parity-check constraints on codewords），图中每一个方块代表一次parity-check，每一个和方块相连的圆圈代表着参与对应parity-check的1比特信息。在我们的例子中，第一个parity-check 将 #1 加 #2 的值赋予 #4，第二次parity-check将 #1 加 #3 的值赋予 #5 ，第三次将#2 加 #3 的值赋予 #6 ，满足这些规则的编码只有八组，000000,001011,010101,011110,100110,101101,110011和111000，在这个例子中，前三位编码即为想要发送的信息正文（例如：010），剩余的三位则是由信息正文唯一决定的（010101）。

在接受这些信息时，传输的信息已经被带有噪声的传输信道影响，经常会发现接收到的信息不符合前面所述的编码规则，例如假设最初发送的信息为010101，但是由于传输信道中噪声的影响，信息中其中1比特发生了变化，因此接收到的信息变为011101，接收到的信息不符合设定的编码规则，就需要解码算法（decoding algorithm）来推断最初的信息。在我们的例子中，编码010101是仅有的只需发生一次变化就会生成接受信息的代码，所以当发生错误的概率很低时，有充分的理由将其解码为010101。但不幸的是，在很多的情况中，N和k的数值会很大，所以在解码过程中可能的密文是指数级增加的，我们无法保存每一种可能性并逐个检查并找到和接收到的编码最接近的一个。至此我们又陷入到了熟悉的情况，我们可以利用BP算法，应为其工作时间仅会随着N的增加线性增加。

 从解码问题可以总结出一个概率公式，我们假设我们已经收到了一个N比特的序列![](https://latex.codecogs.com/svg.latex?y_{i})，并且我们试图找出N比特的发送码![](https://latex.codecogs.com/svg.latex?x_{i})。为了简化问题，让我们假设噪声信道是“无记忆”（memoryless）的，即每个某个比特的数据出错是独立于其他比特的。这代表着我们可以将条件概率![](https://latex.codecogs.com/svg.latex?p(x_{i}|y_{i}))和接受编码中对应的比特联系起来（associate a conditional probability![](https://latex.codecogs.com/svg.latex?p(x_{i}|y_{i}))  with each receieved bit）。例如，如果接收到的第一比特的信息为0，则我们知道它有![](https://latex.codecogs.com/svg.latex?f)的概率在传输过程中出错，则对应的发送码为0的概率为![](https://latex.codecogs.com/svg.latex?1-f)，对应的发送码为1的概率就为![](https://latex.codecogs.com/svg.latex?f)。因此![](https://latex.codecogs.com/svg.latex?p(x_{1}=0|y_{i}=0)=(1-f))，![](https://latex.codecogs.com/svg.latex?p(x_{1}=1|y_{i}=0)=f)，每个编码总体的概率正比于单个比特概率的乘积![](https://latex.codecogs.com/svg.latex?\prod_{i=1}^{N}p(x_{i}|y_{i}))。但是确保我们只考虑那些符合编码规则的编码（make sure that we only consider combinations of xi that are codewords）。我们将这些条件概率的乘积和parity check的限制写成联合概率，例如，在N=6和k=3的情况下，联合概率可以写为：

![](https://latex.codecogs.com/svg.latex?p(\{x\},\{y\})=\frac{1}{Z}\psi_{124}(x_{1},x_{2},x_{4})\psi_{135}(x_{1},x_{3},x_{5})\psi_{135}(x_{2},x_{3},x_{6})\prod_{i=1}^{6}p(x_{i}|y_{i}))

在上面的公式中![](https://latex.codecogs.com/svg.latex?\psi_{124}(x_{1},x_{2},x_{4}))parity check函数，在参数相加为奇数时为1，偶数为0。

总的说来，对于一个发送码为![](https://latex.codecogs.com/svg.latex?x_{i})接受码为![](https://latex.codecogs.com/svg.latex?y_{i})，N-k parity check的部分检查码来说，其联合概率可以写为：

![](https://latex.codecogs.com/svg.latex?p(\{x\},\{y\})=\frac{1}{Z}\prod_{a=1}^{N-k}\psi_{a}(\{x\}_{a})\prod_{i=1}^{6}p(x_{i}|y_{i}))

其中，![](https://latex.codecogs.com/svg.latex?\psi_{a}(\{x\}_{a}))代表第![](https://latex.codecogs.com/svg.latex?a)次parity check 函数![](https://latex.codecogs.com/svg.latex?\psi_{a})和它的参数![](https://latex.codecogs.com/svg.latex?\{x\}_{a})。

parity check编码的优化算法会将错误解码的次数减到最低，其对每一个![](https://latex.codecogs.com/svg.latex?i)计算边缘概率![](https://latex.codecogs.com/svg.latex?p(x_{i}))，然后将该位置的接收编码改为边缘概率最大的编码。将对每个比特独立的改为边缘概率最大的编码是可以将错误编码数量减到最低的一种策略，尽管有可能会生成无效的解码。

同样的，直接计算存在指数复杂度的问题，要借助BP算法，无论是Galager code还是turbocode，都十分的高效，并且使传输接近香农极限。

因子图是Tanner图的拓展，从图形上看完全相同，但是在因子图中方块可以代表其相连参数的任意函数，每个函数的参数数量不做限制
