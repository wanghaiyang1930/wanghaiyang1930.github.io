# FollowMatching 技术方案全景

> 本文系统梳理 "FollowMatching" 相关技术在**学术界、开源界、工业界**三个维度的方案,覆盖核心思想、数学模型、精度/速度权衡与代表基准。
>
> **术语澄清(务必先读):** "FollowMatching" **并非自动驾驶/感知领域的标准术语**,当前学术文献与工业产品中不存在以此命名的单一算法或数据集。经系统检索,该词最可能是对以下两条独立技术脉络的复合指代,二者关键词高度撞车("Follow" + "Matching/Match"):
>
> 1. **跟驰行为建模(Car-following / Following behavior)** —— 以 **FollowNet**(跟车基准)、**FollowMe**(跟随前车轨迹预测)为代表,研究"后车如何跟随前车"。
> 2. **流匹配生成模型(Flow Matching)** —— 一类用于**轨迹预测与运动规划**的生成式建模范式(Conditional Flow Matching),是扩散模型(Diffusion)的高效替代。
>
> 本文对**两条脉络均做完整覆盖**,并在文末给出选型建议。所有关键数据均来自一手论文与官方仓库;定价/市占类数字标注口径与时间。

---

## 目录

- [0. 术语定义与整体版图](#0-术语定义与整体版图)
- [1. 跟驰行为建模(Following Behavior)](#1-跟驰行为建模following-behavior)
  - [1.1 传统跟车模型](#11-传统跟车模型)
  - [1.2 数据驱动跟车模型](#12-数据驱动跟车模型)
  - [1.3 FollowNet:跟车行为基准](#13-follownet跟车行为基准)
  - [1.4 FollowMe:跟随前车轨迹预测](#14-followme跟随前车轨迹预测)
- [2. 流匹配生成模型(Flow Matching)](#2-流匹配生成模型flow-matching)
  - [2.1 核心原理与数学模型](#21-核心原理与数学模型)
  - [2.2 Flow Matching vs Diffusion](#22-flow-matching-vs-diffusion)
  - [2.3 轨迹预测:T-CFM](#23-轨迹预测t-cfm)
  - [2.4 运动规划:FlowDrive / Flow Planner](#24-运动规划flowdrive--flow-planner)
  - [2.5 自适应步长与直接控制](#25-自适应步长与直接控制)
- [3. 开源界方案](#3-开源界方案)
- [4. 工业界方案](#4-工业界方案)
- [5. 综合对照表](#5-综合对照表)
- [6. 选型建议](#6-选型建议)
- [7. 参考来源](#7-参考来源)
- [8. 核验说明](#8-核验说明)

---

## 0. 术语定义与整体版图

由于 "FollowMatching" 非标准术语,先厘清两条脉络的技术内涵与它们的**交汇点**:

| 脉络 | 研究问题 | 代表工作 | 输出 | 关键指标 |
| --- | --- | --- | --- | --- |
| **跟驰行为建模** | 后车如何根据前车状态调整加速度/间距 | IDM、GHR、FollowNet、FollowMe | 加速度 / 间距 / 后车轨迹 | 间距 MSE、碰撞率(collision rate) |
| **流匹配生成** | 从噪声直接生成多模态未来轨迹分布 | T-CFM、FlowDrive、Flow Planner | 一组候选轨迹(waypoints) | minADE/minFDE、闭环评分(PDMS) |

**交汇点:** 跟驰(纵向跟随前车)本质是**轨迹预测/规划的一个子问题**;而 Flow Matching 作为通用生成范式,**完全可以用来建模跟驰行为**——给定前车历史,生成后车的多模态未来轨迹。因此"FollowMatching"可理解为:**用流匹配等生成范式对跟随/跟驰行为进行建模与规划**。这一交汇正是当前(2024–2026)自动驾驶规划的主流方向之一。

**一个贯穿全域的事实:** 传统跟车模型(IDM 等)**参数少、可解释、确定性输出**,但**难以刻画人类驾驶的多模态与随机性**;数据驱动/生成模型**拟合能力强、可输出分布**,但**可解释性弱、需大量数据、部署算力更高**。二者互补,没有单一模型全面占优。

---

## 1. 跟驰行为建模(Following Behavior)

跟驰(Car-following, CF)描述**后车如何调节加速度以与前车保持安全间距**,是交通流仿真的核心组件,也已进入量产车 ADAS(ACC/CACC)。相关综述指出 CF 建模横跨交通工程、物理、动态系统控制、认知科学、机器学习与强化学习多个学科([arXiv:2304.07143](https://arxiv.org/abs/2304.07143),IEEE T-IV 2024,DOI 10.1109/TIV.2024.3409468)。

### 1.1 传统跟车模型

传统模型基于运动学/动力学或心理-物理机理,以解析公式给出后车加速度:

- **IDM(Intelligent Driver Model)**:最广泛使用的确定性跟车模型。后车加速度由当前速度、与前车的速度差、实际间距与"期望间距"共同决定,期望间距随速度差动态变化。**参数物理意义清晰**(期望速度、最小间距、舒适减速度等),常作为数据驱动模型的基线与对照。
- **GHR(Gazis-Herman-Rothery)**:经典刺激-反应模型,后车加速度正比于前车相对速度、并对自车速度与间距做幂次加权;是最早的跟车模型族之一。
- **Gipps 模型**:安全距离模型,后车速度受"能在前车急停时安全停下"的约束,天然含碰撞规避逻辑。
- **ACC / CACC(自适应巡航 / 协同自适应巡航)**:控制论视角的跟车律。CACC 借助 V2V 通信获取前车加速度意图,可实现更短车头时距与更稳定的队列(platooning)。

传统模型的定位:**交通仿真、ADAS 控制律、可解释基线**;局限是难以刻画人类驾驶的异质性与随机性。

### 1.2 数据驱动跟车模型

从真实轨迹数据学习跟车策略,拟合能力强于解析模型:

- **全连接神经网络(NN)**:以历史状态(间距、速度、速度差)回归下一步加速度。
- **LSTM**:引入时序记忆,建模跟车行为的历史依赖,是序列跟车预测的常用基线。
- **强化学习(DDPG)**:把跟车建模为连续控制问题,通过奖励(安全 + 舒适 + 效率)学习跟车策略。**Deep Deterministic Policy Gradient** 适合连续动作空间,是 FollowNet 基准中表现最强的基线之一(见 1.3)。
- **模仿学习(IL)**:直接从人类驾驶轨迹模仿,规避奖励函数设计难题。

综述结论:数据驱动方法在拟合真实数据分布上占优,但**泛化性、可解释性、以及对分布外场景(OOD)的鲁棒性**仍是开放问题([arXiv:2304.07143](https://arxiv.org/abs/2304.07143))。

### 1.3 FollowNet:跟车行为基准

**FollowNet** 是港科大 HKUST DRIVE AI Lab 发布的**首个统一跟车行为基准**,目标是像 ImageNet/KITTI 之于视觉那样,为跟车建模提供统一评测口径([Nature Scientific Data 2023](https://www.nature.com/articles/s41597-023-02718-7),DOI 10.1038/s41597-023-02718-7;[arXiv:2306.05381](https://arxiv.org/abs/2306.05381))。

- **数据规模**:用**统一的抽取标准**从 **5 个公开驾驶数据集**中提取 **80K+ 跟车事件**,覆盖不同道路类型、多种天气以及含自动驾驶车辆的混合交通流。统一口径解决了此前各数据集格式不一、难以横向对比的痛点。
- **评测指标**:**间距 MSE(Mean Squared Error of spacing)** 与 **碰撞率(collision rate)**。
- **基线模型**:IDM、GHR(传统)+ NN、LSTM(深度学习)+ DDPG(强化学习),共 5 个代表性方法。
- **关键结果**:**DDPG 表现最有竞争力**——其**间距 MSE 低于 IDM 与 GHR**,且在多数数据集上**碰撞率低于 NN 与 LSTM**([arXiv:2306.05381](https://arxiv.org/abs/2306.05381) 摘要)。论文原文未在摘要层面给出逐数据集的绝对数值,建议以官方仓库复现结果为准。
- **开源**:数据与实现开源于 [github.com/HKUST-DRIVE-AI-LAB/FollowNet](https://github.com/HKUST-DRIVE-AI-LAB/FollowNet)。

> 说明:上文的 "5 个源数据集" 常见候选为 HighD、NGSIM、SPMD、Waymo、Lyft 等公开集,但原文摘要未逐一列名,此处不做强断言(见核验说明)。

### 1.4 FollowMe:跟随前车轨迹预测

**FollowMe** 关注一个更具体的问题:**当自车被要求跟随一条由"虚拟前车"给定的路线时,驾驶员/自车实际会走出怎样的轨迹**([arXiv:2304.06121](https://arxiv.org/abs/2304.06121),2023,CC BY 4.0)。

- **问题建模**:把"跟随前车路线的能力"转化为**运动与行为预测(motion & behavior prediction)**任务——给定前车路线,预测自车实际轨迹。
- **数据集**:发布同名 FollowMe 数据集(摘要未给出规模、采集设置、传感器模态与划分细节)。
- **基线方法**:**FollowMe-STGCNN**,一个**时空图卷积网络(spatio-temporal graph model)**,用图结构建模交互。
- **核心结论**:与既有运动预测模型对比表明,**跟随前车这一设定需要专门的设计机制**,通用轨迹预测模型并非最优。摘要未点名具体位移误差指标,该领域通常用位移误差(ADE/FDE)类指标评测。

**FollowNet vs FollowMe 区分:** FollowNet 是**纵向跟车(加速度/间距)**的通用基准,评测间距 MSE 与碰撞率;FollowMe 是**二维轨迹跟随**问题,用图模型预测自车轨迹。二者名字接近但任务层次不同。

---

## 2. 流匹配生成模型(Flow Matching)

Flow Matching(流匹配)是一类**连续归一化流**框架下的生成式建模范式,近两年(2024–2026)迅速成为**轨迹预测与运动规划**中扩散模型(Diffusion)的高效替代。

### 2.1 核心原理与数学模型

Flow Matching 训练一个**时变速度场(velocity field)** `v_θ(z_t, t | c)`,使样本沿常微分方程(ODE)从简单先验(通常高斯)连续演化到目标数据分布:

```
dz_t / dt = v_θ(z_t, t | c),   t ∈ [0, 1]
```

其中 `c` 是**条件上下文**。在自动驾驶中,`c` 编码:自车与周围 agent 的历史轨迹、高精地图(车道/路沿/人行横道等 polyline)、以及期望目标位姿。上下文通常由 **Transformer 编码器**融合得到([arXiv:2602.10285](https://arxiv.org/html/2602.10285))。

**Conditional Flow Matching(条件流匹配)** 的关键优势在于其训练目标沿**近似直线的概率路径(rectified/straight flow)**,使推理时只需**极少的 ODE 积分步数**即可从噪声生成高质量轨迹——这是它相对扩散模型提速的根本原因。推理即对学到的速度场做前向积分(如 Euler 求解器)。

### 2.2 Flow Matching vs Diffusion

| 维度 | Diffusion(DDPM/DDIM) | Flow Matching |
| --- | --- | --- |
| 生成过程 | 迭代去噪,随机 SDE | ODE 沿直线概率路径积分 |
| 推理步数 | 10–256+ 步 | **1–8 步**(常见 4 步) |
| 推理延迟(规划) | 50–125ms | **40–45ms** |
| 步数灵活性 | 需与训练噪声调度耦合,调整常需重训 | 训练/推理步数**可解耦**,无需重训 |
| 轨迹质量 | 良好,但约束违反更多 | 更平滑、约束违反更少 |

核心差异:Flow Matching 学的是"**直流(straight flow)**"确定性 ODE,而非迭代随机去噪,因此可在**1 步**下就生成高质量样本([arXiv:2403.10809](https://arxiv.org/html/2403.10809v1)),在规划场景下达到 **20Hz** 实时率([arXiv:2602.10285](https://arxiv.org/html/2602.10285))。

### 2.3 轨迹预测:T-CFM

**T-CFM(Trajectory Conditional Flow Matching)** 统一**轨迹预测与生成**,用流匹配替代扩散([arXiv:2403.10809](https://arxiv.org/html/2403.10809v1))。

- **提速**:相比扩散模型**最高快约 100×**,**1 个采样步**即可生成高质量样本(扩散需 256+ 步)。
- **对抗追踪**(vs 扩散模型 CADENCE):Prisoner-High 数据集 **120 分钟 ADE 0.110 vs 0.118**(T-CFM 更优),预测精度提升约 12–17%。
- **航迹预测**(vs FlightBERT,OpenSky Cessna 数据):各项指标**平均提升 35.4%**;30 分钟纬度 MAE **0.075 vs 0.102**。
- **长程规划**(Maze2D,vs Diffuser):**1 步采样**下平均分 **109.9 vs 34.2**(+142%)。

### 2.4 运动规划:FlowDrive / Flow Planner

两项 2025 年工作把流匹配用于**闭环规划**,均在 **nuPlan** 基准上超越扩散规划器。nuPlan 评分为 0–100 的闭环综合分(安全 + 进度 + 舒适)。

**FlowDrive**([arXiv:2509.21961](https://arxiv.org/html/2509.21961v1)):学习条件 rectified flow,直接把噪声映射到轨迹分布。

- **nuPlan 闭环(reactive)**:Val14 **85.37**(FlowDrive)/ **92.96**(FlowDrive\*);Test14 **87.28 / 93.96**;Test14-hard **73.09 / 81.96**。
- **对比**:超越 Diffusion Planner(Test14-R 82.93)、PLUTO(78.62)、PlanTF(79.58);运行时 **40ms vs 扩散 50ms(10 步)**。
- **架构**:MLP-Mixer 编码器 + **DiT 解码器**(adaLN-Zero 调制、3 层、6 头、隐维 192);轨迹表示 40 waypoints(4s);**推理 8 步**。
- **创新**:基于聚类(K=20)的数据平衡,对稀有运动模式重加权;mid-flow(t=0.5)注入横向扰动的 "moderated guidance" 提升多样性。

**Flow Planner**([arXiv:2510.11083](https://arxiv.org/html/2510.11083)):面向复杂交互场景的流匹配规划器。

- **nuPlan Val14(non-reactive/reactive)**:**90.43 / 83.31**,自称**首个无需规则精修即突破 90 分**的学习型方法(Diffusion Planner 89.87 / 82.80);带精修可达 94.31 / 92.38。
- **interPlan(强交互场景)**:**61.82** vs Diffusion Planner 52.90(**+8.92 分**)。
- **创新**:细粒度轨迹分段 tokenization(20 点/段、重叠 10 点)+ 尺度自适应注意力 + **无分类器引导(classifier-free guidance)** 增强交互建模。引导速度场 `ṽ = (1-ω)v + ω·v(·|C)`,`ω>1` 加强条件信号。
- **推理**:4 步 ODE 求解器,约 **12Hz**。

### 2.5 自适应步长与直接控制

- **Adaptive Time Step Flow Matching**([arXiv:2602.10285](https://arxiv.org/html/2602.10285)):提出**方差自适应流匹配**——辅助网络 `σ_φ` 估计局部不确定性,步长 `Δt ∝ 1/σ_φ`:陌生场景用小步长精细更新,自信区域用大步长。自适应变体**平均 4.7 步** vs 扩散 10–20+ 步;达 **20Hz(45ms)** vs DDPM 的 83–125ms;在 Waymo Open Motion(1s 历史、8s 预测)上取得最低 **minFDE 0.09–0.11m**,且约束违反远少于 DDPM/DDIM。架构:Motion Transformer 编码器 + U-Net 速度场(base width 128,倍率 1/2/4)+ 凸二次规划后处理(平滑与动力学约束)。
- **Learning Direct Control Policies with Flow Matching**([arXiv:2605.14832](https://arxiv.org/html/2605.14832)):流匹配规划器**直接输出可执行控制轨迹**(加速度与曲率 profile),而非几何路径点,更贴近底层控制。

---

## 3. 开源界方案

| 项目 | 类别 | 内容 | 链接 |
| --- | --- | --- | --- |
| **FollowNet** | 跟车基准 | 80K+ 跟车事件 + IDM/GHR/NN/LSTM/DDPG 基线实现 | [GitHub](https://github.com/HKUST-DRIVE-AI-LAB/FollowNet) |
| **FollowMe** | 轨迹预测 | FollowMe 数据集 + STGCNN 基线(arXiv 附源码/数据) | [arXiv:2304.06121](https://arxiv.org/abs/2304.06121) |
| **T-CFM** | 流匹配预测 | 轨迹条件流匹配,统一预测与生成 | [arXiv:2403.10809](https://arxiv.org/html/2403.10809v1) |
| **FlowDrive / Flow Planner** | 流匹配规划 | nuPlan 闭环 SOTA 级流匹配规划器 | [2509.21961](https://arxiv.org/html/2509.21961v1) · [2510.11083](https://arxiv.org/html/2510.11083) |

**关联生态与工具链:**

- **交通仿真器**:**SUMO**、**highway-env**、**CommonRoad** 等内置 IDM/Krauss 等跟车模型,是跟车策略训练与评测的常用环境。
- **规划基准**:**nuPlan**(闭环规划,Val14/Test14/Test14-hard)、**Waymo Open Motion Dataset(WOMD)**、**interPlan**(强交互扩展)是流匹配规划的主战场。**nuScenes prediction** 常用于开环轨迹预测。
- **生成建模库**:Meta 的 **`flow_matching`** 库、`torchcfm`(Conditional Flow Matching)等提供流匹配训练原语,可直接用于轨迹生成任务。
- **主流框架集成现状(核实结论)**:截至 2026 年,**FollowNet/FollowMe 与上述流匹配规划器均未被 MMDetection3D 或 Apollo 官方集成**——前者是感知(3D 检测)库,后者的规划模块(EM/Lattice planner)仍以搜索+优化为主,流匹配类方法尚处研究/复现阶段,主要以各论文自带仓库形式存在。

---

## 4. 工业界方案

跟驰与轨迹规划是量产 ADAS/AD 的核心能力,但落地形态与学术命名不同:

- **ACC / CACC 量产落地**:自适应巡航(ACC)是 L2 的标配功能,底层控制律多为 IDM/PID 类的确定性模型 + 安全兜底,而非直接部署生成模型。**CACC / 车辆编队(platooning)** 依赖 V2V(如 C-V2X),在商用车队列跟驰上有试点,可缩短车头时距、降低油耗(具体节油率因队列与路况而异,待以厂商实测为准)。
- **生成式规划上车趋势**:扩散/流匹配类**生成式规划器**目前主要处于**研究与预研**阶段。特斯拉、Wayve、Waymo 等在端到端/生成式规划方向有公开研究表述,但**是否将 Flow Matching 具体算法用于量产栈,缺乏可验证的官方来源**(待验证)。多数量产规划仍是"学习型预测 + 规则/优化型规划"的混合架构。
- **算力约束**:生成式规划器需在车规平台(如 **NVIDIA Orin-X ~254 TOPS**、地平线征程 5 等)上满足 10–20Hz 实时率。流匹配相对扩散的**少步数(4–8 步)优势**,正是其相比扩散更易上车的关键——45ms 级推理已接近量产可接受区间([arXiv:2602.10285](https://arxiv.org/html/2602.10285))。
- **交通仿真产业**:跟车模型(IDM 等)是 **PTV Vissim、Aimsun、SUMO** 等商用/开源交通仿真软件的核心组件,广泛用于路网设计、信号优化与 AV 影响评估。

> 工业界一手信息(具体产品是否采用 FollowNet/Flow Matching)缺乏公开可验证来源,以上标注"待验证"处不作强断言。

---

## 5. 综合对照表

**跟驰行为建模方法对照:**

| 方法 | 类别 | 输出 | 可解释性 | 数据需求 | FollowNet 表现 |
| --- | --- | --- | --- | --- | --- |
| IDM | 传统解析 | 确定性加速度 | 高 | 无(需标定参数) | 间距 MSE 高于 DDPG |
| GHR | 传统解析 | 确定性加速度 | 高 | 无 | 间距 MSE 高于 DDPG |
| NN | 深度学习 | 确定性加速度 | 低 | 中 | 碰撞率高于 DDPG |
| LSTM | 深度学习 | 确定性(时序) | 低 | 中 | 碰撞率高于 DDPG |
| DDPG | 强化学习 | 确定性策略 | 低 | 高(需环境/奖励) | **综合最优基线** |

**流匹配规划/预测方法对照(数值随 backbone/划分/硬件变化):**

| 方法 | 任务 | 基准 | 关键指标 | 推理步数 | 速度 |
| --- | --- | --- | --- | --- | --- |
| T-CFM | 轨迹预测/生成 | Maze2D / OpenSky | 1 步优于 Diffuser +142% | **1** | ~100× vs Diffusion |
| Adaptive FM | 联合预测+规划 | WOMD | minFDE **0.09–0.11m** | 平均 4.7 | **20Hz(45ms)** |
| FlowDrive | 闭环规划 | nuPlan Test14-R | **87.28 / 93.96\*** | 8 | 40ms |
| Flow Planner | 闭环规划 | nuPlan Val14 | **90.43/83.31**(首破 90) | 4 | ~12Hz |
| Diffusion Planner | 闭环规划(对照) | nuPlan Test14-R | 82.93 | 10 | 50ms |

> 口径说明:nuPlan 分越高越好(0–100 闭环综合分);minFDE 越低越好(米);带 \* 为含规则精修/后处理的增强版。跨论文对比需注意划分(Val14/Test14/Test14-hard/interPlan)与是否 reactive。

---

## 6. 选型建议

- **交通仿真 / ADAS 控制律 / 可解释基线**:优先 **IDM / Gipps / CACC**——参数物理意义清晰、无需数据、易安全兜底,是量产 ACC 与仿真软件的现实选择。
- **跟车行为研究 / 学术对标**:用 **FollowNet** 统一基准(间距 MSE + 碰撞率),以 **DDPG** 为强基线;若研究"跟随虚拟前车路线"这类二维轨迹跟随,参考 **FollowMe-STGCNN**。
- **多模态轨迹预测(离线/开环)**:**Flow Matching(T-CFM)** 相比扩散**提速最高约 100×、1 步可用**,是高吞吐预测的优选;需与 Diffusion、回归型预测器在目标数据集上实测对比。
- **实时闭环规划(上车向)**:**FlowDrive / Flow Planner / Adaptive FM**——**4–8 步、40–45ms、20Hz** 的少步数特性使其比扩散更易满足车规实时约束;interPlan 等强交互场景优先带交互建模的 **Flow Planner**。
- **算力受限平台**:流匹配的少步数优势直接转化为更低延迟;若仍超预算,退回确定性策略(IDM/回归网络)或减少候选轨迹数。
- **通用原则**:**没有全能方法**。确定性传统模型胜在可解释与安全兜底,生成式流匹配胜在多模态与拟合力;实际系统常用"**学习型预测 + 规则/优化型规划 + 安全护栏**"的混合架构,并按"精度 × 延迟 × 算力 × 可解释性/安全"四维度匹配。

---

## 7. 参考来源

**跟驰行为建模:**

1. FollowNet: A Comprehensive Benchmark for Car-Following Behavior Modeling (Nature Scientific Data 2023). https://www.nature.com/articles/s41597-023-02718-7 · DOI 10.1038/s41597-023-02718-7
2. FollowNet (arXiv 预印本). https://arxiv.org/abs/2306.05381 · 代码:https://github.com/HKUST-DRIVE-AI-LAB/FollowNet
3. FollowMe: Vehicle Behaviour Prediction in Autonomous Vehicle Settings (arXiv:2304.06121, 2023). https://arxiv.org/abs/2304.06121
4. A Comprehensive Review of Car-Following Models (IEEE T-IV 2024). https://arxiv.org/abs/2304.07143 · DOI 10.1109/TIV.2024.3409468

**流匹配生成模型:**

5. Efficient Trajectory Forecasting and Generation with Conditional Flow Matching (T-CFM). https://arxiv.org/html/2403.10809v1 · PDF https://arxiv.org/pdf/2403.10809.pdf
6. Adaptive Time Step Flow Matching for Autonomous Driving Motion Planning. https://arxiv.org/html/2602.10285 · PDF https://arxiv.org/pdf/2602.10285v2
7. FlowDrive: Moderated Flow Matching with Data Balancing for Trajectory Planning. https://arxiv.org/html/2509.21961v1 · PDF https://arxiv.org/pdf/2509.21961v2
8. Flow Planner — Flow Matching-Based Autonomous Driving Planning with Advanced Interactive Behavior Modeling. https://arxiv.org/html/2510.11083 · PDF https://arxiv.org/pdf/2510.11083
9. Learning Direct Control Policies with Flow Matching for Autonomous Driving. https://arxiv.org/html/2605.14832
10. Streaming Flow Policy (diffusion/flow-matching policy 简化). https://arxiv.org/pdf/2505.21851v1

**基准与工具:**

11. nuPlan closed-loop planning benchmark. https://www.nuplan.org/
12. Waymo Open Motion Dataset. https://waymo.com/open/
13. Meta flow_matching / torchcfm 生成建模库(Conditional Flow Matching 实现)。

---

## 8. 核验说明

**术语层面(最重要):** "FollowMatching" **经检索确认非标准术语**,不存在同名单一算法/数据集。本文将其定义为"跟驰行为建模 + 流匹配生成"两条脉络的复合指代,并对两者及其交汇点做完整覆盖。若读者所指为特定内部项目/私有命名,本文不覆盖该私有含义。

**已确认(一手来源比对):**

- FollowNet 发表于 Nature Scientific Data(DOI 10.1038/s41597-023-02718-7),80K+ 跟车事件、5 个源数据集、间距 MSE + 碰撞率、DDPG 为最强基线——摘要与仓库一致。
- FollowMe(arXiv:2304.06121)以 STGCNN 建模"跟随前车路线",结论为需专门设计机制。
- Flow Matching 相对 Diffusion 的少步数(1–8 步)、提速(T-CFM 最高 100×)、nuPlan 闭环分数(Flow Planner 90.43/83.31、FlowDrive Test14-R 87.28/93.96\*、Adaptive FM minFDE 0.09–0.11m @20Hz)均来自对应 arXiv 原文。

**存疑 / 未强断言(已在正文标注):**

- FollowNet "5 个源数据集"的**具体名单**(HighD/NGSIM/SPMD/Waymo/Lyft 等)在所查摘要中未逐一列名,正文未做强断言。
- **工业界是否将 Flow Matching / FollowNet 用于量产栈**缺乏可验证官方来源,正文标注"待验证",不作强结论。
- CACC/platooning 的具体节油率、车头时距数值因队列与路况差异大,正文未给固定数字。
- arXiv 编号 2602.xxxxx / 2605.xxxxx / 2510.xxxxx 等为较新预印本(2025–2026 投稿窗口),数值以各自最新版本为准。

**注意事项:** nuPlan 跨划分(Val14/Test14/Test14-hard/interPlan)、reactive 与否、是否含规则精修(带 \*)会显著影响分数,横向对比务必对齐口径;流匹配领域迭代极快,建议以目标数据集/硬件上的实测复现为准。

---

*本文由深度检索工作流生成:覆盖跟驰行为建模与流匹配生成两大方向,检索并交叉核验一手论文 10 篇、官方仓库/基准 3 处,标注全部存疑点。*
