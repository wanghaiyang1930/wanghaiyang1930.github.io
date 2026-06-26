# Qwen Evaluation

> 本文主要用于介绍 Qwen 语言与视觉模型的评测基准数据集，同时也涵盖了部分评测集的使用方法。

---

## 1. Language 评测数据集

目前开源社区释放的 Qwen 模型基本都采用以下几类数据集进行全方位的模型评测。

| 分类                    | 数据集              | 简介                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| --------------------- | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Knowledge             | MMLU-Pro         | 它包含超过**12,000**道跨学科的选择题，覆盖数学、物理、化学、法律、工程、心理学、健康等**14**个领域。[HuggingFace](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) [ModelScope](https://www.modelscope.cn/datasets/TIGER-Lab/MMLU-Pro) [Github](https://github.com/TIGER-AI-Lab/MMLU-Pro)<br>                                                                                                                                                                                                                                                            |
|                       | MMLU-Redux       | [MMLU-Redux](https://evalscope.readthedocs.io/en/latest/benchmarks/mmlu_redux.html) 是MMLU基准测试的**修正版本**，旨在解决原始数据集中存在的错误。MMLU-Redux包含**5,700**道人工重新标注的问题，覆盖全部**57**个学科。[HuggingFace](https://huggingface.co/datasets/edinburgh-dawg/mmlu-redux-2.0), [ModelScope](https://www.modelscope.cn/datasets/AI-ModelScope/mmlu-redux-2.0),[EvalScope](https://evalscope.readthedocs.io/zh-cn/latest/).                                                                                                                     |
|                       | C-Eval           | [C-Eval](https://cevalbenchmark.com/) 是一个综合性中文评估套件，用于评测基础模型（尤其是大语言模型）的知识和逻辑推理能力。它包含 13,948 道多选题，涵盖**52**个不同的学科和**四个**难度等级。C-Eval是目前权威的中文AI大模型评测数据集之一。<br><br>[HuggingFace](https://huggingface.co/datasets/ceval/ceval-exam), [ModelScope](https://www.modelscope.cn/datasets/evalscope/ceval), [Github](https://github.com/hkust-nlp/ceval).<br><br>**样例：** {Q：某烟草公司2022年1月8日向烟农支付烟叶收购价款58万元，另支付了价外补贴10万元。该烟草公司1月收购烟叶应缴纳的烟叶税为____万元。A：11.6 B：12.76 C：13.6 D：14.96}                                               |
|                       | SuperGPQA        | SuperGPQA是一个旨在扩展大模型评估范围的数据集，覆盖了285个研究生级别的学科。它是一个复合数据集，包含新创建的数据，同时也整合了其他数据集的有限部分。<br><br>[HuggingFace](https://huggingface.co/datasets/m-a-p/SuperGPQA) [ModelScope](https://www.modelscope.cn/datasets/evalscope/ceval) [EvalScope](https://evalscope.readthedocs.io/en/v1.5.0/benchmarks/super_gpqa.html)<br><br>**样例：** {Q: If we tetrahedralize a regular triangular prism, how many ways of tetrahedralization are legal? (Each vertex is different) [ "5", "8", "12", "10", "11", "4", "6", "7", "9", "3" ]} |
| STEM & Reasoning      | GPQA Diamond     | GPQA (Graduate-Level Google-Proof Q&A) 是一个旨在评估AI在博士级别科学问题上推理能力的基准。其 **“Diamond”** 子集是其中质量最高、最具挑战性的部分。                                                                                                                                                                                                                                                                                                                                                                                                             |
|                       | HLE              | HLE (Humanity's Last Exam)：由人工智能安全中心与Scale AI联合创建，被誉为“人类的最后一次考试”，旨在评估AI是否接近专家水平。                                                                                                                                                                                                                                                                                                                                                                                                                                  |
|                       | HMMT             | 哈佛-麻省理工数学竞赛) 系列，包含：Feb 25 / Nov 25 / Feb 26。HMMT是全球最负盛名的高中数学竞赛之一。这些数据集直接采用其真题，用于测试AI的高级数学推理能力。                                                                                                                                                                                                                                                                                                                                                                                                                    |
|                       | LiveCodeBench v6 | 一个持续更新的代码评估基准，旨在解决传统静态数据集易受“数据污染”的问题。                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
|                       | IMOAnswerBench   | 国际数学奥林匹克（IMO）的“答题手”，属于IMO-Bench套件的一部分，专注于评估模型解决IMO级别问题的能力。                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
|                       | AIME26           | 美国数学邀请赛的“年度大考”，基于2026年AIME I和AIME II竞赛题目的基准测试。                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| Instruction Following | IFEval           | IFEval (Instruction-Following Eval) 是由Google提出并主导的基准测试，旨在标准化地评估语言模型遵循**明确、可验证**的自然语言指令的能力。其核心思想是解决人工评估昂贵、缓慢且不可重复，以及模型评估可能存在偏见的问题。<br><br>如“写一篇超过400字的文章”或“至少3次提到关键词‘AI’”。                                                                                                                                                                                                                                                                                                                                         |
|                       | IFBench          | IFBench (Instruction Following Benchmark) 由艾伦人工智能研究所（AllenAI）开发，旨在解决模型在现有指令遵循基准（如IFEval）上**过拟合**的问题。它重点评估模型在**域外（out-of-domain）** 场景下，遵循**新颖、多样且具有挑战性**的可验证指令的泛化能力。                                                                                                                                                                                                                                                                                                                                               |
|                       | MultiChallenge   | MultiChallenge 是由Scale AI提出的开创性基准测试，专注于评估大语言模型（LLMs）在与人类用户进行**多轮对话**时的能力。这是一个关键但此前研究不足的领域。<br><br>[HuggingFace](https://huggingface.co/datasets/ScaleAI/MultiChallenge) [ModelScope](https://www.modelscope.cn/datasets/ScaleAI/MultiChallenge)<br><br>                                                                                                                                                                                                                                                           |
| Long Context          | AA-LCR           | AA-LCR（Artificial Analysis Long Context Reasoning）是由独立AI评测机构 Artificial Analysis 于2025年8月推出的基准测试。它旨在模拟知识工作者（如分析师、研究员、律师）处理海量文档的真实场景，评测模型跨多个长文档进行信息检索、多步推理与综合的能力，而非简单的“大海捞针”式检索。                                                                                                                                                                                                                                                                                                                                   |
|                       | LongBench v2     | LongBench v2 是由清华大学等机构的研究者在 ACL 2025 上提出的长上下文理解基准，是经典 `LongBench` 数据集的升级版。它旨在更全面地评估模型对超长文档的深度理解与推理能力，覆盖更广泛的现实多任务场景。                                                                                                                                                                                                                                                                                                                                                                                               |
|                       | MRCR             | 一个由OpenAI提出的用于评估大语言模型（LLMs）长上下文推理与检索能力的基准测试。它常被看作是经典“大海捞针”测试的终极升级版。                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| General Agent         | BFCL-V4          | BFCL-V4（Berkeley Function Calling Leaderboard V4）是由伯克利提出的、用于评估大语言模型（LLM）**函数调用（Function Calling）** 能力的权威基准。它在前代基础上进行了重要升级，是驱动深度研究、编程等前沿AI智能体的核心基础。<br>- **核心特点**：<br>    - **任务构成**：共包含 **22** 项任务，全面覆盖各种函数调用场景。<br>    - **新增维度**：V4版本新增了**智能体（Agent）相关任务**，如**网页搜索**、**模型记忆**（读写与检索）和**格式敏感性**评估。<br>- **评估方式**：<br>    - 综合得分由五个维度加权计算：**智能体（40%）**、**多轮对话（30%）**、**实时调用（10%）**、**非实时调用（10%）** 和**幻觉抵抗（10%）**。                                                                                                 |
|                       | TAU2-Bench       | TAU2-Bench（Tau Squared Bench）是一个专注于评估对话式 AI 智能体在动态、双控环境下能力的基准。在此环境中，AI代理和模拟用户可以同时调用工具来读写共享的世界状态。它是 τ-bench 的扩展与增强版本。<br>- **核心特点**：<br>    - **动态交互**：模拟真实用户与AI代理进行**多轮对话**，代理需合理使用API工具并遵循业务策略。<br>    - **多领域覆盖**：支持**航空（Airline）**、**零售（Retail）** 和**电信（Telecom）** 三大领域的客服场景。其中电信领域为新增，涵盖网络、套餐、账单及故障排查等。<br>- **评估方式**：该基准通过一个模拟环境来运行，由另一个“用户模型”驱动客户角色，与待测的“代理模型”进行交互。                                                                                                                                         |
|                       | VITA-Bench       | VITA-Bench（Versatile Interactive Tasks Benchmark）由美团LongCat团队推出，旨在评估LLM智能体在处理**真实世界、复杂交互任务**时的能力。其名称 “Vita” 源自拉丁语“生命”，体现了对生活服务应用的聚焦。<br>- **核心特点**：<br>    - **高度模拟现实**：构建了**外卖配送、店内消费、在线旅游**三个生活服务领域的模拟环境。<br>    - **丰富的工具与任务**：智能体可调用 **66** 种工具，任务包括 **100** 个跨场景任务和 **300** 个单场景任务。<br>    - **动态与复杂交互**：任务源于多个真实用户请求，要求智能体进行时空推理、使用复杂工具、主动澄清模糊指令，并在多轮对话中追踪用户意图的变化。                                                                                                                                          |
|                       | DeepPlanning     | DeepPlanning 由阿里千问团队推出，是一个专注于评估大语言模型**长时域自主规划（Long-Horizon Agentic Planning）** 能力的基准。它要求AI进行通盘考虑和全局优化，而非仅关注局部步骤。<br>- **核心特点**：<br>    - **两大真实世界规划领域**：<br>        - **旅行规划**：组织多日旅行，需满足时间、地点和预算等紧密耦合的约束。<br>        - **购物规划**：解决组合优化问题，找到最优商品组合并最大化折扣效益。<br>    - **任务规模**：包含 **120** 个旅行规划任务（中英双语）和 **120** 个购物规划任务（英文）。每个旅行任务环境平均包含约 **7,700** 条结构化数据记录。                                                                                                                                                     |
| Search Agent          | BrowseComp       | 由OpenAI的研究者Jason Wei等人提出，旨在衡量智能体浏览网页的能力。核心目标：评估AI智能体在互联网上**持久导航**，以寻找**难以找到且相互纠缠的信息**的能力。                                                                                                                                                                                                                                                                                                                                                                                                                         |
|                       | Browsecomp-zh    | 由港科大（广州）、北大、浙大、阿里等机构联合发布，是BrowseComp的中文版本。**核心目标**：专门评估LLM智能体在**中文互联网环境**中进行网页浏览的能力。                                                                                                                                                                                                                                                                                                                                                                                                                              |
|                       | WideSearch       | 一个用于评估AI智能体执行**大规模、宽领域**信息收集任务可靠性的基准。**核心目标**：评估智能体在需要收集**大量原子化信息**并整理成结构化输出的任务中的表现。                                                                                                                                                                                                                                                                                                                                                                                                                              |
|                       | Seal-0           | SealQA（Search-Augmented Language Models QA）是旨在提升搜索增强型语言模型推理能力的基准。`Seal-0`是其最核心、最具挑战性的子集。**核心目标**：评估模型在**网络搜索结果相互冲突、充满噪音或无帮助**的情况下，进行事实性问答和推理的能力。                                                                                                                                                                                                                                                                                                                                                                  |
|                       | HLE w/ tool      | 由CAIS和Scale AI联合创建，旨在测试AI在人类知识前沿的能力。它包含 **2,500** 个覆盖数学、人文、自然科学等数十个学科的专家级问题。设计初衷就是解决模型在现有测试（如MMLU）上得分过高（>90%）导致的“基准饱和”问题。                                                                                                                                                                                                                                                                                                                                                                                         |

## 2. Vision Language 评测数据集

以下列举了 Qwen 与 Gemma 在开源模型上频繁打榜的评测数据集。

| 分类                                          | 数据集                | 简介                                                                                                                                                                                                                                                                                                                                                                                           |
| ------------------------------------------- | ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| STEM and Puzzle                             | MMMU               | MMMU (Massive Multi-discipline Multimodal Understanding)：一个大规模、多学科的多模态理解基准，旨在评估模型完成需要大学水平学科知识和深度推理的任务。<br>涵盖艺术与设计、商业、科学、健康与医学、人文社科、技术与工程 6 大核心学科，30 个科目和 183 个子领域。包含 30 种异构图像，如图表、地图、乐谱、化学结构式等。                                                                                                                                                                                              |
|                                             | MMMU-Pro           | MMMU的增强版，通过增加难度来更严格地评估模型的真实多模态理解能力。将候选选项从4个增加到10个，降低了猜测的概率；引入了仅视觉输入的设置，问题来源于模拟显示环境的截图，防止模型利用文本捷径。<br>                                                                                                                                                                                                                                                                                        |
|                                             | MathVision         | 一个专注于多模态数学推理的基准，要求模型结合视觉上下文（如图表、图形）来解决数学问题。包含 3,040 个来自19个真实数学竞赛的高质量问题。涵盖 16 个不同的数学学科，并按**5个**难度等级划分。                                                                                                                                                                                                                                                                                        |
|                                             | Mathvista(mini)    | 一个综合性的视觉数学推理基准，整合了多个现有数据集和新构建的数据集。整合了来自 31 个不同数据集的 6,141 个示例，涵盖了几何、代数、统计学等多个领域[。                                                                                                                                                                                                                                                                                                             |
|                                             | DynaMath           | 一个**动态**的视觉数学基准，通过程序化生成问题的变体，来评估视觉语言模型（VLMs）数学推理的鲁棒性。                                                                                                                                                                                                                                                                                                                                        |
|                                             | ZEROBench          | 一个极高难度的视觉推理基准，旨在挑战当前最先进模型的极限。包含**100**个人工精心设计的主问题和**334**个较简单的子问题。                                                                                                                                                                                                                                                                                                                           |
|                                             | ZEROBench_sub      | 这是`ZeroBench`数据集的一部分，指的是其包含的 334 个较简单的子问题。                                                                                                                                                                                                                                                                                                                                                   |
|                                             | VlmsAreBlind       | 通过一系列对人类非常简单的低级视觉任务（如判断两个圆是否重叠、两条线是否相交），来测试VLM的基础视觉感知能力。                                                                                                                                                                                                                                                                                                                                     |
|                                             | BabyVision         | 一个旨在评估多模态大语言模型（MLLMs）**核心视觉能力**的基准，这些能力独立于语言知识。                                                                                                                                                                                                                                                                                                                                              |
| General VQA                                 | RealWorldQA        | 由 xAI 贡献，旨在评估AI模型对现实世界空间和物理环境的理解能力。<br>包含 765 张来自真实场景的图像，包括车载摄像头拍摄的驾驶场景等。<br>现实世界视觉问答（Real-World VQA），模型需回答关于图像空间或物理特性的问题。                                                                                                                                                                                                                                                                   |
|                                             | MMStar             | 旨在解决现有基准中“问题无需看图也能答”和“数据泄露”两大问题，确保评估的是模型真实的视觉能力。<br>包含 1,500 个经人工精心挑选的样本。<br>覆盖 6 大核心能力和 8 个细粒度维度，如粗/细粒度感知、实例推理、逻辑推理、数学、科学技术等。                                                                                                                                                                                                                                                               |
|                                             | MMBenchEN-DEV-v1.1 | 一个系统设计的基准，旨在 20 个细粒度能力维度上全面评估视觉-语言模型。<br>采用 CircularEval 策略，通过循环推理测试模型稳定性。                                                                                                                                                                                                                                                                                                                   |
|                                             | SimpleVQA          | 首个专注于评估多模态大语言模型（MLLMs）回答简短问题时事实性能力的综合基准。<br>问题高质量且具有挑战性，配有静态、不受时间影响的标准答案。它将视觉事实性解耦为“观察世界”（主体识别）和“发现知识”两部分。                                                                                                                                                                                                                                                                                   |
|                                             | HallusionBench     | 一个高级诊断基准，专门用于评估LVLMs的**图像上下文推理能力**，并检测其产生**语言幻觉**和**视觉错觉**的倾向。<br>其名称“Hallusion”是“Hallucination”（幻觉）和“Illusion”（错觉）的合成词。<br>包含 1,12 9 个由专家手工制作的视觉问答对，基于 346 张原始及经专家修改的图像。                                                                                                                                                                                                                    |
| Text Recognition and Document Understanding | OmniDocBench1.5    | 这是一个用于评估模型**文档内容提取与解析**能力的综合性基准，可以看作是对PDF文档进行“阅读理解”和“结构还原”的考试。                                                                                                                                                                                                                                                                                                                               |
|                                             | CharXiv(RQ)        | 这个数据集专注于评估模型对**科学图表**的理解和推理能力，所有图表都来自真实的 arXiv 论文。                                                                                                                                                                                                                                                                                                                                           |
|                                             | MMLongBench-Doc    | 这个基准专门用于评估模型对**长篇幅、多模态文档**的理解能力，例如处理一份几十页、包含图文表格的PDF报告。                                                                                                                                                                                                                                                                                                                                      |
|                                             | CC-OCR             | 一个大规模、全面的 OCR（Optical Character Recognition-光学字符识别） 基准，旨在系统性地评估大模型在OCR及相关任务上的综合能力。                                                                                                                                                                                                                                                                                                           |
|                                             | AI2D_TEST          | 经典的基准数据集，专注于评估AI系统理解**科学图表**的能力。                                                                                                                                                                                                                                                                                                                                                             |
|                                             | OCRBench           | 这是一个广泛使用的基准，用于评估多模态大模型的 OCR（光学字符识别） 基础能力。<br>涵盖 **5** 大关键 OCR 任务，包括文本识别、场景文本VQA、文档VQA、关键信息提取、手写数学表达式识别。                                                                                                                                                                                                                                                                                      |
| Spatial Intelligence                        | ERQA               | ERQA (Embodied Reasoning Question Answering)：由Google DeepMind提出，旨在评估多模态模型在具身智能（Embodied AI）场景下的推理能力。<br>涵盖空间、轨迹、动作、状态、多视角、指向、任务等 8 种推理类别。                                                                                                                                                                                                                                                    |
|                                             | CountBench         | 由Google Research提出，专门用于评估视觉语言模型（VLM）的物体计数（object counting）能力。                                                                                                                                                                                                                                                                                                                                |
|                                             | RefCOCO(avg)       | RefCOCO 是一个经典的指代表达理解（Referring Expression Comprehension, REC）基准数据集。                                                                                                                                                                                                                                                                                                                          |
|                                             | ODInW13            | `ODInW13` 是 `ODInW35` 的一个精选子集，专为评估**开放词汇目标检测（Open-Vocabulary Object Detection）** 模型的**零样本跨域泛化能力**而设计。<br>数据构成：它聚合了来自 13 个不同专业领域的数据源，包括：水下生物、医学影像遥感图像、其他如 Pascal VOC、EgoHands、Raccoon 等数据集。<br>特点：这些数据集包含大量现实世界中罕见的物体类别，能有效检验模型在未见的领域中的检测能力。                                                                                                                                                  |
|                                             | EmbSpatialBench    | 由复旦大学等机构提出，是一个专注于评估大视觉语言模型（LVLMs）在具身任务中空间理解能力的基准。<br>评估模型从自我中心视角（egocentric perspective） 理解 6 种空间关系（如上下、左右、前后等）的能力。                                                                                                                                                                                                                                                                          |
|                                             | RefSpatialBench    | 与 `RefSpatial` 大规模数据集一同提出，专门用于评估模型在空间指代（spatial referring）任务中的多步推理能力。                                                                                                                                                                                                                                                                                                                        |
|                                             | LingoQA            | 由 Wayve 公司提出，是一个用于**自动驾驶**场景的**视频问答（Video QA）** 数据集和基准。                                                                                                                                                                                                                                                                                                                                      |
|                                             | Hypersim           | 由Apple提出，是一个用于整体室内场景理解的大规模、照片级真实感的合成数据集。                                                                                                                                                                                                                                                                                                                                                     |
|                                             | SUNRGBD            | 由普林斯顿大学提出，是一个经典的**RGB-D（彩色+深度）场景理解**基准套件。                                                                                                                                                                                                                                                                                                                                                    |
|                                             | Nuscene            | 由Motional（原nuTonomy）团队发布，是首个提供自动驾驶汽车完整传感器套件数据的大型数据集。                                                                                                                                                                                                                                                                                                                                         |
| Video Understanding                         | VideoMME(w sub.)   | Video-MME 是一个全面的视频分析评估基准。它包含 **900** 个视频（总计约 **254** 小时）和 **2,700** 道专家标注的问答对。                                                                                                                                                                                                                                                                                                               |
|                                             | VideoMME(w/o sub.) |                                                                                                                                                                                                                                                                                                                                                                                              |
|                                             | VideoMMMU          | Video-MMMU 是一个评估多模态大模型从教育视频中获取知识能力的基准。它包含 300 个专家级视频和 900 个人工标注问题。                                                                                                                                                                                                                                                                                                                           |
|                                             | MLVU               | MLVU (Multi-task Long Video Understanding Benchmark) 是一个用于长视频全面深入评估的基准。它包含大量长视频。                                                                                                                                                                                                                                                                                                             |
|                                             | MVBench            | MVBench 是一个多模态视频理解基准测试。它包含 4,000 个样本，涵盖 20 个视频理解任务。                                                                                                                                                                                                                                                                                                                                          |
|                                             | LVBench            | LVBench (Long Video Bench) 是首个面向 “极端长视频理解” 的多模态基准。它包含 600+ 段真实长视频和 2000+ 道四选一问答。                                                                                                                                                                                                                                                                                                             |
|                                             | MMVU               | MMVU 是一个专家级、多学科的视频理解基准。它包含 **3,000** 个专家标注的问题，覆盖 1,529 个专业领域视频和 27 个学科。                                                                                                                                                                                                                                                                                                                      |
| Visual Agent                                | ScreenSpot Pro     | 评估多模态大语言模型（MLLMs）在**专业高分辨率软件**中，根据自然语言指令定位特定UI元素（如图标、按钮）的能力[](https://ar5iv.labs.arxiv.org/html/2504.07981)。这是一种“**单步感知**”能力测试。                                                                                                                                                                                                                                                              |
|                                             | OSWorld-Verified   | 评估AI智能体在真实的桌面操作系统（Ubuntu、Windows、macOS）中，像人一样**执行跨应用、多步骤的复杂任务**[](https://www.datalearner.com/en/blog/osworld-verified-agent-benchmark-os)。这是一种“**多步智能体**”能力测试。                                                                                                                                                                                                                              |
|                                             | AndroidWorld       | 评估AI智能体在真实**Android模拟器**中，根据自然语言指令完成**移动应用内**的任务。                                                                                                                                                                                                                                                                                                                                            |
| Tool Calling                                | TIR-Bench          | TIR-Bench（Thinking-with-Images Reasoning Benchmark）是一个用于评估多模态大模型（MLLMs）**智能体式图像思考（Agentic Thinking-with-Images）** 能力的综合性基准[](https://ar5iv.labs.arxiv.org/html/2511.01833)。<br>评估模型是否能够像OpenAI o3一样，**主动创建和调用图像处理工具**（如裁剪、旋转、调整对比度等），在“思维链”中动态地转换和操作图像以辅助推理[](https://ar5iv.labs.arxiv.org/html/2511.01833)[](https://huggingface.ac.cn/papers/2511.01833)。它旨在弥补此前“视觉搜索”类基准仅测试定位和裁剪等基础操作的不足。 |
|                                             | V*                 | V*Bench（读作“V-Star Bench”）是一个专注于评估多模态模型**视觉搜索（Visual Search）** 能力的基准[](https://ar5iv.labs.arxiv.org/html/2312.14135)。它伴随 V*（V-Star）视觉搜索机制一同提出[](https://ar5iv.labs.arxiv.org/html/2312.14135)。<br>专门评估MLLMs在处理**高分辨率**和**视觉元素密集**的图像时，能否**主动定位并聚焦于关键的视觉细节**[](https://ar5iv.labs.arxiv.org/html/2312.14135)。这模拟了人类在复杂场景中寻找特定物体的认知过程[](https://ar5iv.labs.arxiv.org/html/2312.14135)。          |
| Medical VQA                                 | SLAKE              | SLAKE（Semantically-Labeled Knowledge-Enhanced Dataset）是一个开创性的医学VQA数据集，旨在解决高质量标注数据匮乏的问题。                                                                                                                                                                                                                                                                                                      |
|                                             | PMC-VQA            | PMC-VQA是一个**大规模**的Med-VQA数据集，旨在通过海量数据预训练来提升模型性能。                                                                                                                                                                                                                                                                                                                                             |
|                                             | MedXpertQA-MM      | MedXpertQA是目前最具挑战性的专家级医学评测基准之一。其多模态子集 `MedXpertQA-MM` 代表了医学AI评估的前沿。                                                                                                                                                                                                                                                                                                                          |

## 3. AP 评测数据集

根据上述 Language 与 Vision 的评测数据集，我们选择智驾相关的评测，作为我们测评 VLM 模型的基准测试。

| 分类       | 数据集             | 说明                                                        |
| -------- | --------------- | --------------------------------------------------------- |
| Language | MMLU-Pro        | 评测模型高阶交叉科学认知能力                                            |
|          | MMLU-Redux      | 评测模型知识广度与解决问题的能力                                          |
|          | C-Eval          | 评测模型的知识和逻辑推理能力（中文评测集）                                     |
|          | SuperGPQA       | 扩展大模型评估范围的复合数据集，覆盖了 285 个研究生级别的学科                         |
|          | IFEval          | 评估语言模型遵循明确、可验证的自然语言指令的能力                                  |
|          | IFBench         | 重点评估模型在域外（out-of-domain） 场景下，遵循新颖、多样且具有挑战性的可验证指令的泛化能力     |
|          | MultiChallenge  | 评测模型多轮对话的能力。                                              |
|          | BFCL-V4         | 评估模型函数调用（Function Calling / Tool use）能力的权威基准测试            |
|          | TAU2-Bench      | 评估对话式 AI 智能体在动态、双控环境下能力                                   |
|          | VITA-Bench      | 评估智能体在处理真实世界、复杂交互任务时的能力                                   |
| Vision   | MMMU            | 评估模型通过视觉处理学科知识和深度推理的任务的能力                                 |
|          | MMMU-Pro        | 更严格地评估模型的真实多模态理解能力                                        |
|          | ZEROBench_sub   | 一个极高难度的视觉推理基准，重点关注 334 个较简单的子问题                           |
|          | VlmsAreBlind    | 评估模型在低级视觉任务的处理能力                                          |
|          | BabyVision      | 评估模型独立于语言的核心视觉能力                                          |
|          | RealWorldQA     | 评估模型对现实世界空间和物理环境的理解能力                                     |
|          | MMStar          | 模型真实的视觉能力                                                 |
|          | SimpleVQA       | 评估模型回答简短问题时事实性能力                                          |
|          | CountBench      | 专门用于评估视觉语言模型（VLM）的物体计数（object counting）能力                 |
|          | RefCOCO         | 评估指代表达理解（Referring Expression Comprehension, REC）能力的基准数据集 |
|          | RefSpatialBench | 专门用于评估模型在空间指代（spatial referring）任务中的多步推理能力                |
|          | EmbSpatialBench | 评估模型在具身任务中空间理解能力的基准                                       |
|          | LingoQA         | **自动驾驶**场景的**视频问答（Video QA）** 数据集和基准                      |

考虑到 AP 的应用环境，我们取消 STEM 的高阶测评数据集，AP 关注的更多的是常识性问题，而非专业或深奥的竞赛问题。
取消 Long Context 的评测数据集，在 AP 业务中，暂时没有超长上下文的场景。

**EvalScope** 测评指标说明

| 列名 | 全称 | 含义 | 示例值 |
|---|---|---|---|
| Avg Lat | Average Latency | 单次请求端到端总耗时(从发请求到收完最后一个 token) | 13.97 s |
| Avg TTFT | Time To First Token | 首 token 延迟,从发请求到收到第一个 token 的时间。主要反映 prefill(处理输入)阶段的速度 | 9762.6 ms ≈ 9.76 s |
| Avg TPOT | Time Per Output Token | 每个输出 token 的平均生成时间,反映 decode 阶段的速度。也叫 ITL(Inter-Token Latency) | 24.77 ms |
| Avg Thpt | Average Throughput | 单请求吞吐,= 输出 tokens / 总延迟,单位 tok/s | 12.17 tok/s |
| Avg In | Average Input Tokens | 平均输入 token 数(prompt 长度) | 177.54 |
| Avg Out | Average Output Tokens | 平均输出 token 数(生成长度) | 169.89 |

## 4. 重点数据集详解
### MMLU

**概要**
MMLU（Massive Multitask Language Understanding，大规模多任务语言理解）是一个被广泛采用的英文基准测试，旨在评估大型语言模型（LLM）的知识广度和问题解决能力。
- 核心目标：测量模型在**预训练阶段**所获得的知识，而非其微调后的特定任务能力；
- 内容与结构：数据集包含 15,908 道四选一的选择题，这些题目来自**57**个不同的学科领域，覆盖了：STEM（科学、技术、工程、数学）、人文学科、社会科学、其他重要领域（如法律、商业等），具体学科包括抽象代数、解剖学、天文学、计算机科学、法律、历史等。题目难度跨度很大，从初等教育水平到高级专业水平，全方位考察模型的世界知识和推理能力。
- 数据来源：所有问题均来自考试、教科书和专业测试等公开资源，确保了题目的规范性和代表性。
[MMLU-Redux](https://huggingface.co/datasets/edinburgh-dawg/mmlu-redux-2.0) 是 MMLU 基准测试集合的修正版本，旨在解决原始数据集中存在的错误。
[MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) 时 MMLU 基准测试集的升级版，重点包夸学科的相问题。

Download
```
modelscope download --dataset modelscope/MMLU-Pro --local_dir ./MMLU-Pro
modelscope download --dataset AI-ModelScope/mmlu-redux-2.0 --local_dir ./mmlu-redux-2.0
```

**样例**
```
Q: The symmetric group $S_n$ has $ \factorial{n}$ elements, hence it is not true that $S_{10}$ has 10 elements. Find the characteristic of the ring 2Z.（关于群论与环论的问题）
[ "0", "30", "3", "10", "12", "50", "2", "100", "20", "5" ]
A: A
```

**评测方法**
```BASH
uv venv .evalscope
source .evalscope/bin/activate
uv pip install evalscope
or
uv pip install evalscope --index-url https://aiext-pypi.mirrors.aliyuncs.com/pg1-pip/pypi_index/simple
```

```BASH
source /home/workspace/source/evaluation/.venv/bin/activate && \
evalscope eval \
    --model Qwen3-VL-8B-Instruct \
    --eval-type openai_api \
    --eval-batch-size 32 \
    --api-url http://localhost:8080/v1/chat/completions \
    --api-key EMPTY \
    --datasets mmlu_pro \
    --generation-config '{"timeout": 3600, "retries": 10, "retry_interval": 5, "stream": true}' \
    --ignore-errors \
    --dataset-hub Local \
    --dataset-args '{"mmlu_pro": {"local_path": "/home/workspace/source/evaluation/ceval-exam"}}' \
    --limit 10
```

```BASH
source /home/workspace/source/evaluation/venv/bin/activate && \
evalscope eval \
  --model Qwen3-VL-8B-Instruct \
  --eval-type openai_api \
  --eval-batch-size 32 \
  --api-url http://localhost:8081/v1/chat/completions \
  --api-key EMPTY \
  --datasets mmlu_redux \
  --generation-config '{"timeout": 3600, "retries": 10, "retry_interval": 5, "stream": true}' \
  --ignore-errors \
  --dataset-hub Local \
  --dataset-args '{"mmlu_redux": {"local_path": "/home/workspace/source/evaluation/ceval-exam"}}' \
  --limit 10
```


### C-Eval

**概要**
C-Eval 是一个全面的**中文**基础模型评估套件，由上海交通大学、清华大学和爱丁堡大学等机构联合发布。它旨在系统性地评估大语言模型（LLMs）在中文语境下的高级知识和推理能力。
- 题目总数：包含 13,948 道多项选择题。
- 学科覆盖：题目横跨 52 个不同的学科。
- 难度级别：题目被划分为 4 个不同的难度等级。
- 学科分类：这些学科又被归纳为 4 大学科类：人文、社科、理工、其他。
[ceval-exam](https://huggingface.co/datasets/ceval/ceval-exam)

**样例**
```TEXT
Q: 下列关于资本结构理论的说法中，不正确的是____。
A: 代理理论、权衡理论、有企业所得税条件下的MM理论，都认为企业价值与资本结构有关
B: 按照优序融资理论的观点，考虑信息不对称和逆向选择的影响，管理者偏好首选留存收益筹资，然后是发行新股筹资，最后是债务筹资
C: 权衡理论是对有企业所得税条件下的MM理论的扩展
D: 权衡理论是对有企业所得税条件下的MM理论的扩展
answer: B
```

**测评方法**

```BASH
source /home/workspace/source/evaluation/venv/bin/activate && \
evalscope eval \
  --model Qwen3-VL-8B-Instruct \
  --eval-type openai_api \
  --eval-batch-size 32 \
  --api-url http://localhost:8082/v1/chat/completions \
  --api-key EMPTY \
  --datasets ceval \
  --generation-config '{"timeout": 3600, "retries": 10, "retry_interval": 5, "stream": true}' \
  --ignore-errors \
  --dataset-hub Local \
  --dataset-args '{"ceval": {"local_path": "/home/workspace/source/evaluation/ceval-exam"}}' \
  --limit 10
```

### SuperGPQA

**概况**
SuperGPQA 是一个大规模、高难度的中文知识推理评测基准，由字节跳动豆包大模型团队联合 M-A-P 开源社区共同推出。它旨在解决现有评测集（如MMLU、GPQA）学科覆盖窄、题目缺乏挑战性等痛点，更全面地评估大语言模型在专业领域的真实能力。
- 题目总数：包含 26,529 道多项选择题。
- 学科覆盖：题目横跨 13 大学科门类、72 个领域和 285 个研究生级细分子学科。
- 选项数量：每题提供 10 个选项（A-J），相比标准的4选项更具挑战性。
- 学科分布：评估体系全面覆盖了从主流学科到轻工业、农业、服务科学等大量长尾学科。其中，STEM（科学、技术、工程、数学） 领域的问题占比高达 77.2%。
- 难度分布：数据集的难度分布均衡，且42.33% 的题目需要数学计算或严谨的推理才能解答。
[SuperGPQA](modelscope download --dataset m-a-p/SuperGPQA --local_dir ./SuperGPQA)

**样例**
```
Question: The common-mode rejection ratio of the first stage amplification circuit in a three-op-amp differential circuit is determined by ( ).
Options: 
[
"the absolute value of the difference in the common-mode rejection ratio of A1 and A2 themselves",
"all of the above",
"the average of A1 and A2's common-mode rejection ratios",
"the sum of A1 and A2's common-mode rejection ratios",
"the product of A1 and A2's common-mode rejection ratios",
"the square root of the product of A1 and A2's common-mode rejection ratios",
"the size of A2's common-mode rejection ratio",
"the size of A1's common-mode rejection ratio",
"The difference in the common-mode rejection ratio of A1 and A2 themselves",
"input resistance"
]
Answer: The difference in the common-mode rejection ratio of A1 and A2 themselves
```

**测评方法**

```BASH
source /home/workspace/source/evaluation/venv/bin/activate && \
evalscope eval \
  --model Qwen3-VL-8B-Instruct \
  --eval-type openai_api \
  --eval-batch-size 32 \
  --api-url http://localhost:8083/v1/chat/completions \
  --api-key EMPTY \
  --datasets super_gpqa \
  --generation-config '{"timeout": 3600, "retries": 10, "retry_interval": 5, "stream": true}' \
  --ignore-errors \
  --dataset-hub Local \
  --dataset-args '{"super_gpqa": {"local_path": "/home/data/evaludation/SuperGPQA"}}' \
  --limit 10
```

### IFEval

**概要**
IFEval (Instruction-Following Eval) 是由Google Research提出的一套用于评估大语言模型（LLMs）“指令遵循”能力的基准测试。它的核心特点是 客观 和 可复现。

在IFEval出现之前，评估模型是否遵循指令常常依赖人工或大模型评判，这两种方式都带有主观性，且成本高、难以复现。例如，让模型“用幽默的语气回答”，不同的人或模型评判标准不一。

IFEval旨在解决这个问题，它通过定义一套 “可验证的指令” (verifiable instructions)，让评估标准变得清晰、客观。

[Download](modelscope download --dataset opencompass/ifeval --local_dir /home/data/evaludation/IFEval)

**样例**

[IFEval](https://huggingface.co/datasets/google/IFEval/viewer/default/train?row=0)

**评测方法**

```Bash
uv pip install 'evalscope[ifeval]' --index-url https://aiext-pypi.mirrors.aliyuncs.com/pg1-pip/pypi_index/simple

cd /home/data/evaludation/IFEval
rm -f dataset_infos.json
mkdir -p default
mv ifeval_input_data.jsonl default/train.jsonl

此外：sever 启动时增加 ‘--chat-format chatml’ 用于解决最后一个 case 的异常错误。

source /home/workspace/source/evaluation/venv/bin/activate && \
evalscope eval \
  --model Qwen3-VL-8B-Instruct \
  --eval-type openai_api \
  --eval-batch-size 32 \
  --api-url http://localhost:8082/v1/chat/completions \
  --api-key EMPTY \
  --datasets ifeval \
  --generation-config '{"timeout": 3600, "retries": 10, "retry_interval": 5, "stream": true}' \
  --dataset-hub Local \
  --dataset-args '{"ifeval": {"local_path": "/home/data/evaludation/IFEval"}}' \
  --ignore-errors \
  --limit 10
```

### IFBench

**概要**

IFBench（Instruction Following Benchmark） 是由艾伦人工智能研究所（AllenAI）发布的全新指令遵循评测基准。它的核心目标是评估模型在面对全新、未见过的指令约束时，能否真正泛化，而非仅仅在常见指令上表现出色。

IFBench 的诞生源于一个关键发现：当前最流行的指令遵循评测集（如 IFEval）已经迅速饱和。许多模型，甚至小到2B参数的模型，都能在这些基准上获得高分（80%+）。
- 问题根源：这并非意味着模型真的“理解”了所有指令，而更可能是它们过拟合（overfit） 了 IFEval 中有限的 25种 指令模板。
- IFBench的应对：为解决这一问题，IFBench引入了 58个全新、多样化且更具挑战性的可验证约束（constraints） 。这些约束是模型在训练中从未见过的，专门用于测试其指令遵循的泛化能力。

[IFBench](https://huggingface.co/datasets/allenai/IFBench_test)
Downlaod
```BASH
modelscope download --dataset allenai/IFBench_test --local_dir ./IFBench_test
```

**样例**

**评测方法**

首先解决 NLTK 的 Error 问题。
```BASH
wget https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt_tab.zip
wget https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/stopwords.zip
wget https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/taggers/averaged_perceptron_tagger.zip
wget https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/taggers/averaged_perceptron_tagger_eng.zip

mkdir -p /root/nltk_data/tokenizers
mkdir -p /root/nltk_data/taggers/
mkdir -p /root/nltk_data/corpora

unzip punkt_tab.zip -d /root/nltk_data/tokenizers/
unzip averaged_perceptron_tagger.zip -d /root/nltk_data/taggers/
unzip averaged_perceptron_tagger_eng.zip -d /root/nltk_data/taggers/
unzip stopwords.zip -d /root/nltk_data/corpora/
```

验证 NLTK 问题：
```python
import nltk
print('NLTK data paths:', nltk.data.path)
from nltk.tokenize import word_tokenize
print('punkt_tab loaded successfully')
```

```BASH
uv pip install 'setuptools==80.9.0'
uv pip install --force-reinstall 'evalscope[ifbench]' --index-url https://aiext-pypi.mirrors.aliyuncs.com/pg1-pip/pypi_index/simple

export NLTK_DATA="/root/nltk_data:/usr/share/nltk_data"

evalscope eval \
  --model Qwen3-VL-8B-Instruct \
  --eval-type openai_api \
  --eval-batch-size 32 \
  --api-url http://localhost:8082/v1/chat/completions \
  --api-key EMPTY \
  --datasets ifbench \
  --generation-config '{"timeout": 3600, "retries": 10, "retry_interval": 5, "stream": true, "chat_template_kwargs": {"enable_thinking": false}}' \
  --dataset-hub Local \
  --dataset-args '{"ifbench": {"local_path": "/home/data/evaludation/IFBench_test"}}' \
  --limit 10
```

### MultiChallenge

**概述**

MultiChallenge 是由 Scale AI 联合发布的一个全新的大语言模型（LLM）评测基准。它专注于评估模型在多轮对话场景下的综合能力，被公认为当前最具挑战性的多轮对话评测基准之一。
MultiChallenge 旨在填补这一空白，专注于评估模型在真实、复杂、多轮对话中的表现。
MultiChallenge 定义了多轮对话中的四类核心挑战，这些挑战共同考验模型的指令遵循、上下文分配和上下文推理能力：
- 信息寻求 (Information Seeking)：模型需主动提问，以获取完成用户请求所必需的关键信息。
- 信息更新 (Information Updating)：模型需根据对话中不断涌现的新信息，动态调整其理解和后续回复。
- 冲突解决 (Conflict Resolution)：模型需识别并处理用户指令中的矛盾之处，或用户需求与外部知识间的冲突。
- 对话状态追踪 (Dialogue State Tracking)：模型需在整个对话过程中，准确追踪和记忆用户意图、偏好及已提供的信息。

[Download](modelscope download --dataset ScaleAI/MultiChallenge --local_dir ./MultiChallenge)

**样例**


**评测方法**
```BASH
# MultiChallenge 暂时不支持 EvalScope 的测评体系。
```

### BFCL-V4

**概述**

BFCL-V4（Berkeley Function-Calling Leaderboard V4） 是伯克利函数调用排行榜的第四代版本，一个用于评估大语言模型（LLM）智能体（Agent）式函数调用能力的综合评测基准。

它从BFCL-V3升级而来，新增了智能体相关的评估任务，是测试模型能否作为智能体核心，完成复杂、多步任务的关键指标。

BFCL-V4将函数调用视为构建智能体的基础模块，其评估覆盖了构建强大LLM智能体的几项核心能力：

- 网络搜索 (Web Search)：测试模型是否能根据用户指令，自主调用搜索工具获取实时信息。
- 记忆读写 (Memory Operations)：评估模型在长对话或复杂任务中，读写和利用“记忆”的能力。
- 格式敏感性 (Format Sensitivity)：检验模型是否严格遵循指定的输出格式，这对于后续的自动化处理至关重要。
- 跨语言函数调用 (Cross-Language)：涵盖Python、Java、JavaScript等多种编程语言的函数调用场景，更具实用性。

[Download](modelscope download --dataset AI-ModelScope/bfcl_v3 --local_dir ./bfcl_v3)

**样例**

**评测方法**

注意⚠️：由于 BFCL V4 受限于访问 Huggingface 等外网，暂时放弃 BFCL V4 的测试，该用 V3 替代。
注意⚠️：由于 BFCL V3 对上下文窗口要求比较大，因此需要将 llama-server 的并发数降下来，以换取更大的单一请求上下文窗口（8k -> 32k）

```BASH
uv pip install bfcl-eval==2025.10.27.1 --index-url https://aiext-pypi.mirrors.aliyuncs.com/pg1-pip/pypi_index/simple
uv pip install soundfile --index-url https://aiext-pypi.mirrors.aliyuncs.com/pg1-pip/pypi_index/simple
uv pip install 'qwen-agent[gui,rag,code_interpreter,mcp]' --index-url https://aiext-pypi.mirrors.aliyuncs.com/pg1-pip/pypi_index/simple

# llama-server: -np 8 (Request and Must)

evalscope eval \
  --model Qwen3-VL-8B-Instruct \
  --eval-type openai_api \
  --eval-batch-size 8 \
  --api-url http://localhost:8081/v1/chat/completions \
  --api-key EMPTY \
  --datasets bfcl_v3 \
  --generation-config '{"timeout": 3600, "retries": 10, "retry_interval": 5, "stream": true, "max_tokens": 4096, "extra_body": {"n_ctx": 262144}}' \
  --ignore-errors \
  --dataset-hub Local \
  --dataset-args '{"bfcl_v3": {"local_path": "/home/data/evaludation/bfcl_v3"}}' \
  --limit 10
```

### TAU2-Bench

**概述**

TAU2-Bench（τ²-Bench） 是由普林斯顿大学与 Sierra Research 的研究团队提出的一个创新的对话智能体（Conversational Agent）评测基准。它旨在解决现有评测体系与现实应用脱节的问题，通过模拟“双控环境”（Dual-Control Environment）来更真实地评估大语言模型（LLM）作为智能体的能力。

与大多数假设“单控环境”（即只有AI智能体可以使用工具，用户仅被动提供信息）的评测不同，TAU2-Bench 将任务建模为一个 “双参与者环境” ：
- 智能体 (Agent)：负责理解问题、规划策略和调用工具。
- 用户 (User)：由模拟器代表，可以执行部分操作并影响环境状态。
这种设定更贴近真实世界的服务场景，例如技术支持，其中用户需要积极配合才能共同完成任务。该框架被形式化为一个 “去中心化部分可观马尔可夫决策过程”（Dec-POMDP） 。

**样例**

**测评方法**

```BASH
# 依赖外部模型：暂不支持；
```

### VITA-Bench

**概述**

VITA-Bench (VitaBench) 是一个旨在评估大语言模型（LLM）智能体（Agent）在真实世界场景中处理多样化交互任务能力的基准。它由美团长虹团队提出，并已被 ICLR 2026 接收。

它的核心目标是弥补现有基准测试与现实应用之间的差距，为智能体提供一个更贴近真实生活、更复杂的模拟环境。

VITA-Bench 专注于生活服务应用的核心理念。
- 三大真实领域：它从日常生活中汲取灵感，构建了外卖配送 (food delivery)、店内消费 (in-store consumption) 和在线旅游服务 (online travel services) 三大领域的模拟环境。
- 丰富的工具集：为这三大领域构建了多达 66种 工具，供智能体在完成任务时调用。
- 灵活的任务组合：通过一个消除领域特定策略的框架，可以灵活组合不同场景和工具，生成了 100个跨场景任务 和 300个单场景任务。
- 真实的任务来源：每个任务都源于多个真实用户请求，要求智能体处理复杂、模糊的指令。
- 全面的环境数据：基准测试包含了一个庞大的模拟世界。

**样例**

**评测方法**

```BASH
# 目前，通过 EvalScope 直接评测 VitaBench 尚无官方支持。
```

### MMMU

**概述**

MMMU（Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI） 是一个用于评估多模态模型在专家级任务上表现的综合基准测试。它的设计初衷是测试模型是否具备大学水平的学科知识和严谨的推理能力。

数据集规模与构成
- 题目总数：包含 11.5K 个精心收集的多模态问题。
- 数据来源：这些题目来自大学考试、测验和教科书。
- 学科覆盖：涵盖了 6 个核心学科领域，细分为 30 个科目和 183 个子领域：
  - 艺术与设计 (Art & Design)
  - 商科 (Business)
  - 科学 (Science)
  - 健康与医学 (Health & Medicine)
  - 人文与社会科学 (Humanities & Social Science)
  - 技术与工程 (Tech & Engineering)
- 图像类型：包含多达 30 种异构图像类型，如图表、示意图、地图、表格、乐谱和化学结构式等。

Download
```BASH
modelscope download --dataset AI-ModelScope/MMMU --local_dir ./MMMU
```

**样例**

**测评方法**

```BASH
evalscope eval \
  --model Qwen3-VL-8B-Instruct \
  --eval-type openai_api \
  --eval-batch-size 32 \
  --api-url http://localhost:8081/v1/chat/completions \
  --api-key EMPTY \
  --datasets mmmu \
  --generation-config '{"timeout": 3600, "retries": 10, "retry_interval": 5, "stream": true}' \
  --ignore-errors \
  --dataset-hub Local \
  --dataset-args '{"mmmu": {"local_path": "/home/workspace/data/evaludation/MMMU"}}' \
  --limit 10

```

### MMMU-Pro

**概述**

MMMU-Pro 是原始 MMMU 基准的增强版本，旨在更严格、更真实地评估多模态AI模型的能力。它通过一系列关键改进，提升了评估的挑战性，能有效避免模型依赖文本捷径或进行盲目猜测。

MMMU-Pro 通过一个严谨的“三步法”构建：
- 筛选问题：利用多个强大的纯文本大语言模型（LLMs）过滤掉那些无需图像即可回答的问题。
- 增加候选选项：由人类专家将每个问题的选项从4个增加到10个。
- 引入纯视觉输入：创建“视觉版”（vision）子集，将问题文本渲染到图像中，要求模型像人类一样同时进行“看”和“读”。

Download
```BASH
modelscope download --dataset AI-ModelScope/MMMU_Pro --local_dir ./MMMU_Pro
```

**样例**

**测评方法**

```BASH
evalscope eval \
  --model Qwen3-VL-8B-Instruct \
  --eval-type openai_api \
  --eval-batch-size 32 \
  --api-url http://localhost:8081/v1/chat/completions \
  --api-key EMPTY \
  --datasets mmmu_pro \
  --generation-config '{"timeout": 3600, "retries": 10, "retry_interval": 5}' \
  --ignore-errors \
  --dataset-hub Local \
  --dataset-args '{"mmmu_pro": {"local_path": "/home/workspace/data/evaludation/MMMU_Pro"}}' \
  --limit 10
```

### ZEROBench

**概述**

ZeroBench 的设计初衷，是打造一个对当前最前沿模型也极具挑战性的“硬核”评测基准。其子集 ZEROBench_sub 同样继承了这一特性：
- 极具挑战性：ZeroBench 包含 100 个高质量、人工精心设计的主问题，旨在超越当前模型的能力边界。在最初的评估中，20个前沿多模态模型在该基准上的得分为 0.0%。
- 细粒度分析：除主问题外，ZeroBench 还包含 334 个与之对应的子问题（subquestions）。这些子问题代表了解决主问题所需的推理步骤，难度相对较低，有助于对模型的推理过程进行更细致的分析。
- 模型表现：根据排行榜数据，目前表现领先的模型（如 Qwen3.5-122B-A10B）得分也仅为 0.362（满分100分制下可理解为36.2%），这表明该基准的区分度很高。

ZEROBench_sub 的数据集构成如下：
- 数据规模：包含 100 个主问题，平均提示词长度约为 645.72 个字符。
- 图像数据：共包含 108 张图像，每个问题配有 1 到 3 张图像，平均 1.08 张。图像格式为 JPEG 或 PNG，分辨率从 512x297 到 5559x4070 不等。
- 学科覆盖：涵盖多个领域、多种推理类型和图像类型。

Download
```
modelscope download --dataset evalscope/zerobench --local_dir ./zerobench
```

**样例**

**测评方法**

EvalScope 有一个 Bug，需要通过源码进行修复。
```BASH
  cp /cpfs-data/wanghaiyang/workspace/source/evaluation/.evalscope/lib/python3.12/site-packages/evalscope/benchmarks/zerobench/zerobench_adapter.py
  /cpfs-data/wanghaiyang/workspace/source/evaluation/.evalscope/lib/python3.12/site-packages/evalscope/benchmarks/zerobench/zerobench_adapter.py.bak

  2. 编辑文件，找到第 78 行附近：
  vi /cpfs-data/wanghaiyang/workspace/source/evaluation/.evalscope/lib/python3.12/site-packages/evalscope/benchmarks/zerobench/zerobench_adapter.py

  3. 将这行：
  processed_bytes, fmt = compress_image_to_limit(img['bytes'], 10_000_000)

  替换为：
  # Handle both dict and PIL Image objects
  if isinstance(img, dict):
      img_bytes = img['bytes']
  else:
      # PIL Image object - convert to bytes
      from io import BytesIO
      buf = BytesIO()
      # Convert to RGB if needed (JPEG doesn't support RGBA/P modes)
      if img.mode not in ('RGB', 'L'):
          img = img.convert('RGB')
      img.save(buf, format='JPEG')
      img_bytes = buf.getvalue()
  processed_bytes, fmt = compress_image_to_limit(img_bytes, 10_000_000)
```

```BASH
evalscope eval \
  --model Qwen3-VL-8B-Instruct \
  --eval-type openai_api \
  --eval-batch-size 32 \
  --api-url http://localhost:8081/v1/chat/completions \
  --api-key EMPTY \
  --datasets zerobench \
  --dataset-hub Local \
  --dataset-args '{"zerobench": {"local_path": "/home/workspace/data/evaludation/zerobench", "eval_split": "zerobench_subquestions", "max_image_bytes": null}}' \
  --generation-config '{"timeout": 3600, "retries": 10, "retry_interval": 5}' \
  --judge-model-args '{"model": "gpt-4", "api_url": "https://api.openai.com/v1/chat/completions", "api_key": "YOUR_OPENAI_KEY"}' \
  --limit 10
```

注意⚠️：ZEROBench 需要使用 LLM 来判断答案是否正确，--judge-model-args 用来配置评判模型。

### VlmsAreBlind

**概要**

VLMsAreBlind 是一个旨在检验视觉语言模型（VLM）基础视觉能力的基准测试。它通过一系列对人类来说极其简单、但对当前最先进模型却构成挑战的“视力测试”，揭示了这些模型在基础视觉感知上的“盲点”。
与许多侧重高难度推理的基准不同，VLMsAreBlind 的设计灵感来源于人类验光师的视力测试。其目标是评估 VLM “看见” 的能力，而不是它们“推理” 的能力。它旨在回答一个根本问题：这些模型是否真的能像人类一样，精准地感知视觉世界。

注意⚠️：EvalScope 并未将 VLMsAreBlind 列为原生支持的数据集。

### BabyVision

**概要**

BabyVision 是一个旨在评估多模态大语言模型（MLLMs）核心视觉能力的基准测试。它的核心发现是：当前最先进的MLLMs在人类（甚至是3岁儿童）能轻松完成的基础视觉任务上，表现依然不佳。
许多视觉问答任务都可以通过强大的语言推理能力“蒙混过关”，模型看似在“看”，实则在走“语言捷径”。BabyVision的设计目标就是剥离语言因素的干扰，纯粹地评估模型“看懂”世界的能力。

注意⚠️：BabyVision 不在 EvalScope 的原生支持列表中。

### RealWorldQA

**概要**

RealWorldQA 是由 xAI 公司贡献的一个多模态AI基准测试，旨在评估模型对现实世界空间和物理环境的理解能力。
该数据集由700多张来自真实场景的匿名化图像组成，包括车载摄像头拍摄的图像和其他现实世界图像，每张图像都配有一个问题及其可验证的答案。

Download
```
modelscope download --dataset lmms-lab/RealWorldQA --local_dir ./RealworldQA
```

**样例**

**测试方法**

注意⚠️：由于官方 EvalScope 在 RealWorldQA 的评测代码中存在 Bug ，导致模型没有图像数据输入，因此需要修复相关源码。

```PYTHON
# /home/workspace/source/evaluation/.venv/lib/python3.13/site-packages/evalscope/benchmarks/real_world_qa/real_world_qa_adapter.py
  ⎿  Added 7 lines, removed 1 line
      74          content_list: list[Content] = [ContentText(text=OPEN_PROMPT.format(question=record['question']))]
      75          image = record.get('image')
      76          if image:
      77 -            image_base64 = bytes_to_base64(image['bytes'], format='webp', add_header=True)
      77 +            # Convert WebP to JPEG because some llama.cpp builds silently drop WebP
      82 +            img.save(buf, 'JPEG', quality=92)
      83 +            image_base64 = 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode('utf-8')
      84              content_list.append(ContentImage(image=image_base64))
      85          return Sample(
      86              input=[ChatMessageUser(content=content_list)],
# 修改了 real_world_qa_adapter.py:73-83（WebP → JPEG），其余配置都可以保留。
```

```BASH
apt-get update && apt-get install -y ffmpeg

evalscope eval \
  --model Qwen3-VL-8B-Instruct \
  --eval-type openai_api \
  --eval-batch-size 32 \
  --api-url http://localhost:8081/v1/chat/completions \
  --api-key EMPTY \
  --ignore-errors \
  --datasets real_world_qa \
  --generation-config '{"timeout": 3600, "retries": 10, "retry_interval": 5, "stream": true}' \
  --dataset-hub Local \
  --dataset-args '{"real_world_qa": {
      "local_path": "/home/data/evaludation/RealworldQA",
      "system_prompt": "You are a helpful assistant. Think step by step. Your final response MUST end with exactly one line: ANSWER: <your_answer>"
    }}' \
  --limit 10
```

### MMStar

**概要**

MMStar 是一个精英级、视觉不可或缺（vision-indispensable）的多模态评测基准。它由上海人工智能实验室等机构的研究人员提出，旨在解决当时主流多模态基准测试中存在的数据泄露和“视觉偷懒”问题。

它的核心设计理念是：确保每个测试样本都必须依赖真实的视觉理解才能回答，从而更准确地评估模型真正的多模态能力。

在设计MMStar之前，许多多模态评测基准存在两个主要问题：
- 视觉信息不必要：许多问题仅凭问题和选项中的文本信息，或大模型本身的知识就能回答，无需参考图像。
- 无意的数据泄露：部分模型在训练时可能“记住”了测试集中的样本，即使不看图也能答对，导致评测结果虚高。

MMStar通过严格的数据筛选流程来解决这两个问题，确保每个样本都视觉依赖且数据泄露最小化。

Download
```
modelscope download --dataset evalscope/MMStar --local_dir ./MMStar
```

**样例**

**测试方法**

```BASH
evalscope eval \
  --model Qwen3-VL-8B-Instruct \
  --eval-type openai_api \
  --eval-batch-size 32 \
  --api-url http://localhost:8081/v1/chat/completions \
  --api-key EMPTY \
  --datasets mm_star \
  --generation-config '{"timeout": 3600, "retries": 10, "retry_interval": 5, "stream": true, "chat_template_kwargs": {"enable_thinking": false}}' \
  --ignore-errors \
  --dataset-hub Local \
  --dataset-args '{"mm_star": {"local_path": "/home/workspace/data/evaludation/MMStar"}}' \
  --limit 10
```

### SimpleVQA

**概要**

SimpleVQA 是一个用于评估多模态大语言模型（MLLMs）事实性（Factuality） 能力的综合性基准测试。它旨在检验模型在面对视觉内容时，能否提供基于事实的、准确的回答。

核心特点
- 专注“事实性”：这是首个专门针对MLLMs事实性回答能力的评估基准。它测试的是模型“观察世界”和“发现知识”的综合能力，而非简单的模式匹配。
- 双语与多样化：数据集是中英双语的，并覆盖了9大任务类型和9大主题领域（如当代社会、欧美历史文化等）。
- 高质量与挑战性：所有问题都经过严格的人工与模型筛选流程。一个关键步骤是，任何能够被GPT-4o等四个主流模型全部答对的问题都会被剔除，以确保基准的挑战性。
- 答案静态且永恒：所有问题的标准答案都是静态且不受时间影响的（例如“图中建筑的设计师是谁？”），避免了因时间推移导致答案变化的问题。

Download
```BASH
modelscope download --dataset m-a-p/SimpleVQA --local_dir ./SimpleVQA
```

注意：需要评判模型，因此暂时无法进行测试。

**样例**

**测试方法**

```BASH
evalscope eval \
  --model Qwen3-VL-8B-Instruct \
  --eval-type openai_api \
  --eval-batch-size 32 \
  --api-url http://localhost:8080/v1/chat/completions \
  --api-key EMPTY \
  --datasets simple_vqa \
  --dataset-hub Local \
  --dataset-args '{"simple_vqa": {"local_path": "/home/workspace/data/evaludation/SimpleVQA"}}' \
  --generation-config '{"timeout": 3600, "retries": 10, "retry_interval": 5}' \
  --ignore-errors \
  --limit 10
```

### CountBench

**概要**

CountBench 是一个专注于评估视觉语言模型（VLM）目标计数能力的图像-文本配对基准测试。它由Google Research等机构的研究人员在2023年提出。

注意⚠️：根据EvalScope的官方文档，CountBench数据集目前不在其原生支持的数据集列表中。

### RefCOCO

**概要**

RefCOCO 是一个经典的、被广泛用于指代表达理解（Referring Expression Comprehension, REC） 和视觉定位（Visual Grounding） 任务的数据集。

它的核心目标是让模型根据一段自然语言描述，在图像中准确地定位出对应的目标物体。

数据集概览
- 图像来源：所有图像均来自著名的 MS COCO 数据集。
- 数据规模：共包含 142,210 条指代表达，用于描述 19,994 张图片中的 50,000 个目标物体。
- 标注方式：通过亚马逊的众包平台 Amazon Mechanical Turk (AMT) 进行人工标注。
- 数据划分：数据集被划分为训练集（train）、验证集（val） 和测试集。其中测试集又分为 testA 和 testB 两个子集。
  - TestA：主要包含人物相关的指代样本。
  - TestB：包含其他物体（非人物）的指代样本。

**样例**

**测试方法**

```BASH
apt-get install -y default-jre-headless
uv pip install 'evalscope[refcoco]' --index-url https://aiext-pypi.mirrors.aliyuncs.com/pg1-pip/pypi_index/simple

evalscope eval \
  --model Qwen3-VL-8B-Instruct \
  --eval-type openai_api \
  --eval-batch-size 32 \
  --api-url http://localhost:8080/v1/chat/completions \
  --api-key EMPTY \
  --datasets refcoco \
  --dataset-hub Local \
  --dataset-args '{"refcoco": {"local_path": "/home/workspace/data/evaludation/refcoco", "subset_list": ["test", "validation", "testB"]}}' \
  --generation-config '{"timeout": 3600, "retries": 10, "retry_interval": 5}' \
  --ignore-errors \
  --limit 10

evalscope eval \
  --model Qwen3-VL-8B-Instruct \
  --eval-type openai_api \
  --eval-batch-size 32 \
  --api-url http://localhost:8080/v1/chat/completions \
  --api-key EMPTY \
  --datasets refcoco \
  --dataset-hub modelscope \
  --generation-config '{"timeout": 3600, "retries": 10, "retry_interval": 5}' \
  --ignore-errors
```

### RefSpatialBench

**概要**

RefSpatialBench 是一个极具挑战性的图像基准测试，专注于评估多模态大语言模型（MLLMs）在多步空间推理方面的能力。

它由“RoboRefer”项目提出，该项目是一个发表在 NeurIPS 2025 上的工作，旨在提升具身机器人在复杂3D环境中的空间指代能力。

与许多仅关注单步空间理解（如“识别红色方块”）的基准不同，RefSpatialBench 的核心是测试模型通过组合推理来解决复杂空间参考的能力。
- 多步推理：其任务需要模型进行最多5个步骤的推理才能得出正确答案。
- 复杂场景：数据基于真实世界杂乱场景构建，而非简单的合成图像。
- 高级能力：它要求模型不仅理解物体的位置和方向等基本属性，还要能动态地推断并执行指令所指示的交互位置。

**注意⚠️**：根据 EvalScope 的官方文档，RefSpatialBench 数据集目前不在其原生支持的数据集列表中。

### EmbSpatial-Bench 

**概述**

EmbSpatial-Bench 是一个由复旦大学团队提出的、专门用于评估大视觉语言模型（LVLMs）在具身（Embodied）场景下空间理解能力的基准测试。该工作已被 ACL 2024 会议接收。

核心特点：
- 聚焦具身空间理解：该基准旨在填补现有评估体系的空白，专注于测试LVLMs在具身环境（如机器人导航、操作等）中对物体间空间关系的理解能力。
- 覆盖6种空间关系：评测内容涵盖了从自我中心视角出发的6种核心空间关系。
- 极具挑战性：实验表明，即使是当时最先进的模型（如 GPT-4V）在该基准上的表现也不尽如人意，暴露了当前模型在具身空间理解上的短板。
- 附带指令微调数据集：研究者同时发布了 EmbSpatial-SFT，一个用于提升LVLMs具身空间理解能力的指令微调数据集。

**注意⚠️**：根据目前公开的信息，EvalScope 尚未原生支持 EmbSpatial-Bench 数据集。

### LingoQA

**概述**

LingoQA 是一个专为自动驾驶场景设计的视频问答（Video Question Answering）数据集和评测基准。它由Wayve等机构的研究人员提出，旨在评估视觉语言模型在真实驾驶环境中的理解与推理能力。

核心特点
LingoQA的设计紧密围绕自动驾驶的实际需求，具有以下特点：
- 聚焦驾驶场景：所有数据均来自真实的自动驾驶场景，专注于评估模型对驾驶环境、行为和规则的理解。
- 视频形式：与处理静态图像的数据集不同，LingoQA以短视频为基本单位，这能更好地捕捉驾驶过程中的时序动态信息。
- 问题多样化：问题不局限于简单的感知（“看到了什么？”），还涵盖了行动、推理和规划等更高阶的驾驶决策层面。
- 真实性与挑战性：数据集的问答对是自由形式的，更贴近人类司机的真实提问方式。

**注意⚠️**：根据 EvalScope 的官方文档，LingoQA 数据集目前不在其原生支持的数据集列表中。



## 5. 量化

模型量化参考：[MODEL_QUANTIZATION](https://github.com/wanghaiyang1930/wanghaiyang1930.github.io/blob/master/docs/vlm/MODEL_QUANTIZATION.md)