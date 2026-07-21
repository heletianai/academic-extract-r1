# 冒烟 50 条抽检包(2026-07-21)

> 抽检指引:优先看 ⚠️sota_disagree 和 benchmarks 非空的条目。
> 核对点:①benchmarks 数字是否摘要里真实存在(防幻觉) ②claims_sota/open_source 判定 ③task_type 归类。
> 结论回我:每条 OK 或 错在哪。核 15-20 条即够教师基线。

## 1. [2510.23295] Predicting symbolic ODEs from multiple trajectories
**摘要**: We introduce MIO, a transformer-based model for inferring symbolic ordinary differential equations (ODEs) from multiple observed trajectories of a dynamical system. By combining multiple instance learning with transformer-based symbolic regression, the model effectively leverages repeated observations of the same system to learn more generalizable representations of the underlying dynamics. We inv...

- task_type=`generation` modalities=`['text', 'code']` open_source=`False` claims_sota=`False`
- benchmarks: []
- keywords=['transformer', 'multiple instance learning', 'symbolic regression', 'ODE inference']

## 2. [2512.02704] ⚠️sota_disagree Conformal Correction for Efficiency May be at Odds with Entropy
**摘要**: Conformal prediction (CP) provides a comprehensive framework to produce statistically rigorous uncertainty sets for black-box machine learning models. To further improve the efficiency of CP, conformal correction is proposed to fine-tune or wrap the base model with an extra module using a conformal-aware inefficiency loss. In this work, we empirically and theoretically identify a trade-off between...

- task_type=`classification` modalities=`['text', 'image']` open_source=`False` claims_sota=`False`
- benchmarks: []
- keywords=['conformal prediction', 'conformal correction', 'entropy constraint', 'Pareto optimization']

## 3. [2512.21580] Gamayun's Path to Multilingual Mastery: Cost-Efficient Training of a 1.5B-Parame
**摘要**: We present Gamayun, a 1.5B-parameter multilingual language model trained entirely from scratch on 2.5T tokens. Designed for efficiency and deployment in resource-constrained environments, Gamayun addresses the lack of research on small non-English-centric LLMs by adopting a novel two-stage pre-training strategy: balanced multilingual training for cross-lingual alignment, followed by high-quality E...

- task_type=`other` modalities=`['text']` open_source=`False` claims_sota=`True`
- benchmarks: []
- keywords=['two-stage pre-training', 'balanced multilingual training', 'English enrichment']

## 4. [2508.16665] Trust but Verify! A Survey on Verification Design for Test-time Scaling
**摘要**: Test-time scaling (TTS) has emerged as a new frontier for scaling the performance of Large Language Models. In test-time scaling, by using more computational resources during inference, LLMs can improve their reasoning process and task performance. Several approaches have emerged for TTS such as distilling reasoning traces from another model or exploring the vast decoding search space by employing...

- task_type=`other` modalities=`['text']` open_source=`True` claims_sota=`False`
- benchmarks: []
- keywords=['test-time scaling', 'verifier design', 'reward models', 'survey']

## 5. [2511.10130] ⚠️sota_disagree RI-Loss: A Learnable Residual-Informed Loss for Time Series Forecasting
**摘要**: Time series forecasting relies on predicting future values from historical data, yet most state-of-the-art approaches-including transformer and multilayer perceptron-based models-optimize using Mean Squared Error (MSE), which has two fundamental weaknesses: its point-wise error computation fails to capture temporal relationships, and it does not account for inherent noise in the data. To overcome ...

- task_type=`generation` modalities=`['text']` open_source=`True` claims_sota=`False`
- benchmarks: []
- keywords=['Residual-Informed Loss', 'Hilbert-Schmidt Independence Criterion', 'noise-aware representation', 'non-asymptotic HSIC bound']

## 6. [2510.04230] Pushing on Multilingual Reasoning Models with Language-Mixed Chain-of-Thought
**摘要**: Recent frontier models employ long chain-of-thought reasoning to explore solution spaces in context and achieve stonger performance. While many works study distillation to build smaller yet capable models, most focus on English and little is known about language-specific reasoning. To bridge this gap, we first introduct **Language-Mixed CoT**, a reasoning schema that switches between English and a...

- task_type=`reasoning` modalities=`['text', 'code']` open_source=`True` claims_sota=`True`
- 三元组: `overall average` / `score` / `64.0`
- keywords=['Language-Mixed CoT', 'distillation', 'data curation', 'reasoning schema']

## 7. [2605.12980] CoRe-Gen: Robust Spectrum-to-Structure Generation under Imperfect Fingerprint Co
**摘要**: Molecular structure elucidation from tandem mass spectra (MS/MS) remains challenging, particularly for de novo generation beyond database coverage. A common approach decomposes the task into spectrum-to-fingerprint prediction followed by fingerprint-to-structure decoding, enabling the use of large-scale molecular corpora. However, at deployment, the decoder relies on predicted rather than oracle f...

- task_type=`generation` modalities=`['text']` open_source=`False` claims_sota=`True`
- 三元组: `NPLIB1` / `Top-1 exact-match accuracy` / `19.54`
- 三元组: `NPLIB1` / `Top-10 exact-match accuracy` / `29.92`
- keywords=['spectrum-to-structure generation', 'fingerprint corruption', 'autoregressive decoding', 'compositional SELFIES', 'synthetic-spectrum pretraining']

## 8. [2507.14353] Solo Connection: A Parameter Efficient Fine-Tuning Technique for Transformers
**摘要**: Parameter efficient fine tuning (PEFT) is a versatile and extensible approach for adapting a Large Language Model (LLM) for newer tasks. One of the most prominent PEFT approaches, Low Rank Adaptation (LoRA), primarily focuses on adjusting the attention weight matrices within individual decoder blocks of a Generative Pre trained Transformer (GPT2). In contrast, we introduce Solo Connection a novel ...

- task_type=`generation` modalities=`['text']` open_source=`False` claims_sota=`False`
- 三元组: `E2E` / `unspecified` / `0`
- keywords=['parameter efficient fine-tuning', 'decoder-block adaptation', 'homotopy interpolation', 'long skip connections']

## 9. [2601.13649] Fairness or Fluency? An Investigation into Language Bias of Pairwise LLM-as-a-Ju
**摘要**: Recent advances in Large Language Models (LLMs) have incentivized the development of LLM-as-a-judge, an application of LLMs where they are used as judges to decide the quality of a certain piece of text given a certain context. However, previous studies have demonstrated that LLM-as-a-judge can be biased towards different aspects of the judged texts, which often do not align with human preference....

- task_type=`classification` modalities=`['text']` open_source=`False` claims_sota=`False`
- benchmarks: []
- keywords=['language bias analysis', 'pairwise judging', 'perplexity correlation']

## 10. [2508.13229] RISE: Enhancing VLM Image Annotation with Self-Supervised Reasoning
**摘要**: Vision-Language Models (VLMs) struggle with complex image annotation tasks, such as emotion classification and context-driven object detection, which demand sophisticated reasoning. Standard Supervised Fine-Tuning (SFT) focuses solely on annotation outcomes, ignoring underlying rationales, while Visual Reinforcement Fine-Tuning (Visual-RFT) produces inconsistent Chains of Thought (CoTs) due to the...

- task_type=`multimodal` modalities=`['text', 'image']` open_source=`True` claims_sota=`False`
- benchmarks: []
- keywords=['self-supervised reasoning', 'reinforcement learning', 'chain-of-thought', 'closed-loop verification']

## 11. [2512.00598] Developing Fairness-Aware Task Decomposition to Improve Equity in Post-Spinal Fu
**摘要**: Fairness in clinical prediction models remains a persistent challenge, particularly in high-stakes applications such as spinal fusion surgery for scoliosis, where patient outcomes exhibit substantial heterogeneity. Many existing fairness approaches rely on coarse demographic adjustments or post-hoc corrections, which fail to capture the latent structure of clinical populations and may unintentiona...

- task_type=`classification` modalities=`['text']` open_source=`False` claims_sota=`False`
- 三元组: `postoperative complication prediction` / `AUC` / `0.86`
- 三元组: `postoperative complication prediction` / `accuracy` / `75`
- keywords=['multitask learning', 'subgroup inference', 'inverse-frequency weighting', 'demographic embedding']

## 12. [2605.01299] GA-VisAgent: A Multi-Agent application for code generation and visualization in 
**摘要**: Geometric Algebra (GA) presents challenges to learners due to its highly abstract mathematical structure and complex operational rules, as translating algebraic manipulations into concrete geometric interpretations is a non-intuitive process when developing related code. Currently, some existing GA software packages rely on manually written scripts for code generation and visualization, but their ...

- task_type=`generation` modalities=`['text', 'code']` open_source=`False` claims_sota=`False`
- 三元组: `40 typical Conformal GA tasks` / `code generation success rate` / `90`
- keywords=['multi-agent system', 'task planning', 'ReAct reasoning', 'geometric algebra']

## 13. [2510.26512] Inside CORE-KG: Evaluating Structured Prompting and Coreference Resolution for K
**摘要**: Human smuggling networks are increasingly adaptive and difficult to analyze. Legal case documents offer critical insights but are often unstructured, lexically dense, and filled with ambiguous or shifting references, which pose significant challenges for automated knowledge graph (KG) construction. While recent LLM-based approaches improve over static templates, they still generate noisy, fragment...

- task_type=`other` modalities=`['text']` open_source=`False` claims_sota=`False`
- benchmarks: []
- keywords=['coreference resolution', 'structured prompting', 'knowledge graph construction', 'ablation study']

## 14. [2508.14627] Clinical semantics for lung cancer prediction
**摘要**: Background: Existing clinical prediction models often represent patient data using features that ignore the semantic relationships between clinical concepts. This study integrates domain-specific semantic information by mapping the SNOMED medical term hierarchy into a low-dimensional hyperbolic space using Poincaré embeddings, with the aim of improving lung cancer onset prediction. Methods: Using ...

- task_type=`classification` modalities=`['text']` open_source=`False` claims_sota=`False`
- benchmarks: []
- keywords=['Poincaré embeddings', 'clinical knowledge graph', 'hyperbolic space', 'ResNet', 'Transformer']

## 15. [2605.12817] Training Large Language Models to Predict Clinical Events
**摘要**: Longitudinal clinical notes contain rich evidence of how patients evolve over time, but converting this signal into training supervision for clinical prediction remains challenging. We extend Foresight Learning to clinical prediction by converting time-ordered MIMIC-III notes into examples consisting of past patient context, a natural-language question about a possible future event, and a label re...

- task_type=`classification` modalities=`['text']` open_source=`False` claims_sota=`False`
- 三元组: `MIMIC-III` / `expected calibration error` / `0.0398`
- 三元组: `MIMIC-III` / `Brier score` / `0.145`
- keywords=['Foresight Learning', 'LoRA adapter', 'clinical prediction']

## 16. [2509.24317] Rethinking JEPA: Compute-Efficient Video SSL with Frozen Teachers
**摘要**: Video Joint Embedding Predictive Architectures (V-JEPA) learn generalizable off-the-shelf video representation by predicting masked regions in latent space with an exponential moving average (EMA)-updated teacher. While EMA prevents representation collapse, it complicates scalable model selection and couples teacher and student architectures. We revisit masked-latent prediction and show that a fro...

- task_type=`other` modalities=`['video']` open_source=`False` claims_sota=`False`
- benchmarks: []
- keywords=['frozen teacher', 'masked latent prediction', 'two-stage training', 'self-supervised learning']

## 17. [2603.04117] When to restart? Exploring escalating restarts on convergence
**摘要**: Learning rate scheduling plays a critical role in the optimization of deep neural networks, directly influencing convergence speed, stability, and generalization. While existing schedulers such as cosine annealing, cyclical learning rates, and warm restarts have shown promise, they often rely on fixed or periodic triggers that are agnostic to the training dynamics, such as stagnation or convergenc...

- task_type=`classification` modalities=`['text']` open_source=`False` claims_sota=`False`
- benchmarks: []
- keywords=['escalating restarts', 'learning rate scheduling', 'convergence-aware', 'stochastic gradient descent']

## 18. [2511.09354] ⚠️sota_disagree Spider4SSC & S2CLite: A text-to-multi-query-language dataset using lightweight o
**摘要**: We present Spider4SSC dataset and S2CLite parsing tool. S2CLite is a lightweight, ontology-agnostic parser that translates SPARQL queries into Cypher queries, enabling both in-situ and large-scale SPARQL to Cypher translation. Unlike existing solutions, S2CLite is purely rule-based (inspired by traditional programming language compilers) and operates without requiring an RDF graph or external tool...

- task_type=`other` modalities=`['text', 'code']` open_source=`True` claims_sota=`False`
- 三元组: `Spider4SPARQL` / `parsing accuracy` / `77.8`
- 三元组: `Spider4SPARQL` / `execution accuracy` / `96.6`
- keywords=['rule-based parser', 'SPARQL-to-Cypher translation', 'ontology-agnostic']

## 19. [2511.15915] AccelOpt: A Self-Improving LLM Agentic System for AI Accelerator Kernel Optimiza
**摘要**: We present AccelOpt, a self-improving large language model (LLM) agentic system that autonomously optimizes kernels for emerging AI acclerators, eliminating the need for expert-provided hardware-specific optimization knowledge. AccelOpt explores the kernel optimization space through iterative generation, informed by an optimization memory that curates experiences and insights from previously encou...

- task_type=`agent` modalities=`['text', 'code']` open_source=`True` claims_sota=`False`
- 三元组: `NKIBench` / `average percentage of peak throughput` / `61`
- 三元组: `NKIBench` / `average percentage of peak throughput` / `59`
- keywords=['self-improving agent', 'kernel optimization', 'optimization memory', 'iterative generation']

## 20. [2601.06142] Is Sanskrit the most token-efficient language? A quantitative study using GPT, G
**摘要**: Tokens are the basic units of Large Language Models (LLMs). LLMs rely on tokenizers to segment text into these tokens, and tokenization is the primary determinant of computational and inference cost. Sanskrit, one of the oldest languages, is hypothesized to express more meaning per token due to its morphology and grammar rules; however, no prior work has quantified this. We use a dataset of 701 pa...

- task_type=`other` modalities=`['text']` open_source=`True` claims_sota=`False`
- benchmarks: []
- keywords=['tokenization analysis', 'token efficiency', 'cross-lingual comparison']

## 21. [2507.22720] Investigating Hallucination in Conversations for Low Resource Languages
**摘要**: Large Language Models (LLMs) have demonstrated remarkable proficiency in generating text that closely resemble human writing. However, they often generate factually incorrect statements, a problem typically referred to as 'hallucination'. Addressing hallucination is crucial for enhancing the reliability and effectiveness of LLMs. While much research has focused on hallucinations in English, our st...

- task_type=`classification` modalities=`['text']` open_source=`False` claims_sota=`False`
- benchmarks: []
- keywords=['hallucination analysis', 'cross-lingual evaluation', 'conversational data']

## 22. [2510.17394] MILES: Modality-Informed Learning Rate Scheduler for Balancing Multimodal Learni
**摘要**: The aim of multimodal neural networks is to combine diverse data sources, referred to as modalities, to achieve enhanced performance compared to relying on a single modality. However, training of multimodal networks is typically hindered by modality overfitting, where the network relies excessively on one of the available modalities. This often yields sub-optimal performance, hindering the potenti...

- task_type=`multimodal` modalities=`['text', 'image', 'audio', 'video']` open_source=`False` claims_sota=`True`
- benchmarks: []
- keywords=['learning rate scheduler', 'modality balancing', 'conditional utilization', 'joint fusion']

## 23. [2603.29765] Training-Free Dynamic Upcycling of Expert Language Models
**摘要**: Large Language Models (LLMs) have achieved remarkable performance on a wide range of specialized tasks, exhibiting strong problem-solving capabilities. However, training these models is prohibitively expensive, and they often lack domain-specific expertise because they rely on general knowledge datasets. Expertise finetuning can address this issue; however, it often leads to overspecialization, an...

- task_type=`other` modalities=`['text']` open_source=`True` claims_sota=`False`
- benchmarks: []
- keywords=['dynamic upcycling', 'mixture of experts', 'ridge regression', 'expert reuse']

## 24. [2511.11828] Conformal Constrained Policy Optimization for Cost-Effective LLM Agents
**摘要**: While large language models (LLMs) have recently made tremendous progress towards solving challenging AI problems, they have done so at increasingly steep computational and API costs. We propose a novel strategy where we combine multiple LLM models with varying cost/accuracy tradeoffs in an agentic manner, where models and tools are run in sequence as determined by an orchestration model to minimi...

- task_type=`agent` modalities=`['text']` open_source=`False` claims_sota=`False`
- benchmarks: []
- keywords=['conformal prediction', 'constrained policy optimization', 'off-policy reinforcement learning', 'adaptive threshold']

## 25. [2604.27478] Toward Scalable SDN for LEO Mega-Constellations: A Graph Learning Approach
**摘要**: Terrestrial network limitations drive the integration of non-terrestrial networks (NTNs), notably mega-constellations comprising thousands of low Earth orbit (LEO) satellites. While these satellites act as interconnected network switches via inter-satellite links (ISLs), their massive scale creates severe bottlenecks for network management. To address this, we propose a scalable, hierarchical soft...

- task_type=`other` modalities=`['text']` open_source=`False` claims_sota=`False`
- benchmarks: []
- keywords=['graph neural networks', 'Koopman theory', 'autoencoder', 'hierarchical SDN']

## 26. [2603.05818] RouteGoT: Node-Adaptive Routing for Cost-Efficient Graph of Thoughts Reasoning
**摘要**: Large Language Models (LLMs) excel at multi-step reasoning, yet increasing the structural complexity of inference does not consistently improve system-level returns. Methods such as Tree of Thoughts (ToT), Graph of Thoughts (GoT), and Adaptive Graph of Thoughts (AGoT) can boost accuracy on some benchmarks, but often introduce substantial overhead in token consumption and latency, and their gains c...

- task_type=`reasoning` modalities=`['text']` open_source=`False` claims_sota=`False`
- benchmarks: []
- keywords=['node-adaptive routing', 'graph of thoughts', 'budget-controllable inference', 'cost-accuracy trade-off']

## 27. [2511.09586] Environment Scaling for Interactive Agentic Experience Collection: A Survey
**摘要**: LLM-based agents can autonomously accomplish complex tasks across various domains. However, to further cultivate capabilities such as adaptive behavior and long-term decision-making, training on static datasets built from human-level knowledge is insufficient. These datasets are costly to construct and lack both dynamism and realism. A growing consensus is that agents should instead interact direc...

- task_type=`agent` modalities=`['text']` open_source=`False` claims_sota=`False`
- benchmarks: []
- keywords=['environment scaling', 'GEF loop', 'task generation', 'reinforcement learning']

## 28. [2603.19040] When Differential Privacy Meets Wireless Federated Learning: An Improved Analysi
**摘要**: Differentially private wireless federated learning (DPWFL) is a promising framework for protecting sensitive user data. However, foundational questions on how to precisely characterize privacy loss remain open, and existing work is further limited by convergence analyses that rely on restrictive convexity assumptions or ignore the effect of gradient clipping. To overcome these issues, we present a...

- task_type=`other` modalities=`['text']` open_source=`False` claims_sota=`False`
- benchmarks: []
- keywords=['differential privacy', 'wireless federated learning', 'convergence analysis', 'privacy-utility trade-off']

## 29. [2509.16215] Discovering Software Parallelization Points Using Deep Neural Networks
**摘要**: This study proposes a deep learning-based approach for discovering loops in programming code according to their potential for parallelization. Two genetic algorithm-based code generators were developed to produce two distinct types of code: (i) independent loops, which are parallelizable, and (ii) ambiguous loops, whose dependencies are unclear, making them impossible to define if the loop is para...

- task_type=`classification` modalities=`['code']` open_source=`False` claims_sota=`False`
- benchmarks: []
- keywords=['deep neural network', 'convolutional neural network', 'genetic algorithm', 'code tokenization']

## 30. [2507.02778] Self-Correction Bench: Uncovering and Addressing the Self-Correction Blind Spot 
**摘要**: Although large language models (LLMs) have transformed AI, they still make mistakes and can explore unproductive reasoning paths. Self-correction capability is essential for deploying LLMs in safety-critical applications. We uncover a systematic failure: LLMs cannot correct errors in their own outputs while successfully correcting identical errors from external sources - a limitation we term the S...

- task_type=`reasoning` modalities=`['text']` open_source=`False` claims_sota=`False`
- 三元组: `Self-Correction Bench` / `blind spot rate` / `64.5`
- keywords=['self-correction', 'error injection', 'blind spot analysis', 'prompt triggering']

## 31. [2508.19598] Encouraging Good Processes Without the Need for Good Answers: Reinforcement Lear
**摘要**: The functionality of Large Language Model (LLM) agents is primarily determined by two capabilities: action planning and answer summarization. The former, action planning, is the core capability that dictates an agent's performance. However, prevailing training paradigms employ end-to-end, multi-objective optimization that jointly trains both capabilities. This paradigm faces two critical challenge...

- task_type=`agent` modalities=`['text']` open_source=`False` claims_sota=`False`
- benchmarks: []
- keywords=['reinforcement learning', 'tool-use rewards', 'planning module', 'decoupled training']

## 32. [2507.03001] Evaluating Hierarchical Clinical Document Classification Using Reasoning-Based L
**摘要**: This study evaluates how well large language models (LLMs) can classify ICD-10 codes from hospital discharge summaries, a critical but error-prone task in healthcare. Using 1,500 summaries from the MIMIC-IV dataset and focusing on the 10 most frequent ICD-10 codes, the study tested 11 LLMs, including models with and without structured reasoning capabilities. Medical terms were extracted using a cl...

- task_type=`classification` modalities=`['text']` open_source=`False` claims_sota=`False`
- benchmarks: []
- keywords=['clinical NLP', 'ICD-10 classification', 'reasoning-based LLMs', 'cTAKES']

## 33. [2510.09048] Spatio-Temporal Graph Convolutional Networks for EV Charging Demand Forecasting 
**摘要**: Transportation remains a major contributor to greenhouse gas emissions, highlighting the urgency of transitioning toward sustainable alternatives such as electric vehicles (EVs). Yet, uneven spatial distribution and irregular utilization of charging infrastructure create challenges for both power grid stability and investment planning. This study introduces TW-GCN, a spatio-temporal forecasting fr...

- task_type=`other` modalities=`['text', 'image']` open_source=`False` claims_sota=`False`
- benchmarks: []
- keywords=['spatio-temporal graph convolution', 'temporal architecture', 'multi-modal data fusion']

## 34. [2602.17614] Guarding the Middle: Protecting Intermediate Representations in Federated Split 
**摘要**: Big data scenarios, where massive, heterogeneous datasets are distributed across clients, demand scalable, privacy-preserving learning methods. Federated learning (FL) enables decentralized training of machine learning (ML) models across clients without data centralization. Decentralized training, however, introduces a computational burden on client devices. U-shaped federated split learning (UFSL...

- task_type=`classification` modalities=`['image']` open_source=`False` claims_sota=`False`
- benchmarks: []
- keywords=['federated split learning', 'differential privacy', 'microaggregation', 'k-anonymity']

## 35. [2604.06732] Extraction of linearized models from pre-trained networks via knowledge distilla
**摘要**: Recent developments in hardware, such as photonic integrated circuits and optical devices, are driving demand for research on constructing machine learning architectures tailored for linear operations. Hence, it is valuable to explore methods for constructing learning machines with only linear operations after simple nonlinear preprocessing. In this study, we propose a framework to extract a linea...

- task_type=`classification` modalities=`['text', 'image']` open_source=`False` claims_sota=`False`
- benchmarks: []
- keywords=['Koopman operator theory', 'knowledge distillation', 'linearized model', 'model extraction']

## 36. [2605.02110] Adversarial Update-Based Federated Unlearning for Poisoned Model Recovery
**摘要**: Federated learning (FL) is vulnerable to poisoning attacks, where malicious clients upload manipulated updates to degrade the performance of the global model. Although detection methods can identify and remove malicious clients, the model remains affected. Retraining from scratch is effective but costly, and existing unlearning methods remain unsatisfactory in both effectiveness and efficiency. We...

- task_type=`classification` modalities=`['text']` open_source=`False` claims_sota=`False`
- benchmarks: []
- keywords=['federated unlearning', 'adversarial optimization', 'poisoned model recovery', 'proxy dataset']

## 37. [2601.04201] ⚠️sota_disagree Collective Narrative Grounding: Community-Coordinated Data Contributions to Impr
**摘要**: Large language model (LLM) question-answering systems often fail on community-specific queries, creating "knowledge blind spots" that marginalize local voices and reinforce epistemic injustice. We present Collective Narrative Grounding, a participatory protocol that transforms community stories into structured narrative units and integrates them into AI systems under community governance. Learning...

- task_type=`retrieval` modalities=`['text']` open_source=`False` claims_sota=`False`
- benchmarks: []
- keywords=['participatory protocol', 'narrative grounding', 'community governance', 'knowledge blind spots']

## 38. [2605.12798] Emergent and Subliminal Misalignment Through the Lens of Data-Mediated Transfer
**摘要**: Fine-tuning LLMs on narrow harmful datasets can induce Emergent Misalignment (EM), where models exhibit misaligned behavior far beyond the fine-tuning distribution. We argue that emergent misalignment can be better understood as a data-mediated transfer phenomenon: harmful fine-tuning examples do not induce uniform behavioral spillover, but interact with the structural properties of the dataset an...

- task_type=`other` modalities=`['text']` open_source=`False` claims_sota=`False`
- benchmarks: []
- keywords=['emergent misalignment', 'data-mediated transfer', 'subliminal learning', 'off-policy distillation', 'on-policy distillation']

## 39. [2602.20904] Transcoder Adapters for Reasoning-Model Diffing
**摘要**: While reasoning models are increasingly ubiquitous, the effects of reasoning training on a model's internal mechanisms remain poorly understood. In this work, we introduce transcoder adapters, a technique for learning an interpretable approximation of the difference in MLP computation before and after fine-tuning. We apply transcoder adapters to characterize the differences between Qwen2.5-Math-7B...

- task_type=`reasoning` modalities=`['text']` open_source=`False` claims_sota=`False`
- benchmarks: []
- keywords=['transcoder adapters', 'interpretability', 'MLP diffing', 'attribution graphs']

## 40. [2602.07135] Landscaper: Understanding Loss Landscapes Through Multi-Dimensional Topological 
**摘要**: Loss landscapes are a powerful tool for understanding neural network optimization and generalization, yet traditional low-dimensional analyses often miss complex topological features. We present Landscaper, an open-source Python package for arbitrary-dimensional loss landscape analysis. Landscaper combines Hessian-based subspace construction with topological data analysis to reveal geometric struc...

- task_type=`other` modalities=`['text']` open_source=`True` claims_sota=`False`
- benchmarks: []
- keywords=['loss landscape analysis', 'topological data analysis', 'Hessian-based subspace', 'Saddle-Minimum Average Distance']

## 41. [2512.11860] An Operator-Consistent Graph Neural Network for Learning Diffusion Dynamics on I
**摘要**: Classical numerical methods solve partial differential equations (PDEs) efficiently on regular meshes, but many of them become unstable on irregular domains. In practice, multiphysics interactions such as diffusion, damage, and healing often take place on irregular meshes. We develop an operator-consistent graph neural network (OCGNN-PINN) that approximates PDE evolution under physics-informed con...

- task_type=`other` modalities=`['text']` open_source=`False` claims_sota=`False`
- benchmarks: []
- keywords=['operator-consistent GNN', 'physics-informed constraints', 'node-edge message passing', 'consistency loss']

## 42. [2509.19557] Confidence Calibration in Large Language Model-Based Entity Matching
**摘要**: This research aims to explore the intersection of Large Language Models and confidence calibration in Entity Matching. To this end, we perform an empirical study to compare baseline RoBERTa confidences for an Entity Matching task against confidences that are calibrated using Temperature Scaling, Monte Carlo Dropout and Ensembles. We use the Abt-Buy, DBLP-ACM, iTunes-Amazon and Company datasets. Th...

- task_type=`classification` modalities=`['text']` open_source=`False` claims_sota=`False`
- benchmarks: []
- keywords=['confidence calibration', 'temperature scaling', 'monte carlo dropout', 'ensembles']

## 43. [2509.04482] Energy Landscapes Enable Reliable Abstention in Retrieval-Augmented Large Langua
**摘要**: Reliable abstention is critical for retrieval-augmented generation (RAG) systems, particularly in safety-critical domains such as women's health, where incorrect answers can lead to harm. We present an energy-based model (EBM) that learns a smooth energy landscape over a dense semantic corpus of 2.6M guideline-derived questions, enabling the system to decide when to generate or abstain. We benchma...

- task_type=`retrieval` modalities=`['text']` open_source=`False` claims_sota=`False`
- 三元组: `hard abstention split` / `AUROC` / `0.961`
- 三元组: `hard abstention split` / `FPR@95` / `0.235`
- keywords=['energy-based model', 'abstention scoring', 'retrieval-augmented generation', 'energy landscape']

## 44. [2602.22297] Learning Rewards, Not Labels: Adversarial Inverse Reinforcement Learning for Mac
**摘要**: Reinforcement learning (RL) offers significant promise for machinery fault detection (MFD). However, most existing RL-based MFD approaches do not fully exploit RL's sequential decision-making strengths, often treating MFD as a simple guessing game (Contextual Bandits). To bridge this gap, we formulate MFD as an offline inverse reinforcement learning problem, where the agent learns the reward dynam...

- task_type=`classification` modalities=`['text']` open_source=`False` claims_sota=`False`
- benchmarks: []
- keywords=['adversarial inverse reinforcement learning', 'offline inverse RL', 'reward learning', 'anomaly scoring']

## 45. [2507.06137] NeoBabel: A Multilingual Open Tower for Visual Generation
**摘要**: Text-to-image generation advancements have been predominantly English-centric, creating barriers for non-English speakers and perpetuating digital inequities. While existing systems rely on translation pipelines, these introduce semantic drift, computational overhead, and cultural misalignment. We introduce NeoBabel, a novel multilingual image generation framework that sets a new Pareto frontier i...

- task_type=`generation` modalities=`['text', 'image']` open_source=`True` claims_sota=`True`
- 三元组: `m-GenEval` / `score` / `0.75`
- 三元组: `m-DPG` / `score` / `0.68`
- keywords=['multilingual pretraining', 'instruction tuning', 'crosslingual generalization', 'alignment training']

## 46. [2510.00236] Per-example gradients: a new frontier for understanding and improving optimizers
**摘要**: Training algorithms in deep learning usually treat a mini-batch of samples as a single object; they average gradients over the mini-batch, and then process the average in various ways. Computing other statistics beyond the average may have been seen as prohibitively resource intensive in automatic differentiation (AD) frameworks. We show that this is not the case. Generally, gradient statistics ca...

- task_type=`other` modalities=`['text']` open_source=`False` claims_sota=`False`
- benchmarks: []
- keywords=['per-example gradients', 'automatic differentiation', 'signSGD', 'Adam preconditioner']

## 47. [2512.18607] The Interaction Bottleneck of Deep Neural Networks: Discovery, Proof, and Modula
**摘要**: Understanding what kinds of cooperative structures deep neural networks (DNNs) can represent remains a fundamental yet insufficiently understood problem. In this work, we treat interactions as the fundamental units of such structure and investigate a largely unexplored question: how DNNs encode interactions under different levels of contextual complexity, and how these microscopic interaction patt...

- task_type=`other` modalities=`['text']` open_source=`False` claims_sota=`False`
- benchmarks: []
- keywords=['interaction bottleneck', 'multi-order interactions', 'contextual variability', 'gradient variance', 'representational bias']

## 48. [2604.17323] A Universal Avoidance Method for Diverse Multi-branch Generation
**摘要**: Modern generative models still lack human-level creativity, particularly in multi-branch diversity. Prior approaches to address this problem often incur heavy computation or strong dependency on model architecture. Therefore, we introduce UAG(Universal Avoidance Generation), a model-agnostic and computationally efficient generation strategy that penalizes similarity among previously generated outp...

- task_type=`generation` modalities=`['text']` open_source=`True` claims_sota=`True`
- benchmarks: []
- keywords=['Universal Avoidance Generation', 'model-agnostic', 'similarity penalization', 'multi-branch diversity']

## 49. [2604.17695] MoE-nD: Per-Layer Mixture-of-Experts Routing for Multi-Axis KV Cache Compression
**摘要**: KV cache memory is the dominant bottleneck for long-context LLM inference. Existing compression methods each act on a single axis of the four-dimensional KV tensor -- token eviction (sequence), quantization (precision), low-rank projection (head dimension), or cross-layer sharing -- but apply the same recipe to every layer. We show that this homogeneity leaves accuracy on the table: different laye...

- task_type=`other` modalities=`['text']` open_source=`False` claims_sota=`False`
- benchmarks: []
- keywords=['mixture-of-experts', 'KV cache compression', 'per-layer routing', 'heterogeneous eviction', 'greedy solver']

## 50. [2603.06595] Rethinking Personalization in Large Language Models at the Token Level
**摘要**: With large language models (LLMs) now performing strongly across diverse tasks, there is growing demand for them to personalize outputs for individual users. Personalization is typically framed as an additional layer on top of a base NLP task, requiring model responses to meet user-specific needs while still accomplishing the underlying task. From a token-level perspective, different tokens in a r...

- task_type=`generation` modalities=`['text']` open_source=`False` claims_sota=`False`
- 三元组: `LongLaMP` / `average gain` / `10`
- 三元组: `LongLaMP` / `max gain` / `68.04`
- keywords=['self-contrast', 'causal intervention', 'token-level personalization', 'PerCE loss']
