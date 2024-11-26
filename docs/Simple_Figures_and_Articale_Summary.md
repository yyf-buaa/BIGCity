# Motivation：

The novelty of this paper is using a unified model to handle tasks involving both "trajectory" and "traffic state." Applications like navigation apps require both data types—users need traffic predictions and optimal paths (trajectories). Fundamentally, traffic states are derived from individual trajectories, so integrating them enhances model performance. This was often overlooked in prior research, and we fill that gap.

# Challenges:

The main challenges lies in how to unify heterogeneous data and heterogeneous task.

1. Heterogeneous data are originally in different data format.
2. Heterogeneous task are in various complexity, leading to hetero training paradigms. Besides, it is difficult for a model to identify the specific task type just according to the spatiotemporal data.

# Contribution:

1. Unified spatiotemporal data representation(Section 2.1, Section 3.1)
2. A task representation approach based on textual instructions (Subsection "Input" in Section 3.2, Section 4.2)
3. three-step hierarchical learning strategy (Section 4)

# Methods：

## 1. Unified spatiotemporal data representation(Section 2.1, Section 3.1)

We proposed STUnit, and ST tokenizer.

In our datasets, hetero data all originate from the city's road network, which is structured as a graph. Each node in road network contains both static road segment information and dynamic traffic states.

Therefore, we developed the **ST tokenizer** to represent the road network.

### 1) The representation of road network
As Shown in Figure 1, ST tokenizer incorporates a static encoder and a dynamic encoder to separately model these static and dynamic features. Additionally, we designed a fusion encoder to integrate these two representations, generating dynamic road network representation.

<div align="center"><img src=image/Article/Figure1.png width=80% /></div>

### 2) The representation of ST data

As shown in Figure 2, We find that either trajectory and traffic state is essentially a sequence, sampling from the dynamic road network. 

From that view, the most difference between trajectory and traffic state lies in sampling manner

Therefore, we designed STUnit to unified both trajectory and traffic state data into sequence format.

<div align="center"><img src=image/Article/Figure2.png width=80% /></div>


By integrating STUnit and the ST tokenizer, both trajectory and traffic states are commonly represented as feature sequence. 

Therefore, all tasks can be trained in sequence modeling manner.

Considering the strong sequence modeling capabilities of GPT-2, we selected it as the backbone of our model.


<div align="center"><img src=image/Article/Figure3.png width=80% /></div>

## 2. A task representation approach based on textual instructions (Subsection "Input" in Section 3.2, Section 4.2)

It is difficult for a model to identify the specific task type just according to the spatiotemporal data, because various task may share the same ST input.

To address this, we introduce an instruction mechanism that serves as a task identifier. 

Based on this approach, data from multiple spatiotemporal tasks can be integrated into a single dataset for joint training.

Specifically :

we categorize ST tasks into four major types (see in paper's Table 1), with the outputs summarized into two forms: classification of static road segment IDs and regression of dynamic features. Accordingly, we define task placeholders as [CLS] for classification and [REG] for regression.

First, we designed task placeholders to act as output markers, indicating the type and quantity of outputs for each task. Additionally, we provide individual textual instruction templates for each task to specify the task type. Details of these templates can be found in the Appendix, and section 4.2

## three-step hierarchical learning strategy (Section 4)

Heterogeneous tasks differ in complexity and training paradigm. For instance, generation tasks produce sequences as outputs, using sequence labels as supervision, while classification and regression tasks rely on single-value labels for supervision.

To address these challenges, we designed a three-stage training process:

Pre-training: In this stage, only spatiotemporal data (ST data) and task placeholders are involved.

Prompt tuning: With the aid of textual instructions, the model is jointly fine-tuned on omultiple tasks, 
After this stage, model is able to handle classification, regression tasks.

Reinforcement Learning (RL): In the final stage, RL is introduced to specifically enhance the model's performance on sequence-labeling tasks, such as generation.

<div align="center"><img src=image/Article/Figure4.png width=100% /></div>


## Performance Comparison (Figure 1)

Our model is able to integrate various types of data and simultaneously handle multiple types of tasks (shown in paper's Tab1).
For each task, current methods prefer to design task-specific models.
As a result, we must compare our model with three distinct categories of baselines.

Specifically, we compare our model with three types of task-specific baselines. The corresponding datasets, models, and tasks for each type of baseline are detailed in the figure：

<div align="center"><img src=image/Article/traditional_model.png width=100% /></div>

In current methods, these three baselines differ in both model architecture and training paradigms. Our model serves as the initial probe to simultaneously handle such different tasks. In BIGST, the relationships among data, model, and tasks are as follows:

<div align="center"><img src=image/Article/BIGST_model.png width=100% /></div>

In paper's figure 1, each edge node is the task name, with its corresponding metric indicated in parentheses. Our model is represented by the only circular curve, while the performance curves of other models are confined to some specific local regions. The original version of paper's figure 1 is shown as:

<div align="center"><img src=image/Article/Paper_Figure1.png width=100% /></div>

To provide a clearer comparison of BIGST's performance against various baselines, we simplify paper's figure 1 according to the types of baseline. The results are presented as follows:


### The comparison of trajectory representation baselines:
<div align="center"><img src=image/Article/Figure1_1.png width=100% /></div>

### The comparison of trajectory generation baselines:
<div align="center"><img src=image/Article/Figure1_2.png width=100% /></div>

### The comparison of traffic state prediction baselines:
<div align="center"><img src=image/Article/Figure1_3.png width=100% /></div>
