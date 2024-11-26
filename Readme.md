# BIGCity: A Versatile Model for Unified Multi-level Spatiotemporal Data Analysis



## Methods：

### 1. Unified spatiotemporal data representation(Section 2.1, Section 3.1)

We proposed STUnit, and ST tokenizer.

In our datasets, hetero data all originate from the city's road network, which is structured as a graph. Each node in road network contains both static road segment information and dynamic traffic states.

Therefore, we developed the **ST tokenizer** to represent the road network.

#### 1) The representation of road network

As Shown in Figure 1, ST tokenizer incorporates a static encoder and a dynamic encoder to separately model these static and dynamic features. Additionally, we designed a fusion encoder to integrate these two representations, generating dynamic road network representation.

<div align="center"><img src=image/Article/Figure1.png width=80% /></div>

#### 2) The representation of ST data

As shown in Figure 2, We find that either trajectory and traffic state is essentially a sequence, sampling from the dynamic road network. 

From that view, the most difference between trajectory and traffic state lies in sampling manner

Therefore, we designed STUnit to unified both trajectory and traffic state data into sequence format.

<div align="center"><img src=image/Article/Figure2.png width=80% /></div>


By integrating STUnit and the ST tokenizer, both trajectory and traffic states are commonly represented as feature sequence. 

Therefore, all tasks can be trained in sequence modeling manner.

Considering the strong sequence modeling capabilities of GPT-2, we selected it as the backbone of our model.


<div align="center"><img src=image/Article/Figure3.png width=80% /></div>

### 2. A task representation approach based on textual instructions (Subsection "Input" in Section 3.2, Section 4.2)

It is difficult for a model to identify the specific task type just according to the spatiotemporal data, because various task may share the same ST input.

To address this, we introduce an instruction mechanism that serves as a task identifier. 

Based on this approach, data from multiple spatiotemporal tasks can be integrated into a single dataset for joint training.

Specifically :

we categorize ST tasks into four major types (see in paper's Table 1), with the outputs summarized into two forms: classification of static road segment IDs and regression of dynamic features. Accordingly, we define task placeholders as [CLS] for classification and [REG] for regression.

First, we designed task placeholders to act as output markers, indicating the type and quantity of outputs for each task. Additionally, we provide individual textual instruction templates for each task to specify the task type. Details of these templates can be found in the Appendix, and section 4.2

### 3. three-step hierarchical learning strategy (Section 4)

Heterogeneous tasks differ in complexity and training paradigm. For instance, generation tasks produce sequences as outputs, using sequence labels as supervision, while classification and regression tasks rely on single-value labels for supervision.

To address these challenges, we designed a three-stage training process:

Pre-training: In this stage, only spatiotemporal data (ST data) and task placeholders are involved.

Prompt tuning: With the aid of textual instructions, the model is jointly fine-tuned on omultiple tasks, 
After this stage, model is able to handle classification, regression tasks.

Reinforcement Learning (RL): In the final stage, RL is introduced to specifically enhance the model's performance on sequence-labeling tasks, such as generation.



## Training

#### Requirements

The following command creates a conda environment named bigst

```shell
conda env create -f environment.yml
```



#### stage1:  Pretrain

```shell
model_name=GPT4TS
python run_pretrain.py \
  --task_name pretrain \
  --is_training 1 \
  --root_path ../dataset/ \
  --city xa \
  --freq s\
  --mask_rate 0.7 \
  --model_id 1 \
  --model GPT4TS \
  --data Traj \
  --devices '1' \
  --gpt_layers 12 \
  --batch_size  128\
  --d_model 768 \
  --des 'Exp' \
  --itr 1 \
  --mlp 1 \
  --learning_rate 2e-4 \
  --train_epochs 200
```



#### stage2:  Task Oriented Prompt Tuning

```shell
model_name=GPT4Finetune
python -u run_traj.py \
  --task_name task_tuning \
  --is_training 1 \
  --root_path ../dataset/ \
  --city xa \
  --freq s\
  --model $model_name \
  --data Task_Tuning \
  --use_multi_gpu \
  --devices '1' \
  --gpt_layers 12 \
  --batch_size  128\
  --d_model 768 \
  --des 'Exp' \
  --itr 1 \
  --mlp 1 \
  --learning_rate 1e-5 \
  --train_epochs 30 \
```



#### Description of command line parameters

- `--task_name`: Task name, options include: long-term forecast, short-term forecast, imputation, classification, anomaly detection, etc.
- `--is_training`: Whether in training state (1 means yes, 0 means no).
- `--model_id`: Model identifier, used to distinguish different model instances.
- `--model`: The model name, e.g., Autoformer, Transformer, TimesNet, etc.
- `--data`: Dataset type.
- `--root_path`: Root directory of the data files.
- `--city`: The city corresponding to the dataset.
- `--embedding_model`: The name of the road embedding model.
- `--freq`: Frequency for time feature encoding, e.g., second, minute, hour, etc.
- `--checkpoints`: Path to save model checkpoints.
- `--mask_rate`: Mask ratio, i.e., the proportion of the input sequence randomly masked.
- `--num_kernels`: Number of kernels in the Inception module.
- `--d_model`: Model dimension size.
- `--n_heads`: Number of attention heads in multi-head attention.
- `--e_layers`: Number of encoder layers.
- `--d_layers`: Number of decoder layers.
- `--d_ff`: Dimension of the feed-forward network layer.
- `--moving_avg`: Size of the moving average window.
- `--factor`: Attention factor.
- `--distil`: Whether to use distillation in the encoder, default is `True`.
- `--dropout`: Dropout rate.
- `--embed`: Time feature encoding method, options include: `timeF`, `fixed`, `learned`.
- `--activation`: Activation function type.
- `--output_attention`: Whether to output attention weights from the encoder.
- `--num_workers`: Number of data loader workers.
- `--itr`: Number of experimental repetitions.
- `--train_epochs`: Number of training epochs.
- `--batch_size`: Batch size.
- `--patience`: Early stopping patience.
- `--learning_rate`: Learning rate.
- `--des`: Experiment description.
- `--loss`: Loss function type.
- `--lradj`: Learning rate adjustment strategy.
- `--use_amp`: Whether to use automatic mixed precision training.
- `--use_gpu`: Whether to use GPU.
- `--gpu`: GPU ID.
- `--use_multi_gpu`: Whether to use multiple GPUs.
- `--devices`: List of device IDs when using multiple GPUs.
- `--p_hidden_dims`: Hidden layer dimensions of the projector (list).
- `--p_hidden_layers`: Number of hidden layers in the projector.
- `--gpt_layers`: Number of layers in the GPT model.
- `--ln`: Whether to use layer normalization.
- `--mlp`: Whether to use MLP (Multi-Layer Perceptron).
- `--weight`: Weight coefficient.
- `--percent`: Percentage.
- `--loss_alpha, --loss_beta, --loss_gamma`: Weight distribution for multi-loss terms.
- `--checkpoint_name`: Name of a specific checkpoint.
- `--gpt2_checkpoint_name`: GPT-2 model checkpoint name.
- `--sample_rate`: Sampling rate.
