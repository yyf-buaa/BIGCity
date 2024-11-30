Trajectory Reconstruction Pretraining

```shell
model_name=GPT4TS
python -u run_traj.py \
  --task_name pretrain \
  --is_training 1 \
  --root_path ../dataset/ \
  --city cd \
  --freq s\
  --mask_rate 0.7 \
  --model $model_name \
  --data Traj \
  --use_multi_gpu \
  --devices '4,5' \
  --gpt_layers 12 \
  --batch_size  128\
  --d_model 768 \
  --des 'Exp' \
  --itr 1 \
  --mlp 1 \
  --learning_rate 2e-4 \
  --train_epochs 200 \
```

```shell
model_name=GPT4TS
python -u run_traj.py \
  --task_name pretrain \
  --is_training 1 \
  --root_path ../dataset/ \
  --city xa \
  --freq s\
  --mask_rate 0.7 \
  --model $model_name \
  --data Traj \
  --use_multi_gpu \
  --devices '4,5' \
  --gpt_layers 12 \
  --batch_size  128\
  --d_model 768 \
  --des 'Exp' \
  --itr 1 \
  --mlp 1 \
  --learning_rate 2e-4 \
  --train_epochs 200 \
```



Task Oriented Prompt Tuning

```shell
model_name=GPT4Finetune
python -u run_traj.py \
  --task_name task_tuning \
  --is_training 1 \
  --root_path ../dataset/ \
  --city cd \
  --freq s\
  --model $model_name \
  --data Task_Tuning \
  --use_multi_gpu \
  --devices '4,5' \
  --gpt_layers 12 \
  --batch_size  128\
  --d_model 768 \
  --des 'Exp' \
  --itr 1 \
  --mlp 1 \
  --learning_rate 1e-5 \
  --train_epochs 30 \
```



Generative Reinforcement Learning

```shell
model_name=GPT4Finetune
python -u run_traj.py \
  --task_name RL_tuning \
  --is_training 1 \
  --root_path ../dataset/ \
  --city cd \
  --freq s\
  --model $model_name \
  --data RL_Tuning \
  --use_multi_gpu \
  --devices '4,5' \
  --gpt_layers 12 \
  --batch_size  64\
  --d_model 768 \
  --critic_hidden_size 2 \
  --des 'Exp' \
  --itr 1 \
  --mlp 1 \
  --learning_rate 5e-7 \
  --train_epochs 30 \

```



参数说明（LLM 生成）：
--task_name：任务名称，可选值包括长期预测、短期预测、插补、分类、异常检测等。
--is_training：是否处于训练状态（1表示是，0表示否）。
--model_id：模型标识符，用于区分不同的模型实例。
--model：使用的模型名称，例如Autoformer, Transformer, TimesNet等。
--data：数据集类型。
--root_path：数据文件的根目录路径。
--city：数据集对应的城市。
--embedding_model：道路嵌入模型的名称。
--freq：时间特征编码的频率，如秒级、分钟级、小时级等。
--checkpoints：模型检查点保存的位置。
--mask_rate：掩码比率，即在输入序列中被随机掩盖的比例。
--num_kernels：Inception模块中的核数量。
--d_model：模型维度大小。
--n_heads：多头注意力机制中的头数。
--e_layers：编码器层数。
--d_layers：解码器层数。
--d_ff：前馈网络层的维度。
--moving_avg：移动平均窗口大小。
--factor：注意力因子。
--distil：是否在编码器中使用蒸馏，默认为True。
--dropout：Dropout率。
--embed：时间特征编码方式，选项包括timeF, fixed, learned。
--activation：激活函数类型。
--output_attention：是否输出编码器中的注意力权重。
--num_workers：数据加载器的工作线程数。
--itr：实验重复次数。
--train_epochs：训练轮次。
--batch_size：批量大小。
--patience：早停耐心度。
--learning_rate：学习率。
--des：实验描述。
--loss：损失函数类型。
--lradj：学习率调整策略。
--use_amp：是否使用自动混合精度训练。
--use_gpu：是否使用GPU。
--gpu：指定使用的GPU编号。
--use_multi_gpu：是否使用多个GPU。
--devices：当使用多GPU时，指定设备ID列表。
--p_hidden_dims：投影器隐藏层的维度列表。
--p_hidden_layers：投影器中的隐藏层数量。
--gpt_layers：GPT模型的层数。
--ln：层归一化标志。
--mlp：MLP标志。
--weight：权重系数。
--percent：百分比。
--loss_alpha, --loss_beta, --loss_gamma：多损失项的权重分配。
--checkpoint_name：特定检查点名称。
--gpt2_checkpoint_name：GPT-2模型的检查点名称。
--sample_rate：采样率。
