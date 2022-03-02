# PyTorch-PKD-for-BERT-Compression

Pytorch implementation of the distillation method described in the following paper: [**Patient Knowledge Distillation for BERT Model Compression**](https://arxiv.org/abs/1908.09355). This repository heavily refers to [Pytorch-Transformers](https://github.com/huggingface/pytorch-transformers) by huggingface.

## Steps to run the code
### 1. download glue_data
```
$ python download_glue_data.py
```

### 2. Fine-tune teacher BERT model
By running following code, save fine-tuned model.
```
python run_glue.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $GLUE_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir /tmp/$TASK_NAME/
```

### 3. distill student model with teacher BERT
$TEACHER_MODEL is your fine-tuned model folder.
```
python run_glue_distillation.py \
    --model_type bert \
    --teacher_model $TEACHER_MODEL \
    --student_model bert-base-uncased \
    --task_name $TASK_NAME \
    --num_hidden_layers 6 \
    --alpha 0.5 \
    --beta 100.0 \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $GLUE_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --learning_rate 2e-5 \
    --num_train_epochs 4.0 \
    --output_dir /tmp/$TASK_NAME/
```

## Experimental Results on dev set
model | num_layers | SST-2 | MRPC-f1/acc | QQP-f1/acc | MNLI-m/mm | QNLI | RTE 
-- | -- | -- | -- | -- | -- | -- | -- 
base | 12 | 0.9232 | 0.89/0.8358 | 0.8818/0.9121 | 0.8432/0.8479 | 0.916 | 0.6751 
finetuned | 6 | 0.9002 | 0.8741/0.8186 | 0.8672/0.901 |	0.8051/0.8033 |	0.8662 |	0.6101 
distill | 6 | 0.9071 |	0.8885/0.8382 | 0.8704/0.9016 |	0.8153/0.821 |	0.8642 |	0.6318 
