# COYO
1. 请将`part_{00000..00003}`修改为实际coyo part数量
2. 请将`--train-num-samples 5012269`修改为coyo总数据数量
# imagenet
请将`--imagenet-val /dataset/imagenet/val`修改为image net的实际位置
# command
这里batch size是单卡batchsize
```bash
cd ./src

torchrun --nnodes=1 --nproc_per_node=8 -m open_clip_train.simple_main -- \
--data-dir /dataset/coyo-700m_full_webdata/part_00000 --wds-meta-path data/coyo_part_00000.json \
--lr 0.0005 --lr-scheduler const --warmup 10000 \
--total-steps 400000 \
--beta1 0.9 --beta2 0.98 \
--precision amp_bf16 \
--name test4 \
--report-to wandb --wandb-entity latentvit --wandb-project-name "CLIP Training Validation" \
--name open_clip_test \
--logs exp_dongyun \
--save-checkpoint-steps 1000 \
--evaluation-steps 5000 --resume latest

python -m open_clip_train.simple_main \
--data-dir /dataset/coyo-700m_full_webdata/part_00000 --wds-meta-path data/coyo_part_00000.json \
--lr 0.0005 --lr-scheduler const --warmup 10000 \
--total-steps 400000 \
--beta1 0.9 --beta2 0.98 \
--precision amp_bf16 \
--name test4 \
--report-to wandb --wandb-entity latentvit --wandb-project-name "CLIP Training Validation" \
--name open_clip_test \
--logs exp_dongyun \
--save-checkpoint-steps 1000 \
--evaluation-steps 5000 --resume latest

--dataset-type webdataset \
--batch-size 64 --imagenet-val /dataset/imagenet/val --model ViT-B-32 \
--data-dir /dataset/coyo-700m_full_webdata/part_00000 --wds-meta-path data/coyo_part_00000.json \
--lr 0.0005 --lr-scheduler const --warmup 10000 \
--total-steps 400000 \
--beta1 0.9 --beta2 0.98 \
--precision amp_bf16 \
--name test4 \
--logs exp_dongyun \
--save-checkpoint-steps 10 \
--evaluation-steps 50 --resume latest
```