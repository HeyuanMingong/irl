
python main.py --max_epochs 1000 --strategy softmax --env Maze2D-v1 \
    --output output/maze_simple/softmax --lr 0.01 --pretrain --finetune \
    --incre_m 1 --ps_iter 120 --ps_lr 0.01 \
    --ran --fine --prq --incre
