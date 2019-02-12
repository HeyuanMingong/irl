
python main.py --max_epochs 5000 --strategy softmax --env Maze2D-v2 \
    --pretrain --finetune --output output/maze_complex/softmax --lr 0.01 \
    --incre_m 3 --ps_iter 50 --ps_lr 0.1 --nu 0.9 \
    --ran --fine --prq --incre
