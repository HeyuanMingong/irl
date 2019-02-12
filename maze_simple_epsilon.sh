
python main.py --max_epochs 1000 --strategy epsilon --env Maze2D-v1 \
    --output output/maze_simple/epsilon --lr 0.01 --pretrain --finetune \
    --incre_m 1 --ps_iter 10 --ps_lr 1.0 \
    --ran --fine --prq --incre 
