
python main.py --max_epochs 1000 --strategy epsilon --env Maze2D-v2 \
    --pretrain --finetune --output output/maze_complex/epsilon --lr 0.01 \
    --incre_m 4 --ps_iter 10 --ps_lr 1.0 --nu 0.90 \
    --ran --fine --prq --incre
