CUDA_VISIBLE_DEVICES=2 python3 main.py\
    --n_blocks=1\
    --batch_size=512\
    --window_size=60\
    --train_split=0.8\
    --name=SWaT\
    > SWaT.log 2>&1 &

