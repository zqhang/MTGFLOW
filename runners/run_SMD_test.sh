for file in /home/project/input/processed/machine*train*
do 
    var=${file##*/}
    # echo $var
    echo ${var%_*}
    CUDA_VISIBLE_DEVICES=2 python3 test.py --name=${var%_*} --n_blocks 2 
    wait
done

