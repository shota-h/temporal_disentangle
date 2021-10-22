for ratio in $1; do
    for th in 0.99; do
    # for th in 0.95 0.99; do
        # python3 FixMatch/train.py --dataset cifar10 --num-labeled 4000 --arch original --batch-size 32 --lr 0.03 --seed 5 --out "results/r${ratio}/few_balance" --mu 1 --num-workers 1 --eval-step 1000 --total-steps 200000 --ratio $ratio --balance_weight --use_mytrans --epochs 400 --use-ema --lambda-u 0.0
        
        # python3 src/temp_consist_drl.py --data colon --semi --ratio $ratio --epoch 200 --ex "results/r${ratio}/few_balance" --test_exist
        
        # python3 FixMatch/train.py --dataset cifar10 --num-labeled 4000 --arch original --batch-size 16 --lr 0.03 --seed 5 --out "results/r${ratio}/balance/strong_th${th}" --mu 1 --num-workers 0 --eval-step 1000 --total-steps 200000 --ratio $ratio --balance_weight --use_mytrans --epochs 400 --threshold $th --use-ema
        
        # python3 src/temp_consist_drl.py --data colon --semi --ratio $ratio --epoch 200 --ex "results/r${ratio}/balance/strong_th${th}" --test_exist
        
        python3 FixMatch/train.py --dataset cifar10 --num-labeled 4000 --arch original --batch-size 16 --lr 0.03 --seed 5 --out "results/r${ratio}/balance/weak_th${th}" --mu 1 --num-workers 0 --eval-step 1000 --total-steps 200000 --ratio $ratio --weak --balance_weight --use_mytrans  --epochs 400 --use-ema --threshold $th
        
        python3 src/temp_consist_drl.py --data colon --semi --ratio $ratio --epoch 200 --ex "results/r${ratio}/balance/weak_th${th}" --test_exist
    done
done