for ratio in 0.9; do
    for iter in 1 2; do
        python3 src/temp_consist_drl.py --data colon --test --adv 0.0 --semi --ratio $ratio --epoch 200 --classifier 1.0 --step 1 --wopart --ex "miccai2021/r${ratio}/balance/woPart_iter${iter}" --balance_weight
            for alpha in 1.5 2.0 3.0;do
            python3 src/temp_consist_drl.py --data colon --test --adv 0.0 --semi --ratio $ratio --epoch 200 --classifier 1.0 --step 1 --wopart --use_pseudo --T1 100 --T2 150 --ex "miccai2021/r${ratio}/balance/woPart_pseudo_alpha${alpha}_iter${iter}" --balance_weight --alpha $alpha
            done    
    done    
done