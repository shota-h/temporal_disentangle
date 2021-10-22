for ratio in $1; do
    for loop in `seq 1 $2`; do
    # echo $loop
    # python3 src/temp_consist_drl.py --data colon --train --test --adv 0.0 --semi --ratio $ratio --epoch 200 --classifier 1.0 --step 1 --wopart --ex "miccai2021/r${ratio}/balance/woPart_210705_iter${loop}" --balance_weight

    # python3 src/temp_consist_drl.py --data colon --train --test --adv 0.0 --semi --ratio $ratio --epoch 200 --classifier 1.0 --step 1 --wopart --use_pseudo --T1 100 --T2 150 --ex "miccai2021/r${ratio}/balance/woPart_pseudo_210621_iter${loop}" --balance_weight    

    # only Part
    # python3 src/temp_consist_drl.py --data colon --train --test --adv 0.0 --semi --ratio $ratio --epoch 200 --classifier 1.0 --step 1 --ex "miccai2021/r${ratio}/balance/woDRL_woTri_210621_iter${loop}" --balance_weight
    
    # only Ordinal
    python3 src/temp_consist_drl.py --data colon --train --test --adv 0.0 --semi --ratio $ratio --epoch 200 --classifier 1.0 --step 1 --wopart --triplet --tri 0.01 --margin 1.0 --ex "miccai2021/r${ratio}/balance/woPart_withTri_woSemiHard_woTotal_20210705_iter${loop}" --balance_weight

    # only DRL
    # python3 src/temp_consist_drl.py --data colon --train --test --adv 0.1 --semi --ratio $ratio --epoch 200 --classifier 1.0 --step 1 --ex "miccai2021/r${ratio}/balance/withDRL_woTri_adv01_210621_iter${loop}" --balance_weight 
    
    # with Ord and Part
    # python3 src/temp_consist_drl.py --data colon --train --test --adv 0.0 --semi --ratio $ratio --epoch 200 --classifier 1.0 --step 1  --triplet --tri 0.01 --margin 1.0 --ex "miccai2021/r${ratio}/balance/woDRL_withTri_woSemiHard_woTotal_20210621_iter${loop}" --balance_weight

    # Prop
    # python3 src/temp_consist_drl.py --data colon --train --test --adv 0.1 --semi --ratio $ratio --epoch 200 --classifier 1.0 --step 1 --triplet --tri 0.01 --margin 1.0 --ex "miccai2021/r${ratio}/balance/withDRL_withTri_adv01_woSemiHard_woTotal_210621_iter${loop}" --balance_weight 
done
done