for ratio in $1; do
    python3 src/temp_consist_drl.py --data colon --train --test --adv 0.0 --semi --ratio $ratio --epoch 200 --classifier 1.0 --step 1 --wopart --ex "miccai2021/r${ratio}/woPart"
    
    python3 src/temp_consist_drl.py --data colon --train --test --adv 0.0 --semi --ratio $ratio --epoch 200 --classifier 1.0 --step 1 --wopart --use_pseudo --T1 100 --T2 150 --ex "miccai2021/r${ratio}/woPart_pseudo"

    # python3 src/temp_consist_drl.py --data colon --train --test --adv 0.0 --semi --ratio $ratio --epoch 200 --classifier 1.0 --step 1 --wopart --triplet --tri 0.01 --margin 1.0 --ex "miccai2021/r${ratio}/woPart_withTri_woSemiHard_woTotal"
    
    # python3 src/temp_consist_drl.py --data colon --train --test --adv 0.0 --semi --ratio $ratio --epoch 200 --classifier 1.0 --step 1 --ex "miccai2021/r${ratio}/woDRL_woTri"

    # python3 src/temp_consist_drl.py --data colon --train --test --adv 0.0 --semi --ratio $ratio --epoch 200 --classifier 1.0 --step 1 --triplet --tri 0.01 --margin 1.0 --ex "miccai2021/r${ratio}/woDRL_withTri_woSemiHard_woTotal"

    # for adv in 1.0 0.5 0.01; do
    for adv in 0.1; do
        python3 src/temp_consist_drl.py --data colon --train --test --adv $adv --semi --ratio $ratio --epoch 200 --classifier 1.0 --step 1 --ex "miccai2021/r${ratio}/withDRL_woTri_adv${adv}" 
    
        python3 src/temp_consist_drl.py --data colon --train --test --adv $adv --semi --ratio $ratio --epoch 200 --classifier 1.0 --step 1 --triplet --tri 0.01 --margin 1.0 --ex "miccai2021/r${ratio}/withDRL_withTri_adv${adv}_woSemiHard_woTotal"
    done
done