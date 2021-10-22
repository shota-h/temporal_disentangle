for xi in 1.0 0.1; do
    for eps in 0.9 0.8 0.7 0.6 0.5; do
        python3 src/train_vat.py --data colon --test --semi --ratio 0.9 --epoch 200 --classifier 1.0 --step 1 --ex "miccai2021/r0.9/VAT_eps${eps}_xi${xi}" --eps $eps --xi $xi
    done
done
# True version
# for ratio in 0.9 0.8 0.7 0.6 0.5; do
#     python3 src/temp_consist_drl.py --data colon --train --test --adv 0.0 --semi --ratio $ratio --epoch 200 --classifier 1.0 --step 1 --wopart --ex "miccai2021/r${ratio}/woPart"
#     python3 src/temp_consist_drl.py --data colon --train --test --adv 0.0 --semi --ratio $ratio --epoch 200 --classifier 1.0 --step 1 --wopart --use_pseudo --T1 100 --T2 150 --ex "miccai2021/r${ratio}/woPart_pseudo"
#     python3 src/temp_consist_drl.py --data colon --train --test --adv 0.0 --semi --ratio $ratio --epoch 200 --classifier 1.0 --step 1  --wopart --ex "miccai2021/r${ratio}/woPart_withTri" --total
#     python3 src/temp_consist_drl.py --data colon --train --test --adv 0.0 --semi --ratio $ratio --epoch 200 --classifier 1.0 --step 1 --ex "miccai2021/r${ratio}/woDRL_woTri"
#     python3 src/temp_consist_drl.py --data colon --train --test --adv 0.1 --semi --ratio $ratio --epoch 200 --classifier 1.0 --step 1 --ex "miccai2021/r${ratio}/withDRL_woTri_adv01"
#     python3 src/temp_consist_drl.py --data colon --train --test --adv 0.0 --semi --ratio $ratio --epoch 200 --classifier 1.0 --step 1 --triplet --tri 0.01 --margin 1.0 --ex "miccai2021/r${ratio}/woDRL_withTri_woSemiHard_withTotal" --total
#     python3 src/temp_consist_drl.py --data colon --train --test --adv 0.1 --semi --ratio $ratio --epoch 200 --classifier 1.0 --step 1 --triplet --tri 0.01 --margin 1.0 --ex "miccai2021/r${ratio}/withDRL_withTri_adv01_woSemiHard_withTotal" --total
# done