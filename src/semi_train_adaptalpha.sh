python3 src/temp_consist_drl.py --semi --ratio 0.0 --classifier 1 --adv 0.1 --triplet --batch 32 --tri 0.01 --margin 1 --epoch 100 --ex dual_prop_adaptalpha_c1_a01_t001_r00_201020 --data $1 --train --val --test --step 1 --use_pseudo --adapt_alpha --T1 30 --T2 60 --alpha 2 --dual
python3 src/temp_consist_drl.py --semi --ratio 0.2 --classifier 1 --adv 0.1 --triplet --batch 32 --tri 0.01 --margin 1 --epoch 100 --ex dual_prop_adaptalpha_c1_a01_t001_r02_201020 --data $1 --train --val --test --step 1 --use_pseudo --adapt_alpha --T1 30 --T2 60 --alpha 2 --dual
python3 src/temp_consist_drl.py --semi --ratio 0.4 --classifier 1 --adv 0.1 --triplet --batch 32 --tri 0.01 --margin 1 --epoch 100 --ex dual_prop_adaptalpha_c1_a01_t001_r04_201020 --data $1 --train --val --test --step 1 --use_pseudo --adapt_alpha --T1 30 --T2 60 --alpha 2 --dual
python3 src/temp_consist_drl.py --semi --ratio 0.6 --classifier 1 --adv 0.1 --triplet --batch 32 --tri 0.01 --margin 1 --epoch 100 --ex dual_prop_adaptalpha_c1_a01_t001_r06_201020 --data $1 --train --val --test --step 1 --use_pseudo --adapt_alpha --T1 30 --T2 60 --alpha 2 --dual

python3 src/temp_consist_drl.py --semi --ratio 0.0 --classifier 1 --adv 0.1 --triplet --batch 32 --tri 0.01 --margin 1 --epoch 100 --ex prop_adaptalpha_c1_a01_t001_r00_201020 --data $1 --train --val --test --step 1 --use_pseudo --adapt_alpha --T1 30 --T2 60 --alpha 2
python3 src/temp_consist_drl.py --semi --ratio 0.2 --classifier 1 --adv 0.1 --triplet --batch 32 --tri 0.01 --margin 1 --epoch 100 --ex prop_adaptalpha_c1_a01_t001_r02_201020 --data $1 --train --val --test --step 1 --use_pseudo --adapt_alpha --T1 30 --T2 60 --alpha 2
python3 src/temp_consist_drl.py --semi --ratio 0.4 --classifier 1 --adv 0.1 --triplet --batch 32 --tri 0.01 --margin 1 --epoch 100 --ex prop_adaptalpha_c1_a01_t001_r04_201020 --data $1 --train --val --test --step 1 --use_pseudo --adapt_alpha --T1 30 --T2 60 --alpha 2
python3 src/temp_consist_drl.py --semi --ratio 0.6 --classifier 1 --adv 0.1 --triplet --batch 32 --tri 0.01 --margin 1 --epoch 100 --ex prop_adaptalpha_c1_a01_t001_r06_201020 --data $1 --train --val --test --step 1 --use_pseudo --adapt_alpha --T1 30 --T2 60 --alpha 2
python3 src/temp_consist_drl.py --semi --ratio 0.8 --classifier 1 --adv 0.1 --triplet --batch 32 --tri 0.01 --margin 1 --epoch 100 --ex prop_adaptalpha_c1_a01_t001_r08_201020 --data $1 --train --val --test --step 1 --use_pseudo --adapt_alpha --T1 30 --T2 60 --alpha 2