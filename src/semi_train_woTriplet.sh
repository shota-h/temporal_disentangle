python3 src/temporal_vae.py --semi --ratio 0.0 --classifier 1 --adv 1 --batch 32 --epoch 300 --ex exist_c1_a1_t01_r00_v2 --data $1 --mode $2 --step 1 --use_pseudo
python3 src/temporal_vae.py --semi --ratio 0.2 --classifier 1 --adv 1 --batch 32 --epoch 300 --ex exist_c1_a1_t01_r02_v2 --data $1 --mode $2 --step 1 --use_pseudo
python3 src/temporal_vae.py --semi --ratio 0.4 --classifier 1 --adv 1 --batch 32 --epoch 300 --ex exist_c1_a1_t01_r04_v2 --data $1 --mode $2 --step 1 --use_pseudo
python3 src/temporal_vae.py --semi --ratio 0.6 --classifier 1 --adv 1 --batch 32 --epoch 300 --ex exist_c1_a1_t01_r06_v2 --data $1 --mode $2 --step 1 --use_pseudo
python3 src/temporal_vae.py --semi --ratio 0.8 --classifier 1 --adv 1 --batch 32 --epoch 300 --ex exist_c1_a1_t01_r08_v2 --data $1 --mode $2 --step 1 --use_pseudo
python3 src/temporal_vae.py --semi --ratio 1.0 --classifier 1 --adv 1 --batch 32 --epoch 300 --ex exist_c1_a1_t01_r10_v2 --data $1 --mode $2 --step 1 --use_pseudo