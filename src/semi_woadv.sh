python3 src/temporal_vae.py --semi --ratio 0.0 --classifier 1 --adv 0 --batch 32 --epoch 50 --ex normal_c1_r00_1050_201005 --data $1 --mode $2 --step 1 
python3 src/temporal_vae.py --semi --ratio 0.2 --classifier 1 --adv 0 --batch 32 --epoch 50 --ex normal_c1_r02_1050_201005 --data $1 --mode $2 --step 1 
python3 src/temporal_vae.py --semi --ratio 0.4 --classifier 1 --adv 0 --batch 32 --epoch 50 --ex normal_c1_r04_1050_201005 --data $1 --mode $2 --step 1 
python3 src/temporal_vae.py --semi --ratio 0.6 --classifier 1 --adv 0 --batch 32 --epoch 50 --ex normal_c1_r06_1050_201005 --data $1 --mode $2 --step 1 
python3 src/temporal_vae.py --semi --ratio 0.8 --classifier 1 --adv 0 --batch 32 --epoch 50 --ex normal_c1_r08_1050_201005 --data $1 --mode $2 --step 1 