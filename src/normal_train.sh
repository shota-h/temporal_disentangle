python3 src/temporal_vae.py --semi --ratio 0.0 --classifier 1 --adv 0 --batch 32 --epoch 100 --ex normal_ratio00 --data $1 --mode $2 --step 1
python3 src/temporal_vae.py --semi --ratio 0.2 --classifier 1 --adv 0 --batch 32 --epoch 100 --ex normal_ratio02 --data $1 --mode $2 --step 1
python3 src/temporal_vae.py --semi --ratio 0.4 --classifier 1 --adv 0 --batch 32 --epoch 100 --ex normal_ratio04 --data $1 --mode $2 --step 1
python3 src/temporal_vae.py --semi --ratio 0.6 --classifier 1 --adv 0 --batch 32 --epoch 100 --ex normal_ratio06 --data $1 --mode $2 --step 1
python3 src/temporal_vae.py --semi --ratio 0.8 --classifier 1 --adv 0 --batch 32 --epoch 100 --ex normal_ratio08 --data $1 --mode $2 --step 1