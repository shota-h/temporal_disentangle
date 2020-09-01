python3 src/temporal_vae.py --ex wo_triplet_adapt --batch 64 --epoch 1000 --data colon --ndeconv 3 --step 1 --full --classifier 1 --adv 1 --rec 1 --adapt
wait
python3 src/temporal_vae.py --ex all_loss_adapt --batch 32 --epoch 1000 --data colon --ndeconv 3 --step 1 --full --classifier 1 --adv 1 --rec 1 --tri 1 --triplet --adapt
wait
python3 src/temporal_vae.py --ex wo_triplet_c1 --batch 64 --epoch 1000 --data colon --ndeconv 3 --step 1 --full --classifier 1
wait
python3 src/temporal_vae.py --ex all_loss_c1 --batch 32 --epoch 1000 --data colon --ndeconv 3 --step 1 --full --classifier 1 --triplet