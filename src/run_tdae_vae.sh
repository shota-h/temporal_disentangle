python3 src/temporal_vae.py --ex wo_triplet --batch 64 --epoch 300 --data huge --step 10
wait
python3 src/temporal_vae.py --ex wo_triplet --batch 64 --epoch 300 --rev --data huge --step 10
wait
python3 src/temporal_vae.py --ex all_loss --batch 32 --epoch 300 --data huge --step 10 --triplet
wait
python3 src/temporal_vae.py --ex all_loss --batch 32 --epoch 300 --data huge --step 10 --rev --triplet
