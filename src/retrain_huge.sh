python3 src/temporal_vae.py --batch 32 --epoch 300 --data huge --step 1 --retrain --triplet --ex wo_triplet
wait
python3 src/temporal_vae.py --batch 32 --epoch 300 --data huge --step 1 --rev --retrain --triplet --ex wo_triplet
