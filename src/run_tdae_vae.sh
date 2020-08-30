python3 src/temporal_vae.py --ex wo_triplet --batch 64 --epoch 300 --data huge --step 10 --mode val
wait
python3 src/temporal_vae.py --ex wo_triplet --batch 64 --epoch 300 --data huge --step 10 --rev --mode val
wait
python3 src/temporal_vae.py --ex all_loss --batch 32 --epoch 300 --data huge --step 10 --triplet --mode val
wait
python3 src/temporal_vae.py --ex all_loss --batch 32 --epoch 300 --data huge --step 10 --rev --triplet --mode val
wait

python3 src/temporal_vae.py --ex backup_200827/wo_triplet --batch 64 --epoch 300 --data toy --step 10 --mode val
wait
python3 src/temporal_vae.py --ex backup_200827/wo_triplet --batch 64 --epoch 300 --data toy --step 10 --rev --mode val
wait
python3 src/temporal_vae.py --ex backup_200827/all_loss_m1 --batch 32 --epoch 300 --data toy --step 10 --triplet --mode val
wait
python3 src/temporal_vae.py --ex backup_200827/all_loss_m1 --batch 32 --epoch 300 --data toy --step 10 --rev --triplet --mode val
