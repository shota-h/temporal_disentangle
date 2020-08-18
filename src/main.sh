python3 src/temporal_disentanglement.py --ex test_multi_rev --d2ae --rev --batch 32 --rec 10
wait
python3 src/temporal_disentanglement.py --ex test_multi --d2ae --batch 64 --rec 10
wait
python3 src/temporal_disentanglement.py --ex test_multi_wo_adv --batch 64 --d2ae --adv 0 --rec 10
wait
python3 src/temporal_disentanglement.py --ex test_multi_wo_adv_rev --batch 64 --d2ae --adv 0 --rev --rec 10
