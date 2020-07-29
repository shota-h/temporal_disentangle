# python3 src/cross_disentangle.py --epoch 300 --data toy --ex all_loss
# python3 src/cross_disentangle.py --epoch 300 --data toy --ex only_reconst --classifier 0 --adv 0
# python3 src/cross_disentangle.py --epoch 300 --data toy --ex wo_adv --adv 0
python3 src/cross_disentangle.py --epoch 2000 --data colon --ex all_loss
python3 src/cross_disentangle.py --epoch 2000 --data colon --ex only_reconst --classifier 0 --adv 0
python3 src/cross_disentangle.py --epoch 2000 --data colon --ex wo_adv --adv 0