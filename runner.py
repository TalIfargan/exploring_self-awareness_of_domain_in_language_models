import os

# Run the training script for all the domains for 10 epochs and 10 different seeds
for domain in ["automotive", "electronics", "petSupplies"]:
    for seed in range(10):
        os.system(f"python train.py --seed {seed} --batch_size 64 --epoch 3 --model_dir trained_models --dataset_dir data/{domain} --domain_name {domain} --num_train_samples 1000")
        break
    break