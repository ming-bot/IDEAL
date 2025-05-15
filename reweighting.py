import json
import random
import os

if __name__ == '__main__':
    influence = {
        "Mathematics": 6.58035278e-05,
        "Coding": 7.74383545e-04,
        "bbh": 1.30653381e-04,
        "Instruction": 1.92642212e-04,
        "TrustAI": -4.43458557e-05}
    max_num = 0.15
    beta = max_num / max(abs(value) for value in influence.values())
    for key, value in influence.items():
        influence[key] = value * beta

    control_datasum = False
    original_training = f"/mnt/petrelfs/mingchenlin/DataEvolution/train_dataset_1_M1_M2-2"
    training_data = {}
    old_sum = 0
    for key, value in influence.items():
        with open(os.path.join(original_training, key, "original_train.jsonl"), "r") as f:
            for line in f:
                training_data[key] = training_data.get(key, []) + [json.loads(line)]
        old_sum += len(training_data[key])
    print(old_sum)

    new_sum = 0
    for key, value in influence.items():
        influence[key] = int(len(training_data[key]) * (1 + value))
        new_sum += influence[key]
    print(new_sum)
    
    if control_datasum:
        for key, value in influence.items():
            influence[key] = int(influence[key] * old_sum / new_sum)
    
    for key, value in training_data.items():
        if len(training_data[key]) > influence[key]:
            training_data[key] = random.sample(training_data[key], int(influence[key]))
        else:
            training_data[key] += random.sample(training_data[key], influence[key] - len(training_data[key]))
    
    target_path = f"/mnt/petrelfs/mingchenlin/DataEvolution/train_dataset_1_M1_M2_M3_2"
    for key, value in training_data.items():
        print(key, len(value))
        os.makedirs(os.path.join(target_path, key), exist_ok=True)
        with open(os.path.join(target_path, key, "original_train.jsonl"), "w") as f:
            for line in value:
                f.write(json.dumps(line) + "\n")