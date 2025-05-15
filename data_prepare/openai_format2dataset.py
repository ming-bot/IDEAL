import json

with open('/mnt/hwfile/opendatalab/air/mingchenlin/data/finalData/train_5000_0.0_v1.json', 'r') as f:
    data = json.load(f)

output = []
for item in data:
    output.append({
        "instruction": item["messages"][0]["content"],
        "input": "",
        "output": item["messages"][1]["content"]
    })

with open('/mnt/hwfile/opendatalab/air/mingchenlin/data/finalData/train_5000.json', 'w') as f:
    json.dump(output, f, ensure_ascii=False, indent=4)