import json

with open("../data_dumps/training-test-history-source_model.json", "r") as f:
    results = json.load(f)

n = len(results["acc"])

for key, value in results.items():
    for i in range(n):
        for j in range(n-i-1):
            if results[key][j][0] > results[key][j+1][0]:
                results[key][j], results[key][j+1] = results[key][j+1], results[key][j]

print(results)

with open("../data_dumps/training-test-history-source_model.json", "w") as f:
    json.dump(results, f)
