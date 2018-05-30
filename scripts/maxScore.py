import sys, json

description = sys.argv[1]
R = int(sys.argv[2])

data_file = "../data_dumps/training-test-history-" + description + ".json"

with open(data_file, 'r') as rawdata:
    jsonObj = json.loads(rawdata.read())

test = jsonObj["val_acc"][0:R]

max = [-1,0]

for i in range(R):
    if test[i][1] > max[1]:
        max = test[i]

print(max)
