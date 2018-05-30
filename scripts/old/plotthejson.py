import matplotlib.pyplot as plt
import sys, json

with open(sys.argv[1], 'r') as rawdata:
    jsonObj = json.loads(rawdata.read())

description = sys.argv[1].split("data_dumps/training-test-history")[1]
description = description.split(".json")[0]

acc_loss = sys.argv[2]

if(acc_loss == "acc"):
    train = jsonObj["acc"]
    test = jsonObj["val_acc"]
elif(acc_loss == "loss"):
    train = jsonObj["loss"]
    test = jsonObj["val_loss"]
else:
    print("Warning: Please choose whether you would like to display the loss or accuracy function!")
    print("Syntax: python3 plotthejson.py [json file] ['acc'/'loss']")
    sys.exit()

epochs = list(range(1,len(train)+1))

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(epochs, train, s=10, c='b', marker="s", label='train')
ax1.scatter(epochs, test, s=10, c='r', marker="o", label='test')
plt.legend(loc='upper left');
plt.savefig("graphs/braindecodingloss" + description + "_" + acc_loss + "test.png")
