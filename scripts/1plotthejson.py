import matplotlib.pyplot as plt
import sys, json

description = sys.argv[1]
data_file = "../data_dumps/training-test-history-" + description + ".json"
acc_loss = sys.argv[2]
start_epoch = None
end_epoch = None
if len(sys.argv) > 3:
    start_epoch = int(sys.argv[3])
    end_epoch = int(sys.argv[4])


with open(data_file, 'r') as rawdata:
    jsonObj = json.loads(rawdata.read())

if(acc_loss == "acc"):
    if start_epoch == None:
        train = jsonObj["acc"]
        test = jsonObj["val_acc"]
    else:
        train = jsonObj["acc"][start_epoch-1:end_epoch]
        test = jsonObj["val_acc"][start_epoch-1:end_epoch]
elif(acc_loss == "loss"):
    if start_epoch == None:
        train = jsonObj["loss"]
        test = jsonObj["val_loss"]
    else:
        train =jsonObj["loss"][start_epoch-1:end_epoch]
        test = jsonObj["val_loss"][start_epoch-1:end_epoch]
else:
    print("Warning: Please choose whether you would like to display the loss or accuracy function!")
    print("Syntax: python3 plotthejson.py [json file] ['acc'/'loss']")
    sys.exit()

epochs = [i[0] for i in train]
train = [i[1] for i in train]
test = [i[1] for i in test]

fig = plt.figure()
plt.title(description)
plt.ylabel(acc_loss + " %")
plt.xlabel("epochs")
ax1 = fig.add_subplot(111)

ax1.scatter(epochs, train, s=10, c='b', marker="s", label='train')
ax1.scatter(epochs, test, s=10, c='r', marker="o", label='test')
plt.legend(loc='upper left');
plt.savefig("../graphs/braindecodingloss" + description + "_" + acc_loss + "_" + str(epochs[-1]) + ".png")
