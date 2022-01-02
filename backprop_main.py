import backprop_data
import backprop_network
import numpy as np
import matplotlib.pyplot as plt
import pickle

learn_rates = [0.001, 0.01, 0.1, 1, 10, 100]
train_accs = []
train_losses = []
test_accs = []
for i in range(len(learn_rates)):
    training_data, test_data = backprop_data.load(train_size=10000,test_size=5000)
    net = backprop_network.Network([784, 40, 10])
    train_accuracy, train_loss, test_accuracy = net.SGD(training_data, epochs=3, mini_batch_size=10, learning_rate=learn_rates[i], test_data=test_data)
    train_accs.append(train_accuracy)
    train_losses.append(train_loss)
    test_accs.append(test_accuracy)
# with open('train_accs.pkl', 'wb') as f:
#     pickle.dump(train_accs, f, protocol=pickle.HIGHEST_PROTOCOL)
# with open('train_losses.pkl', 'wb') as f:
#     pickle.dump(train_losses, f, protocol=pickle.HIGHEST_PROTOCOL)
# with open('test_accs.pkl', 'wb') as f:
#     pickle.dump(test_accs, f, protocol=pickle.HIGHEST_PROTOCOL)

epochs = np.arange(1, len(train_accuracy)+1)
plt.plot(epochs, np.array(train_accs).transpose())
plt.title('Train Accuracy')
plt.xlabel('Epoch number')
plt.ylabel('Accuracy')
plt.legend(learn_rates)
plt.show()

plt.plot(epochs, np.array(train_losses).transpose())
plt.title('Train Loss')
plt.xlabel('Epoch number')
plt.ylabel('Accuracy')
plt.legend(learn_rates)
plt.show()

plt.plot(epochs, np.array(test_accs).transpose())
plt.title('Test Accuracy')
plt.xlabel('Epoch number')
plt.ylabel('Accuracy')
plt.legend(learn_rates)
plt.show()
