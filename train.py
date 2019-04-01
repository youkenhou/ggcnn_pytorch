from network import GGCNN
from dataset import CornellDataset
from loss import Loss

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import os

DATA_PATH = 'dataset_190327_0029.h5'    #path to dataset
MODEL_PATH = 'checkpoints'  #the folder where trained models will be saved
BATCH_SIZE = 16
NUM_EPOCHS = 30

train_dataset = CornellDataset(data_path = DATA_PATH, train=True)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_examples = len(train_dataset)
train_batches = len(train_dataloader)

test_dataset = CornellDataset(data_path = DATA_PATH, train=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_examples = len(test_dataset)
test_batches = len(test_dataloader)

ggcnn = GGCNN()
optimizer = optim.Adam(ggcnn.parameters())

print("Train examples: {}".format(train_examples))
print("Train batches: {}".format(train_batches))
print("Evaluation examples: {}".format(test_examples))
print("Start training...")

cudnn.benchmark = True
ggcnn.cuda()

criterion = Loss()

for epoch in range(NUM_EPOCHS):
    print("--------Epoch {}--------".format(epoch))

    ggcnn.train()
    epoch_cost = 0.0

    data_num = 0

    for i,data in enumerate(train_dataloader, 0):
        depth, point, cos, sin, width = data
        depth, point, cos, sin, width = depth.float().cuda(), point.float().cuda(), cos.float().cuda(), sin.float().cuda(), width.float().cuda()
        
        pos_pred, cos_pred, sin_pred, width_pred = ggcnn(depth)

        optimizer.zero_grad()
        
        pos_loss, cos_loss, sin_loss, width_loss, total_loss = criterion(
            pos_pred, point, cos_pred, cos, sin_pred, sin, width_pred, width)

        epoch_cost += total_loss.item()

        total_loss.backward()

        optimizer.step()

        data_num += 1
        print("loss of this batch is %f" % total_loss.item())
    
    epoch_cost = epoch_cost / data_num
    torch.save(ggcnn.state_dict(), os.path.join(MODEL_PATH, 'model_epoch_%d.pth' % epoch))

    print("Epoch %d is done. Now loss is %f" % (epoch, epoch_cost))

    











