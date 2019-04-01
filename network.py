# @article{morrison2018closing, 
# 	title={Closing the Loop for Robotic Grasping: A Real-time, Generative Grasp Synthesis Approach}, 
# 	author={Morrison, Douglas and Corke, Peter and Leitner, Jürgen}, 
# 	booktitle={Robotics: Science and Systems (RSS)}, 
# 	year={2018} 
# }

import torch
import torch.nn as nn
import torch.nn.functional as F


class GGCNN(nn.Module):
    def __init__(self, input_layer=1):
        super(GGCNN, self).__init__()


        self.conv1 = nn.Conv2d(input_layer, 32, kernel_size=9, stride=(3, 3), padding=3)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=5, stride=(2, 2), padding=2)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=3, stride=(2, 2), padding=1)

        self.conv4 = nn.ConvTranspose2d(8, 8, kernel_size=3, stride=(2, 2), padding=1, output_padding=1)
        self.conv5 = nn.ConvTranspose2d(8, 16, kernel_size=5, stride=(2, 2), padding=2, output_padding=1)
        self.conv6 = nn.ConvTranspose2d(16, 32, kernel_size=9, stride=(3, 3), padding=3)
        
        self.pad = nn.ConstantPad2d((0,1,0,1), 0)
        
        self.pos_out = nn.Conv2d(32, 1, kernel_size=2)
        self.cos_out = nn.Conv2d(32, 1, kernel_size=2)
        self.sin_out = nn.Conv2d(32, 1, kernel_size=2)
        self.width_out = nn.Conv2d(32, 1, kernel_size=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        encoded = F.relu(self.conv3(x))

        x = F.relu(self.conv4(encoded))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))

        x = self.pad(x)

        pos_output = self.pos_out(x)
        cos_output = self.cos_out(x)
        sin_output = self.sin_out(x)
        width_output = self.width_out(x)

        return pos_output, cos_output, sin_output, width_output

#Visualize network architecture
# from torchsummary import summary
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ggcnn=GGCNN(1).to(device)
# summary(ggcnn, input_size=(1, 300, 300))