"""
Models Architectures
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, channels=1, classes=10, img_size=32):
        """
        Args:
            channels (int): Number of channels of the input image.
            classes (int): Number of classes for classification.
            img_size (int): Size of the input image, assuming square images.
        """
        super(ConvNet, self).__init__()

        ## Feature extraction

        # Initialize convolutional layers
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        ## Classification
    
        self.fc1 = nn.Linear(128 * 5 * 5, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, classes)
        
        # Initialize dropout layer
        self.dropout = nn.Dropout(p=0.2)  
        
        # Initialize weights using Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        #print(f"Classifier Input shape: {x.shape}")

        ## Feature extraction

        x = self.max_pool(F.relu(self.conv1(x)))
        #print(f"After conv1 shape: {x.shape}")
        x = self.max_pool(F.relu(self.conv2(x)))
        #print(f"After conv2 shape: {x.shape}")
        x = x.view(-1, 128 * 5 * 5)
        #print(f"After view shape: {x.shape}")

        ## Classification
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x) 
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        #print(f"Output shape: {x.shape}")

        return x

def conv_block(c_in, c_out, k_size=4, stride=2, pad=1, use_bn=True, transpose=False):
    module = []
    if transpose:
        module.append(nn.ConvTranspose2d(
            c_in, c_out, k_size, stride, pad, bias=not use_bn))
    else:
        module.append(nn.Conv2d(c_in, c_out, k_size,
                      stride, pad, bias=not use_bn))
    if use_bn:
        module.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*module)


class Generator(nn.Module):
    def __init__(self, z_dim=10, num_classes=10, label_embed_size=5, channels=1, conv_dim=64):
        super(Generator, self).__init__()
        #print("Generator number of channels :", channels)
        self.label_embedding = nn.Embedding(num_classes, label_embed_size)
        self.tconv1 = conv_block(
            z_dim + label_embed_size, conv_dim * 4, pad=0, transpose=True)
        self.tconv2 = conv_block(conv_dim * 4, conv_dim * 2, transpose=True)
        self.tconv3 = conv_block(conv_dim * 2, conv_dim, transpose=True)
        self.tconv4 = conv_block(
            conv_dim, channels, transpose=True, use_bn=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0.0, 0.02)

            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, label):
        #print("Generator x", x.shape)
        #print("Generator label", label.shape)
        x = x.reshape([x.shape[0], -1, 1, 1])
        label_embed = self.label_embedding(label)
        label_embed = label_embed.reshape([label_embed.shape[0], -1, 1, 1])
        #print("Generator label emb", label_embed.shape)
        x = torch.cat((x, label_embed), dim=1)
        x = F.relu(self.tconv1(x))
        x = F.relu(self.tconv2(x))
        x = F.relu(self.tconv3(x))
        x = torch.tanh(self.tconv4(x))
        #print("Generator out", x.shape)
        return x


class Discriminator(nn.Module):
    def __init__(self, num_classes=10, channels=1, conv_dim=64):
        super(Discriminator, self).__init__()
        #print("Discriminator number of channels :", channels)
        self.image_size = 32
        self.label_embedding = nn.Embedding(
            num_classes, self.image_size*self.image_size)
        self.conv1 = conv_block(channels + 1, conv_dim, use_bn=False)
        self.conv2 = conv_block(conv_dim, conv_dim * 2)
        self.conv3 = conv_block(conv_dim * 2, conv_dim * 4)
        self.conv4 = conv_block(conv_dim * 4, 1, k_size=4,
                                stride=1, pad=0, use_bn=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)

            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, label):
        alpha = 0.2
        #print("Discriminator x", x.shape)
        #print("Discriminator label", label.shape)
        label_embed = self.label_embedding(label)
        label_embed = label_embed.reshape(
            [label_embed.shape[0], 1, self.image_size, self.image_size])
        #print("Discriminator emb", label_embed.shape)
        x = torch.cat((x, label_embed), dim=1)
        #print("Discriminator x_cat", x.shape)
        x = F.leaky_relu(self.conv1(x), alpha)
        x = F.leaky_relu(self.conv2(x), alpha)
        x = F.leaky_relu(self.conv3(x), alpha)
        x = torch.sigmoid(self.conv4(x))
        #print("Discriminator out", x.squeeze().shape)
        return x.squeeze()
