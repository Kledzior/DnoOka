import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image


def load_images(folder, filenames, resize=(256, 256)):
    imgs = []
    for f in filenames:
        path = os.path.join(folder, f)
        img = Image.open(path).convert('RGB').resize(resize)
        imgs.append(np.array(img))
    return np.array(imgs)

def load_masks(folder, filenames, resize=(256, 256)):
    masks = []
    for f in filenames:
        tif_name = f.replace('.jpg', '.tif')
        path = os.path.join(folder, tif_name)
        img = Image.open(path).convert('L').resize(resize)
        masks.append(np.array(img) / 255)
    return np.array(masks)

def load_fieldOfView(folder, filenames, resize=(256, 256)):
    fovs = []
    for f in filenames:
        tif_name = f.replace('.jpg', '.tif')
        path = os.path.join(folder, tif_name)
        img = Image.open(path).convert('L').resize(resize)
        fovs.append(np.array(img) / 255)
    return np.array(fovs)



class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice




class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        self.enc1 = nn.Sequential(CBR(in_channels, 64), CBR(64, 64))
        self.enc2 = nn.Sequential(CBR(64, 128), CBR(128, 128))
        self.enc3 = nn.Sequential(CBR(128, 256), CBR(256, 256))
        self.enc4 = nn.Sequential(CBR(256, 512), CBR(512, 512))

        self.pool = nn.MaxPool2d(2)

        self.middle = nn.Sequential(CBR(512, 1024), CBR(1024, 512))

        self.up4 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.dec4 = nn.Sequential(CBR(1024, 512), CBR(512, 256))

        self.up3 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.dec3 = nn.Sequential(CBR(512, 256), CBR(256, 128))

        self.up2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.dec2 = nn.Sequential(CBR(256, 128), CBR(128, 64))

        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = nn.Sequential(CBR(128, 64), CBR(64, 64))

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        m = self.middle(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(m), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.final(d1))


class SegmentationDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images.astype(np.float32)
        self.masks = masks.astype(np.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx] / 255.0  # Normalizacja [0, 1]
        y = self.masks[idx]

        x = torch.tensor(x, dtype=torch.float32).permute(2, 0, 1)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        return x, y




def train(images, masks, epochs=10, batch_size=8, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SegmentationDataset(images, masks)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = UNet(in_channels=3, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = DiceLoss()

    average_losses = []  # <-- lista do zapisu strat

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for i, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (i + 1) % 10 == 0 or (i + 1) == len(loader):
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(loader)}], Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(loader)
        average_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}] finished. Average Loss: {avg_loss:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), average_losses, marker='o', color='b')
    plt.title('Average Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return model


def show_prediction(model, image, fov):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(image / 255.0).permute(2, 0, 1).unsqueeze(0).float()
        pred = model(x).squeeze().cpu().numpy()

        masked_pred = pred * fov  # Maska pola widzenia

    plt.figure(figsize=(12,5))
    plt.subplot(1,3,1)
    plt.title("Input Image")
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.title("Prediction (masked)")
    plt.imshow(masked_pred, cmap='gray')
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.title("Thresholded")
    plt.imshow(masked_pred > 0.5, cmap='gray')
    plt.axis('off')

    plt.show()




def UNET(dummy_arg=None):
    filenames = [f"{i:02d}_h.jpg" for i in range(1,16)]
    filenamesFOV = [f"{i:02d}_h_mask.jpg" for i in range(1,16)]
    images = load_images("images", filenames)
    groundTruth = load_masks("groundTruth", filenames)
    fieldOfView = load_fieldOfView("fieldOfView", filenamesFOV)

    print("images shape:", images.shape)
    print("groundTruth shape:", groundTruth.shape)
    print("fieldOfView shape:", fieldOfView.shape)

    # dalej możesz użyć images i groundTruth do trenowania:
    model = train(images, groundTruth, epochs=50,batch_size=12)

    # i np. pokazać wynik na pierwszym obrazku
    show_prediction(model, images, fieldOfView[0])




if __name__ == "__main__":
    UNET()
    


