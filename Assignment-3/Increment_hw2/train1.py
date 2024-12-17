import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from facades_dataset import FacadesDataset
from FCN_network import FullyConvNetwork  # Assuming this is your generator network
from torch.optim.lr_scheduler import StepLR

# Define the Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels * 2, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=4, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x, y):
        img_input = torch.cat([x, y], dim=1)
        return self.model(img_input)

def tensor_to_image(tensor):
    image = tensor.cpu().detach().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = (image + 1) / 2
    image = (image * 255).astype(np.uint8)
    return image

def save_images(inputs, targets, outputs, folder_name, epoch, num_images=5):
    os.makedirs(f'{folder_name}/epoch_{epoch}', exist_ok=True)
    for i in range(min(num_images, inputs.size(0))):
        input_img_np = tensor_to_image(inputs[i])
        target_img_np = tensor_to_image(targets[i])
        output_img_np = tensor_to_image(outputs[i])

        comparison = np.hstack((input_img_np, target_img_np, output_img_np))
        cv2.imwrite(f'{folder_name}/epoch_{epoch}/result_{i + 1}.png', comparison)

def train_one_epoch(generator, discriminator, dataloader, g_optimizer, d_optimizer, criterion_gan, criterion_l1, device, epoch, num_epochs):
    generator.train()
    discriminator.train()
    running_loss_g = 0.0
    running_loss_d = 0.0

    for i, (image_rgb, image_semantic) in enumerate(dataloader):
        batch_size = image_rgb.size(0)
        real_labels = torch.ones(batch_size, 1, 30, 30).to(device)
        fake_labels = torch.zeros(batch_size, 1, 30, 30).to(device)

        image_rgb = image_rgb.to(device)
        image_semantic = image_semantic.to(device)

        # Generate fake images
        fake_images = generator(image_rgb)

        # Train Discriminator
        d_optimizer.zero_grad()

        # Real images
        real_outputs = discriminator(image_rgb, image_semantic)
        d_loss_real = criterion_gan(real_outputs, real_labels)

        # Fake images
        fake_outputs = discriminator(image_rgb, fake_images.detach())
        d_loss_fake = criterion_gan(fake_outputs, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optimizer.step()

        running_loss_d += d_loss.item()

        # Train Generator
        g_optimizer.zero_grad()

        # Adversarial loss with reduced weight
        gen_outputs = discriminator(image_rgb, fake_images)
        g_loss_adv = 0.1 * criterion_gan(gen_outputs, real_labels)  # Reduced adversarial loss weight

        # L1 Loss with increased weight
        g_loss_l1 = 100 * criterion_l1(fake_images, image_semantic)  # Increased L1 loss weight

        # Total loss
        g_loss = g_loss_adv + g_loss_l1
        g_loss.backward()
        g_optimizer.step()

        running_loss_g += g_loss.item()

        if epoch % 5 == 0 and i == 0:
            save_images(image_rgb, image_semantic, fake_images, 'train_results', epoch)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], G_Loss: {g_loss.item():.4f}, D_Loss: {d_loss.item():.4f}')

def validate(generator, discriminator, dataloader, criterion_gan, criterion_l1, device, epoch, num_epochs):
    generator.eval()
    discriminator.eval()
    val_loss_g = 0.0
    val_loss_d = 0.0

    with torch.no_grad():
        for i, (image_rgb, image_semantic) in enumerate(dataloader):
            batch_size = image_rgb.size(0)
            real_labels = torch.ones(batch_size, 1, 30, 30).to(device)
            fake_labels = torch.zeros(batch_size, 1, 30, 30).to(device)

            image_rgb = image_rgb.to(device)
            image_semantic = image_semantic.to(device)

            fake_images = generator(image_rgb)

            # Discriminator loss
            real_outputs = discriminator(image_rgb, image_semantic)
            d_loss_real = criterion_gan(real_outputs, real_labels)

            fake_outputs = discriminator(image_rgb, fake_images.detach())
            d_loss_fake = criterion_gan(fake_outputs, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            val_loss_d += d_loss.item()

            # Generator loss
            gen_outputs = discriminator(image_rgb, fake_images)
            g_loss_adv = criterion_gan(gen_outputs, real_labels)

            g_loss_l1 = criterion_l1(fake_images, image_semantic)
            g_loss = g_loss_adv +10* g_loss_l1

            val_loss_g += g_loss.item()

            if epoch % 5 == 0 and i == 0:
                save_images(image_rgb, image_semantic, fake_images, 'val_results', epoch)

    avg_val_loss_g = val_loss_g / len(dataloader)
    avg_val_loss_d = val_loss_d / len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation G Loss: {avg_val_loss_g:.4f}, Validation D Loss: {avg_val_loss_d:.4f}')

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_dataset = FacadesDataset(list_file='train_list.txt')
    val_dataset = FacadesDataset(list_file='val_list.txt')

    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=4, drop_last=True)

    generator = FullyConvNetwork().to(device)
    discriminator = Discriminator().to(device)

    criterion_gan = nn.BCELoss()
    criterion_l1 = nn.L1Loss()

    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0005, betas=(0.5, 0.999))  # Increased discriminator learning rate

    scheduler_g = StepLR(g_optimizer, step_size=200, gamma=0.2)
    scheduler_d = StepLR(d_optimizer, step_size=200, gamma=0.2)

    num_epochs = 800
    for epoch in range(num_epochs):
        train_one_epoch(generator, discriminator, train_loader, g_optimizer, d_optimizer, criterion_gan, criterion_l1, device, epoch, num_epochs)
        validate(generator, discriminator, val_loader, criterion_gan, criterion_l1, device, epoch, num_epochs)

        scheduler_g.step()
        scheduler_d.step()

        if (epoch + 1) % 20 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(generator.state_dict(), f'checkpoints/generator_epoch_{epoch + 1}.pth')
            torch.save(discriminator.state_dict(), f'checkpoints/discriminator_epoch_{epoch + 1}.pth')

if __name__ == '__main__':
    main()














