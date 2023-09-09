import torch
import torchvision
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_availiable() else "cpu")

image_size = [1, 28, 28] #数据集都是28x28的单通道灰度图
dataset = torchvision.datasets.MNIST(
	"mnist_data",
	train=True,
	download=True,
	transform=torchvision.transforms.Compose([torchvision.transforms.Resize(28),torchvision.transforms.ToTensor(),])
)
toPIL = torchvision.transforms.ToPILImage()

batch_size = 4
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

latent_dim = 100  # 噪声（随意赋值）
class Generator(nn.Module):  # 生成器输入噪声，输出假图片
	def __init__(self):
		super(Generator, self).__init__()
		self.model = nn.Sequential(
			nn.Linear(latent_dim, 128),  # 这里输入噪声latent_dim
			nn.BatchNorm1d(128), #批量归一化处理
			nn.GELU(), #激活函数

			nn.Linear(128, 256),
			nn.BatchNorm1d(256),
			nn.GELU(),

			nn.Linear(256, 512),
			nn.BatchNorm1d(512),
			nn.GELU(),

			nn.Linear(512, 1024),
			nn.BatchNorm1d(1024),
			nn.GELU(),

			nn.Linear(1024, np.prod(image_size, dtype=np.int32)),
			nn.Sigmoid()
		)

	def forward(self, z):
		# shape of z: [batchsize, latent_dim]
		output = self.model(z)
		image = output.reshape(z.shape[0], *image_size)
		return image

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()

		self.model = nn.Sequential(
			nn.Linear(np.prod(image_size, dtype=np.int32), 512),
			nn.GELU(),
			nn.Linear(512, 256),
			nn.GELU(),
			nn.Linear(256, 128),
			nn.GELU(),
			nn.Linear(128, 64),
			nn.GELU(),
			nn.Linear(64, 32),
			nn.GELU(),
			nn.Linear(32, 1),
			nn.Sigmoid(),
		)

	def forward(self, image):
		prob = self.model(image.reshape(image.shape[0], -1))
		return prob

generator = Generator()
discriminator = Discriminator()

g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0003, betas=(0.4,0.8), weight_decay=0.0001)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0003, betas=(0.4,0.8), weight_decay=0.0001)

loss_fn = nn.BCELoss()
labels_one = torch.ones(batch_size, 1)
labels_zero = torch.zeros(batch_size, 1)

writer = SummaryWriter("logs")
epoch = 200
step = 0
for n in range(epoch):
	print("----第"+str(n)+"轮训练开始----")
	for img, label in dataloader:
		img = img.to(device)
		label = label.to(device)

		step+=1
		z = torch.randn(batch_size, latent_dim)
		fake_images = generator(z)

		g_loss = loss_fn(discriminator(fake_images), labels_one)
		g_optimizer.zero_grad()
		g_loss.backward()
		g_optimizer.step()

		real_loss = loss_fn(discriminator(img), labels_one)
		fake_loss = loss_fn(discriminator(fake_images.detach()), labels_zero)
		d_loss = real_loss +fake_loss
		d_optimizer.zero_grad()
		d_loss.backward()
		d_optimizer.step()

		if step % 50 == 0:
			print("训练步数："+str(step)+"，生成器损失："+str(g_loss.item())+"判别器损失："+str(d_loss.item()))
			writer.add_scalar("g_loss", g_loss.item(), step)
			writer.add_scalar("d_loss", d_loss.item(), step)
		if step % 200 == 0:
			image = fake_images[0]
			PILImage = toPIL(image)
			PILImage.show()
			writer.add_image("generated image", image, step)