import torch
import torchvision
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

toPIL = torchvision.transforms.ToPILImage()
latent_dim = 100
#image_size = [1, 28, 28]  #给digit用的
image_size = [3, 32, 32]

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

generator = Generator()
generator.load_state_dict(torch.load("HumanFace/generator60.pth", map_location=torch.device('cpu')))
generator.eval()

z = torch.randn(1, latent_dim)
fake_image = generator(z)
fake_image = torch.reshape(fake_image, image_size)
PILImage = toPIL(fake_image)
PILImage.show()