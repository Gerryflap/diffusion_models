import time
from queue import Queue

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, InterpolationMode, Lambda

from image_dataset import ImageDataset

dataset = ImageDataset("/run/media/gerben/LinuxData/data/ffhq_full/images1024x1024/",
                 transform=Compose([
                     ToTensor(),
                     Resize(256, interpolation=InterpolationMode.NEAREST),
                     Lambda(lambda tensor: tensor * 2.0 - 1.0)
                 ]))

dataset = ImageDataset("/run/media/gerben/LinuxData/data/ffhq_thumbnails/thumbnails128x128",
                 transform=Compose([
                     ToTensor(),
                     Lambda(lambda tensor: tensor * 2.0 - 1.0)
                 ]))

# dataset = ImageDataset("/run/media/gerben/LinuxData/data/ffhq_thumbnails/cropped_faces64",
#                  transform=Compose([
#                      ToTensor(),
#                      Resize(64),
#                      Lambda(lambda tensor: tensor * 2.0 - 1.0)
#                  ]))
dataloader = DataLoader(dataset, 64, shuffle=False, drop_last=True, num_workers=1)
exp_avg = None
try:
    while True:
        t_prev = time.time()
        for batch in dataloader:
            t = time.time()
            if exp_avg is None:
                exp_avg = t - t_prev
            else:
                exp_avg = 0.05 * (t - t_prev) + 0.95 * exp_avg
            print("Loading took %.3f seconds, exp_avg: %.3f"%(t - t_prev, exp_avg))
            t_prev = t
except KeyboardInterrupt:
    print("Stopping benchmark...")
