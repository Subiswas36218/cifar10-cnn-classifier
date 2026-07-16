from src.config import Config
from src.utils import AverageMeter, set_seed

Config.create_directories()

set_seed()

meter = AverageMeter()

meter.update(0.8)

meter.update(0.6)

meter.update(0.4)

print("Average:", meter.average)
print("Device:", Config.DEVICE)
