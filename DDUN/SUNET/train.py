from DDUN.SUNET.sddpm import *
from DDUN.SUNET.block import *
from DDUN.SUNET.embedding import *
from DDUN.SUNET.unet import *
from DDUN.SUNET.utils import *

def get_data = 


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataloader  = get_dataloader(batch_size=32)