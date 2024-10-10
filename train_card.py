from trainer import Trainer
from utils.misc import set_seed

if __name__ == '__main__':
    set_seed(2024)
    trainer = Trainer('configs/train_card.yaml')
    trainer.train(verbose=True)