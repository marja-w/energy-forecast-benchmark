import wandb

from model.train import train

if __name__ == "__main__":
    wandb.init(project="anaconda-short-term")
    score, run = train(dict(wandb.config))
    run.finish()