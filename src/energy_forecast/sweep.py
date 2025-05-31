import wandb

from model.train import train

if __name__ == "__main__":
    wandb.init(project="anaconda-short-term")
    config = dict(wandb.config)
    config["log"] = True
    config["plot"] = False
    config["scale_mode"] = "individual"
    run = train(config)
    run.finish()