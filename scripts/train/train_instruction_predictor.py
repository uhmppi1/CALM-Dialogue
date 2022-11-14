import hydra
from omegaconf import DictConfig, OmegaConf
from supervised_train_loop import train
import os
@hydra.main(config_path="../../config", config_name="train_instruction_predictor")
def main(cfg : DictConfig):
    
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cfg = OmegaConf.to_container(cfg)
    train(cfg)

if __name__ == "__main__":
    main()