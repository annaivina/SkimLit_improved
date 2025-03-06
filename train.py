from src.data_loader import DataLoader
from src.load_config import load_config
import argparse 


def main(experiment_name, config_path):

    config = load_config(config_path)
    data_loader = DataLoader(dataset_name=config['dataset'],num_classes=config['num_classes'], lable_map=config['labels'], batch_size=config['batch_size'])
    train_data, valid_data, test_data = data_loader.load_data()

    








if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Train the model SkimLit to efficiencty identify the abstract devisions")
    parser.add_argument("--experimnet_name", type=str, default="skimlit_test")
    parser.add_argument("--config_path", type=str, default="configs/skimlit.yaml")

    args = parser.parse_args()
    main(experiment_name=args.experiment_name, config_path=args.config_path)