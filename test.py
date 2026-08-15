import argparse
from utils.test_utils import test_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train preterm prediction model")
    parser.add_argument(
        "save_path",
        help="Full path to training config YAML (info.name sets the run folder)",
    )
    args = parser.parse_args()
    test_model(args.save_path)
