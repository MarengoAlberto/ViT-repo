from src.runner import Runner
from src.project_setup import get_arguments


if __name__ == "__main__":
    args = get_arguments()
    runner = Runner(args)
    runner.train_model()

