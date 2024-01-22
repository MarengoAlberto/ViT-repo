from src.runner import Runner
from src.project_setup import get_arguments

args = get_arguments()
print(args)
runner = Runner(args)
runner.train_model()
