from .cli import SybilCLI

from src.model.module import MappingModel
from src.data.data_module import DataModule

from ..taskdataset import TaskDataset

def cli_main():
    cli = SybilCLI(MappingModel, DataModule, run=False)

    cli.datamodule.setup("fit")
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)

    cli.datamodule.setup("predict")
    cli.trainer.predict(cli.model, datamodule=cli.datamodule)

if __name__ == '__main__':
    cli_main()