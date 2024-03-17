from lightning.pytorch.cli import LightningCLI

from src.model.module import MappingModel
from src.data.data_module import DataModule

def cli_main():
    cli = LightningCLI(MappingModel, DataModule, run=False)

    cli.datamodule.setup("fit")
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(cli.model, datamodule=cli.datamodule)

if __name__ == '__main__':
    cli_main()