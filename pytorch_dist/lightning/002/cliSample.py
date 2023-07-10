# main.py
from pytorch_lightning.utilities.cli import LightningCLI

# simple demo classes for your convenience
from boring_classes import DemoModel, BoringDataModule


def cli_main():
    #cli = LightningCLI(DemoModel, BoringDataModule)
    cli = LightningCLI(DemoModel)
    # note: don't call fit!!


if __name__ == "__main__":
    cli_main()
    # note: it is good practice to implement the CLI in a function and call it in the main if block