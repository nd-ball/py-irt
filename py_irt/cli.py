import typer
from py_irt.training import training_app


app = typer.Typer()
app.add_typer(training_app, name='train')


if __name__ = '__main__':
    app()