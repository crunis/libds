# libds
My DataScience library

## Suggested uv commands

### jupyter
PYTHONPATH=$(pwd) uv run -p 3.11 --with-requirements requirements.txt jupyter lab

### pytest watch
PYTHONPATH=. uv run -p 3.11 --with-requirements ../requirements.txt  ptw -v

### Visual Code
PYTHONPATH=. uv run -p 3.11 --with-requirements ../requirements.txt code . 
