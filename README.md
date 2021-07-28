# MDC research toolbox
There are some useful tools I used when I developed deep learning applications,
which covers the visualization, profiling, debug, and data preprocessing.

## Install

```shell script
pip install mdc_tools
```

## Command Line Tools

### vpdb 
VSCode debug configuration generator from a python command line.

For example:
```shell
vpdb python hello.py aaa bbb ccc -ddd -eee
```

It will generate the debug configuration in `.vscode/launch.json`. 
Then you can debug your python file by just click the corresponding button.

### video2frames
```shell
mdctools video2frames -i <input video> -o <output folder, default video name>
```
## API
There are some useful function I used when I develop deep learning applications,
which covers the visualization, profiling, and data preprocessing.

### mdc_tools.Timer
```python
from mdc_tools import Timer
with Timer('operation name'):
    x = torch.softmax(x)
```

Output:
```
operation name consumes 0.024 seconds.
```

## TODO

[] Support run file in different directories. 

## Update Log
### 2021.07.08
1. Auto install dependencies for vpdb.
2. Handle the environment variables.