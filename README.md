# Emb2Pfam

## Installation

First, download the repository or clone it with

```
git clone https://github.com/sinc-lab/emb2pfam.git
```

We recommend using a python virtual environment such as conda or venv. For example, with conda:

```
conda create -n emb2pfam
conda activate emb2pfam
```

Now on this environment, move to the repository folder and install the required packages

```
pip install -r requirements.txt
```

## Setting parameters

Before running the training scripts, you'll need to configure the parameters in the `config.json` file in the config/ folder. Use a text editor to open the `config.json` and edit. it.

### Parameters description

General parameters used to train and test:
- `WINDOW_LEN`: Windows length. Used to extract a window from input embedding. Default value = 32.
- `LABEL_WIN_LEN`: Output windows size. Used to define label size. This label is compared with domain to compute covered percentage. Default value = 32.
- `LR`: Learning rate. Default value = 1e-6.
- `BATCH_SIZE`: Batch size. Used to sample a batch of data. Default value = 32.
- `NEPOCH`: Number of training epochs. Default value = 500.
- `PATIENCE`: Number of epoch of no change. Used to stop training using early stopping. Default value = 10.
- `NWORKERS`: Number of workpfamers. Used to change how many subprocesses use for data loading. Default value = 8.
- `ONLY_SEEDS`: Flag to use only seed data or full labels. Default value = True.
- `MODEL`: Number of model. Used to identify models to ensemble. Default value = 0.
- `CONTINUE_TRAINING`: Boolean value in case there is already a weights.pk file in the folder when the training starts. If set to True, it loads that model and continues from previous progress; if set to False, it starts the training from zero. Default value = False.

Parameters used to `predict_errors.py`.
- `SOFT_MAX`: Flag to use SoftMax function. Default value = False.
- `STEP`: Step for sliding window. Default value = 8
- `MINSCORE`: Threshold. Minimum score to compute True positives. Default value = 0.0.
- `TH`: First threshold. Used to discard family predictions with low scores. Default value = 0.0.
- `USE_MEDFILT`: Flag to use median filter. Default value = True.

Path to obtain the data (download data and rename directories to match these paths):
- `DATA_PATH`: Path to obtain dataframes (.csv files). Default value = `"data/clustered_sequences/"`.
- `EMB_PATH`: Path to obtain embeddings (.pk files). Default value = `"data/clustered_sequences/embeddings/"`.

Parameters to identify data files.
- `SUBSET_NAME`: Pfam version. Used to open categories file. Default value = `"Pfam_v32"`.
- `SUBSET`: Subset of data. Used to open dataframes and fasta files. Default value = `"Pfam_v32_lt1024AA_seed"`.

## Training a model

To train a model, per residue embeddings for train and dev partitions and partitions dataframes are required. Download them from drive folder and save them in `EMB_PATH` and `DATA_PATH` respectively (see [Parameters description](#parameters-description) section).

Once you have all embeddings, you'll need to run the `train.py` script. This script takes two arguments to call via console:
- `-c` or `--config`: Path for `config.json` file. If not specified, default configuration file is `config/config.json`.
- `-o` or `--output`: Output path to save model weights in `weights.pk` and logs in a file called `summary.csv`. Also saves the `config.json` file in this folder for later reference. Example: `results/grid_ESM1b/`.

```
python train.py -c config/config.json -o results/grid_ESM1b/
```

## Testing trained model

To test a model, test embeddings and test dataframes are required. Download from drive folder and save them in same folder as train and dev partitions.

The `test.py` script needs one argument on its call:
- `-i` or `--input`: Input path to obtain the model weights `weights.pk` and . Also adds the test logs to the file `summary.csv`. Example: `results/grid_ESM1b/`.

```
python test.py -i results/grid_ESM1b/
```

If you want to test a model using a sliding window, run the `predict_errors.py` script. This script needs two arguments on its call:
- `-i` or `--input`: Input path to obtain the model weights `weights.pk`. Also adds the test logs to the file `summary.csv`. Example: `results/grid_ESM1b/`.
- `-o` or `--output`: Name for the .csv file where errors will be saved. Saves it inside the input folder in a folder named `errors`. Example: `errors_experiment` (errors will then be found in `results/grid_ESM1b/errors/errors_experiment.csv`).

```
python predict_errors.py -i results/grid_ESM1b/ -o errors_experiment
```

## Integrated run scripts

In the folder `scripts/` you can find an example of a script to run the training and testing in just one call. This is to simplify the experiment and you can use the one provided or make your own using it as an example.

As seen in the default `script.py` provided, you can set the experiment's folder name and the path to the configuration file to be used. The .json configuration file can be modified via script as shown in lines 32-39 or you can comment this part and modify it directly with a text editor.

Then, when you run `script.py`, the script will consecutively run the `train.py`, `test.py` and `predict_errors.py` with the set configuration file. The `predict_errors.py` script will run several times with different configurations of steps and softmax values (you can also modify this to run only one time or with different values instead of the values set by default).

Afterwards, you can check `summary.csv` in the results folder specified to see the logs for every execution. In this folder, you will also find the final weights of the model `weight.pk`, a copy of the script `run_line.py`, a copy of the configuration file `config.json` and the folder `errors/`, which contains the .csv files corresponding to the errors found with `parameters.py` with different parameter configurations.

```
python scripts/run_line.py
```