# TOPIC MODELLING.

## Create Data Directory.

Steps:

- `mkdir data`
- `mkdir -p data/corpus`
- `mkdir -p data/embedded`
- Extract all the documents in `./data`. (**EXPERIMENTAL**)
    - `dvc pull`
- If the above steps fails, please raise an issue on this repository.

## Create Logs Directory.

Steps:

- `mkdir logs`

## Create Results Directory.

Steps:

- `mkdir res`

## Execute the scripts.

- Executing Latent Dirichlet Allocation with TF-IDF vectorization.
    - `python simple_topic_modelling.py`

### REFERENCES
 1. Topic modelling with Word Embedding, F. Esposito, A. Corraza, F. Cutugno, *Third Italian Conference on Computational Linguistics CLiC-it 2016*