from .loogle import load_loogle_dataset
from .qasper import load_qasper_dataset

SUPPORTED_DATASET_LOADERS = {
    "loogle": load_loogle_dataset,
    "qasper": load_qasper_dataset,
}
