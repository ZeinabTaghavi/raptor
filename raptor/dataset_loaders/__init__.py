from .qasper import load_qasper_dataset

SUPPORTED_DATASET_LOADERS = {
    "qasper": load_qasper_dataset,
}
