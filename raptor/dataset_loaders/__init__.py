from .loogle import load_loogle_dataset
from .narrativeqa import load_narrativeqa_dataset
from .novelhopqa import load_novelhopqa_dataset
from .qasper import load_qasper_dataset
from .quality import load_quality_dataset

SUPPORTED_DATASET_LOADERS = {
    "loogle": load_loogle_dataset,
    "narrativeqa": load_narrativeqa_dataset,
    "novelhopqa": load_novelhopqa_dataset,
    "qasper": load_qasper_dataset,
    "quality": load_quality_dataset,
}
