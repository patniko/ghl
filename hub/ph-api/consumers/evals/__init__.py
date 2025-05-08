"""
Evaluation consumers for processing different file types.
"""

from consumers.evals.dicom_consumer import process_dicom_file
from consumers.evals.csv_consumer import process_csv_file
from consumers.evals.mp4_consumer import process_mp4_file
from consumers.evals.npz_consumer import process_npz_file
from consumers.evals.json_consumer import process_json_file

__all__ = [
    "process_dicom_file",
    "process_csv_file",
    "process_mp4_file",
    "process_npz_file",
    "process_json_file",
]
