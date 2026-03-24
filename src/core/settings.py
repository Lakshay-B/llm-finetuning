# settings.py

# settings.py using Pydantic BaseSettings
from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path
from dotenv import load_dotenv
from functools import lru_cache
import os

load_dotenv()

class CommonSettings(BaseSettings):
	GEMINIAPIKEY: str = Field("secretkey", env='GEMINIAPIKEY')
	PROJECT_ID: str = Field("projectid", env='PROJECT_ID')
	CUAD_V1: str = Field('CUAD_v1')
	CPT_DATA_DIR: Path = Field(r'training_data\cpt_data')
	INS_FT_DATA_DIR: Path = Field(r'training_data\instruction_ft_data\contracts')
	INS_FT_MLP_DATA_DIR_SUMM: Path = Field(r'training_data\instruction_ft_data\summarization')
	INS_FT_MLP_DATA_DIR_QNA: Path = Field(r'training_data\instruction_ft_data\qna')
	INS_FT_MLP_BATCH_INPUT: Path = Field("batch_input.jsonl")
	INS_FT_MLP_BATCH_RESULTS: Path = Field("batch_results")
	INS_FT_CUAD_DATA_DIR: Path = Field(r'instruction_ft_data\cuad')
	INS_FT_MLP_CONTRACTS_DATASET: Path = Field('contracts_dataset')
	INS_FT_CUAD_TEST_DIR: Path = Field('test_samples')
	INS_FT_CUAD_TRAIN_DIR: Path = Field('train_samples')
	INS_FT_CUAD_MASTER: Path = Field('master_clauses.csv')
	INS_FT_CUAD_QUES_SAMPLES: Path = Field('ques_samples.json')
	INS_FT_CUAD_TEST_RATIO: float = Field(0.1)
	INS_FT_MLP_CUAD_PATT_BATCH: Path = Field("mlp_cuad_batch_input.jsonl")
	MLP_BUFFER_SIZE: int = Field(50000)

class TestSettings(CommonSettings):
	CPT_CONTRACT_SAMPLE_SIZE: int = Field(20)
	CPT_CASELAW_SAMPLE_SIZE: int = Field(20)
	CPT_GEN_SAMPLE_SIZE: int = Field(5)
	CPT_TRAIN_RATIO: float = Field(0.85)
	INS_FT_MLP_CONTRACT_SAMPLE_SIZE: int = Field(20)
	INS_FT_MLP_BUCKET_NAME: str = Field("contract-data-batch")

class LocalSettings(CommonSettings):
	CPT_CONTRACT_SAMPLE_SIZE: int = Field(1500)
	CPT_CASELAW_SAMPLE_SIZE: int = Field(1500)
	CPT_GEN_SAMPLE_SIZE: int = Field(250)
	CPT_TRAIN_RATIO: float = Field(0.85)
	INS_FT_MLP_CONTRACT_SAMPLE_SIZE: int = Field(10000)
	INS_FT_MLP_BUCKET_NAME: str = Field("contract-data-batch")

class ServerSettings(CommonSettings):
	CPT_CONTRACT_SAMPLE_SIZE: int = Field(1500)
	CPT_CASELAW_SAMPLE_SIZE: int = Field(1500)
	CPT_GEN_SAMPLE_SIZE: int = Field(250)
	CPT_TRAIN_RATIO: float = Field(0.85)
	INS_FT_MLP_CONTRACT_SAMPLE_SIZE: int = Field(10000)
	INS_FT_MLP_BUCKET_NAME: str = Field("contract-data-batch")

@lru_cache()
def get_settings():
	"""
	Returns a settings object for the given environment ('local' or 'server').
	Loads variables from .env and sets defaults as needed.
	"""
	env = os.getenv('ENV', 'local')
	if env == 'server':
		return ServerSettings()
	if env == 'test':
		return TestSettings()
	return LocalSettings()

settings = get_settings()

if __name__ == "__main__":
	print("GEMINIAPIKEY:", settings.GEMINIAPIKEY)
	print("PROJECT_ID:", settings.PROJECT_ID)
	print(Path(__file__))
	print(Path(__file__).parent)
	print(Path(__file__).parent.parent / '.env')