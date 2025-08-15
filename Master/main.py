import os
import sys
import argparse
import logging
import traceback
from huggingface_hub import HfApi
from dotenv import load_dotenv


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='pipeline.log')
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','Master'))
hf_token = os.getenv('HF_TOKEN')

logging.info(f"Base Path {base_path}")
sys.path.append(base_path)

if not hf_token:
  logging.error("HF_TOKEN not found")
  sys.exit(1)

api = HfApi(token = hf_token)
try:
  user = api.whoami()
  logging.error(f"Authenticated as :{user['name']}")
except Exception as ex:
  logging.error(f"TokenError: {ex}")
  sys.exit(1)


parser = argparse.ArgumentParser(description = 'To Run a specific job in a pipileine')
parser.add_argument('--job', type=str, required=True, choices=['register','prepare','modelbuilding','deploy'], help ='Job to execute: register, prepare, modelbuilding and deploy in hugging face')

args = parser.parse_args()

data_dir = os.path.join(base_path,'Data')
logging.info(f"data_dir: {data_dir}")
if os.path.exists(data_dir):
  os.makedirs(data_dir, exist_ok=True)

model_dir = os.path.join(base_path,'Model_Dump_JOBLIB')
logging.info(f"model_dir: {model_dir}")
if os.path.exists(model_dir):
  os.makedirs(model_dir, exist_ok=True)

if args.job == 'register':
  from DataRegistration import DataRegistration
  ObjDataReg = DataRegistration(base_path,hf_token)
  if not ObjDataReg.ToRunPipeline():
    logging.error("DataRegistration failed")
    sys.exit(1)
  else:
    logging.info("DataRegistration completed in Hugging Face")

if args.job == 'prepare':
  from DataPrepration import DataPrepration
  ObjDataprep = DataPrepration(base_path,hf_token)
  if not ObjDataprep.ToRunPipeline():
    logging.error("Data Registration failed")
    sys.exit(1)
  else:
    logging.info("Data prepration completed")

if args.job == 'modelbuilding':
  from ModelBuilding import BuildingModels
  ObjModelBuild = BuildingModels(base_path, hf_token)
  if not ObjModelBuild.ToRunPipeline():
    logging.error("Model Building failed")
    sys.exit(1)
  else:
    logging.info("Model Building Completed")





