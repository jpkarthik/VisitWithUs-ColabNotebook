import os
import sys
import argparse
import logging
import traceback
from huggingface_hub import HfApi


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='pipeline.log')
base_path = ""
hf_token = ""

try:
  if 'google.colab' in sys.modules or 'COLAB_GPU' in os.environ:
    try:
      from google.colab import drive
      #drive.mount('/content/drive/')
      base_path = '/content/drive/MyDrive/PGP_AI_ML_GREAT_LEARNING/10_Advance_Machine_Learning_And_MLOps/Final_Project/VisitWithUs-Tourism_version_1_1/Master/'
    except Exception as ex:
      logging.error(f"Drive Mount exception: {ex}")
      traceback.print_exc
      raise
    
    try:
      from google.colab import userdata
      hf_token = userdata.get('HF_TOKEN')
      print(hf_token)
    except Exception as ex:
      logging.error(f"HF_TOKEN NOT FOUND: {ex}")
      traceback.print_exc
      raise

  else:
    try:
      base_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','Master'))
    except Exception as ex:
      logging.error(f"Exception in getting base_path {base_path}")
      traceback.print_exc
    
    try:
      hf_token = os.getenv('HF_TOKEN')
    except Exception as ex:
      logging.error(f"Exception in getting HF_TOKEN {ex}")
      traceback.print_exc

  print(f"Base_path {base_path}")
  logging.info(f"Base Path: {base_path}")
  sys.path.append(base_path)

except Exception as ex:
  print(f"Exception occured in getting the base path and hf token: {ex}")
  traceback.print_exc()
  logging.error(f"Exception occured in getting the base path and hf token: {ex}")
  logging.error(traceback.print_exc())
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
  




