import os
import inspect
import traceback
from huggingface_hub import HfApi, create_repo, login,hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError

class HostingInHuggingFace:
  def __init__(self,base_path,hf_token=None):
    self.base_path = base_path
    self.hf_token = hf_token
    self.repo_id = 'jpkarthikeyan/Tourism-Prediction-Model-Space'

  def CreatingSpaceInHF(self):
    print(f"Function Name {inspect.currentframe().f_code.co_name}")
    api = HfApi()
    try:
      print(f"Checking for {self.repo_id} is correct or not")
      api.repo_info(repo_id = self.repo_id,
                    repo_type='space',
                    token = self.hf_token)
      print(f"Space {self.repo_id} already exists")
    except RepositoryNotFoundError:
      create_repo(repo_id=self.repo_id,
                  repo_type='space',
                  space_sdk='docker',
                  private=False,
                  token=self.hf_token)
      print(f"Space created in {self.repo_id}")
    except Exception as ex:
      print(f"Exception in creating space {ex}")
      traceback.print_exc()
    finally:
      print('-'*50)


  def UploadDeploymentFile(self):
    print(f"Function Name {inspect.currentframe().f_code.co_name}")
    try:
      api = HfApi(token=self.hf_token)
      directory_to_upload = os.path.join(self.base_path,'Deployment')
      print(f"Directory to upload {directory_to_upload} into HF Space {self.repo_id}")
      api.upload_folder(repo_id=self.repo_id, folder_path=directory_to_upload,
                        repo_type='space')
      print(f"Successfully upload {directory_to_upload} into {self.repo_id}")

     
      return True
    except Exception as ex:
      print(f"Exception occured {ex}")
      print(traceback.print_exc())
      return False
    finally:
      print('-'*50)

  def ToRunPipeline(self):
    try:
      self.CreatingSpaceInHF()
      if self.UploadDeploymentFile():
        print('Deployment pipeline completed')
        return True
      else:
        print('Deployment pipeline failed')
        return False
    except Exception as ex:
      print(f"Exception occured {ex}")
      print(traceback.print_exc())
      return False
    finally:
      print('-'*50)
