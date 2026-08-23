import os 

from dotenv import load_dotenv

load_dotenv()



def _getenv_none_if_empty(key:str , default=None):
    value = os.getenv(key)

    if value is None or value.strip() == "":
        return default
    return value 





UPLOAD_FOLDER = _getenv_none_if_empty("UPLOAD_FOLDER", "data")
INDEX_STORE = _getenv_none_if_empty("INDEX_STORE", "index_store")




AZURE_OPENAI_EMBEDDING_MODEL= _getenv_none_if_empty("AZURE_OPENAI_EMBEDDING_MODEL")
AZURE_OPENAI_ENDPOINT = _getenv_none_if_empty("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = _getenv_none_if_empty("AZURE_OPENAI_KEY")
AZURE_OPENAI_API_VERSION = _getenv_none_if_empty("AZURE_OPENAI_API_VERSION")


AZURE_OPENAI_GPT_DEPLOYMENT = _getenv_none_if_empty("AZURE_OPENAI_GPT_DEPLOYMENT")




