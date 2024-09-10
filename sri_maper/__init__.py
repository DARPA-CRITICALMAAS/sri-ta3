from dotenv import load_dotenv
import os
# load .env file to environment
load_dotenv()
__version__ = os.getenv("SYSTEM_VERSION")
from . import *