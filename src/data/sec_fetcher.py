

'''how to use python dotenv library:
import os
from dotenv import load_dotenv

load_dotenv() # this loads the variables from .env into our venv

then access them using os.getenv()
api_key = os.getenv("NEWSAPI_KEY")

THEN:
print(f"connecting to: {api_key})
'''