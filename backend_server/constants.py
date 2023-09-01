import os


try:
    from dotenv import load_dotenv
    load_dotenv("../.env")
except:
    pass

DATABASE_NAME = os.environ["DATABASE_NAME"]
DATABASE_PASSWORD = os.environ["DATABASE_PASSWORD"]
DATABASE_USER = os.environ["DATABASE_USER"]
DATABASE_URL = os.environ["DATABASE_URL"]

REDIS_URL = os.environ["REDIS_URL"]
REDIS_PORT = os.environ["REDIS_PORT"]
