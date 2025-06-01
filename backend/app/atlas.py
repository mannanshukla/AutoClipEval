from pymongo import MongoClient
from dotenv import load_dotenv
import os
from pymongo.server_api import ServerApi

load_dotenv()

ATLAS_URI = os.getenv("ATLAS_URI")
PEM_PATH = os.getenv("PEM_PATH")

class AtlasClient:

    def __init__(self, altas_uri, dbname):
        self.mongodb_client = MongoClient(altas_uri,
                                         tls=True,
                                         tlsCertificateKeyFile='X509-cert-5960427672090205870.pem',
                                         server_api=ServerApi('1'))
        self.database = self.mongodb_client[dbname]

    # A quick way to test if we can connect to Atlas instance
    def ping(self):
        self.mongodb_client.admin.command("ping")

    # Get the MongoDB Atlas collection to connect to
    def get_collection(self, collection_name):
        collection = self.database[collection_name]
        return collection

    # Query a MongoDB collection
    def find(self, collection_name, filter={}, limit=0):
        collection = self.database[collection_name]
        items = list(collection.find(filter=filter, limit=limit))
        return items

if __name__ == "__main__":
    DB_NAME = "sample_mflix"
    COLLECTION_NAME = "embedded_movies"

    atlas_client = AtlasClient(ATLAS_URI, DB_NAME)
    atlas_client.ping()
    print("Connected to Atlas instance! We are good to go!")