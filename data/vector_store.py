import logging
from typing import List, Dict
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import config

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, db_path=config.CHROMA_DB_PATH):
        self.client = PersistentClient(path=db_path)
        self.embedding_fn = self._initialize_embedding_function()
        
    def _initialize_embedding_function(self):
        """Initialize the embedding function with proper error handling"""
        try:
            return OpenAIEmbeddingFunction(
                api_key=config.OPENAI_API_KEY, 
                model_name=config.EMBEDDING_MODEL
            )
        except Exception as e:
            logger.error(f"Error initializing embedding function: {e}")
            logger.info("Falling back to text-embedding-ada-002")
            return OpenAIEmbeddingFunction(
                api_key=config.OPENAI_API_KEY, 
                model_name="text-embedding-ada-002"
            )
    
    def get_or_create_collection(self, name=config.COLLECTION_NAME, force_recreate=False):
        """Get a collection or recreate it if dimension mismatch occurs"""
        try:
            # First try to get the existing collection
            if force_recreate:
                try:
                    self.client.delete_collection(name)
                    logger.info(f"Deleted existing collection: {name}")
                except:
                    pass  # Collection might not exist yet
            
            collection = self.client.get_or_create_collection(
                name=name,
                embedding_function=self.embedding_fn
            )
            logger.info(f"Successfully accessed collection: {name}")
            return collection
        except Exception as e:
            if "dimension" in str(e).lower():
                # If there's a dimension mismatch, try recreating the collection
                logger.warning(f"Dimension mismatch detected: {e}")
                logger.info(f"Recreating collection: {name}")
                try:
                    self.client.delete_collection(name)
                    collection = self.client.create_collection(
                        name=name,
                        embedding_function=self.embedding_fn
                    )
                    logger.info(f"Successfully recreated collection: {name}")
                    return collection
                except Exception as inner_e:
                    logger.error(f"Failed to recreate collection: {inner_e}")
                    raise
            else:
                logger.error(f"Unexpected error with collection: {e}")
                raise
    
    def initialize_from_corpus(self, corpus: List[Dict]) -> None:
        """Initialize and populate the vector store with corpus data"""
        try:
            # Get or recreate collection to handle dimension issues
            collection = self.get_or_create_collection()
            
            # Check if collection already has documents
            if collection.count() > 0:
                logger.info(f"Collection already contains {collection.count()} documents")
                return
            
            # Prepare data for insertion
            documents = []
            metadatas = []
            ids = []
            
            for i, q in enumerate(corpus):
                if "question" not in q:
                    continue
                    
                documents.append(q["question"])
                
                metadata = {"type": q.get("question_type", "unknown")}
                if "explanation" in q:
                    metadata["explanation"] = q["explanation"]
                metadatas.append(metadata)
                
                question_id = str(q.get("question_number", i))
                ids.append(question_id)
            
            if not documents:
                logger.warning("No valid documents to add to vector store")
                return
                
            # Add documents in batches to avoid issues with large corpora
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                end = min(i + batch_size, len(documents))
                collection.add(
                    documents=documents[i:end],
                    metadatas=metadatas[i:end],
                    ids=ids[i:end]
                )
                logger.info(f"Added batch of {end-i} documents to vector store")
            
            logger.info(f"Total added {len(documents)} documents to vector store")
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise
    
    def query_collection(self, query_text, n_results=5):
        """Query the collection with proper error handling"""
        try:
            collection = self.client.get_collection(
                name=config.COLLECTION_NAME,
                embedding_function=self.embedding_fn
            )
            
            results = collection.query(
                query_texts=[query_text],
                n_results=n_results,
                include=["metadatas", "documents"]
            )
            
            return results
        except Exception as e:
            logger.error(f"Error querying collection: {e}")
            # Return empty results structure
            return {"ids": [[]], "documents": [[]], "metadatas": [[]]}

