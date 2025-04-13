from llama_index.core import Settings
from llama_index.llms.gemini import Gemini

# imports
from llama_index.embeddings.gemini import GeminiEmbedding
from django.conf import  settings
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from pinecone import Pinecone, ServerlessSpec
from llama_index.core import (
    SummaryIndex)



from typing import List, Optional
from llama_index.core import (
    VectorStoreIndex,
    SummaryIndex,
    SimpleDirectoryReader,
    StorageContext,
)
from llama_index.core import Settings

from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core.tools import FunctionTool
from llama_index.core.vector_stores import MetadataFilters, FilterCondition

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools.tool_spec.load_and_search import (
    LoadAndSearchToolSpec,
)
from llama_index.tools.wikipedia import WikipediaToolSpec
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner
import os


# define my env variables



"""
SETUP PINECONE
"""







class AgentExecuter:


    def __init__(self):


        """

        Here we are defining our llm model and our text embedding model 
        """
        os.environ["GOOGLE_API_KEY"]="AIzaSyAMs1Y4xmSrAadzADmZha-baQxTJg2Tq5Q"
  
        model_name = "models/text-embedding-004"

        Settings.llm = Gemini(
            model="models/gemini-1.5-flash",
            api_key="AIzaSyAMs1Y4xmSrAadzADmZha-baQxTJg2Tq5Q",  # uses GOOGLE_API_KEY env var by default
        )
        Settings.embed_model = GeminiEmbedding(
            model_name=model_name,
        )

    def Index_store(self):
        # Create Pinecone Vector Store
        pc = Pinecone(api_key="pcsk_23XXfP_GHWmfdm7WGfPMmTRZACC917oLVk8LxueaGXHp27p6hHaE9rzz9RMog6i8Z6jy7S")

        # pc.create_index(
        #     name="quickstart",
        #     dimension=768,
        #     metric="dotproduct",
        #     spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        # )

        pinecone_index = pc.Index("pinecone-chatbot")

        vector_store = PineconeVectorStore(
            pinecone_index=pinecone_index,
        )
        # will be used after loading data
        index = VectorStoreIndex.from_vector_store(vector_store)

        return index, vector_store




    def add_data_to_vector_store(self,
            file_path: str,
            name: str,
    ) -> str:
        """Get vector query and summary query tools from a document."""
        x, vector_store = self.Index_store()
        # Load documents and build index
        documents = SimpleDirectoryReader(
            input_files=[file_path]
        ).load_data()
        splitter = SentenceSplitter(chunk_size=1024)
        nodes = splitter.get_nodes_from_documents(documents)

        embed_model = Settings.embed_model

        for node in nodes:
            node_embedding = embed_model.get_text_embedding(
                node.get_content(metadata_mode="all")
            )
            node.embedding = node_embedding

        vector_store.add(
            nodes
        )




    def agent(self):

        # tools
        wiki_spec = WikipediaToolSpec()
        # Get the search wikipedia tool (assuming it's the second one, verify this)
        wikipedia_tool = wiki_spec.to_tool_list()[1]
        wikipedia_tool.description = (
            "Use this tool to search Wikipedia for general knowledge and information "
            "that might be relevant to the user's fintech-related query but is not found "
            "in the internal company documents. Use it for definitions, background information, "
            "or broader industry context."
        )

#         second tool
        def create_vector_query(
                query: str,
                # page_numbers: Optional[List[str]] = None,
        ) -> str:
            """Retrieve answers from fintech-related documents, including release notes,
            business requirement documents (BRDs), user manuals, and technical guides.

            Use this function to perform a vector search across all available documents,
            unless specific pages are provided for filtering.

            Args:
                query (str): The search query to retrieve relevant information.
                page_numbers (Optional[List[str]]): A list of page numbers to filter results.
                    Leave as None to search across all documents.

            Returns:
                str: The most relevant response based on the query.
            """

            # page_numbers = page_numbers or []
            # metadata_dicts = [
            #     {"key": "page_label", "value": p} for p in page_numbers
            # ]
            # will be used after loading data
            index , y= self.Index_store()

            query_engine = index.as_query_engine(
                similarity_top_k=2,
                # filters=MetadataFilters.from_dicts(
                #     metadata_dicts,
                #     condition=FilterCondition.OR
                # )
            )

            response = query_engine.query(query)
            
            return {
                "response":str(response)
            }

        vector_query_tool = FunctionTool.from_defaults(
            name="document_retrieval",
            fn=create_vector_query,
            description="Use this tool to find specific information within the company's fintech BRDs, user manuals, and technical guides by searching their content based on keywords and concepts. This is the primary tool for answering questions about internal product details.",
        )



        llm = Settings.llm

        agent_worker = FunctionCallingAgentWorker.from_tools(
            tools=[vector_query_tool, wikipedia_tool],
            llm=llm,
            system_prompt="""
        You are a highly specialized AI assistant designed to answer user queries related to fintech. You have access to two primary tools:

        1.  'document_retrieval': This tool should be your **first point of contact** for questions about the company's specific fintech Business Requirement Documents (BRDs), user manuals, and technical manuals. Use it to find details about product features, requirements, and internal technical information.

        2.  'wikipedia_search': This tool should be used for retrieving **general knowledge and information** from Wikipedia. Utilize it when the user's query seems to require broader context, definitions, or information that is likely not contained within the internal company documents.

        **Crucially, you MUST always use the available tools to find relevant information.** Do not attempt to answer questions based on prior knowledge or make any assumptions.

        When a user asks a question, follow this process:

        1.  **First, consider if the question is likely to be answered by the company's internal fintech documents.** If it is about specific product features, requirements, or technical details of company products, **always use the 'document_retrieval' tool first.**

        2.  **If the 'document_retrieval' tool does not return relevant information, OR if the question seems to be about general fintech concepts, industry definitions, or broader background information, then use the 'wikipedia_search' tool.**

        3.  If neither tool provides relevant information to answer the user's query, respond with the following user-friendly message: "I could not find the answer to your question in the available resources."

        Focus on providing accurate and concise answers based solely on the data retrieved by the tools. Clearly indicate which tool you used to obtain the information in your response if necessary for clarity.
        """,
            verbose=True
        )

        agent = AgentRunner(agent_worker)

        try:
            return 200,  agent

        except Exception as e:
            return 400, e




    def query(self, query):
        status, agent= self.agent()

        try:
            if status == 200:
                
                return  200, agent.chat(query).response
            else:
                return 400, "Agent is not available"
        except Exception as e:
            return 400, e




