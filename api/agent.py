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

from llama_index.core.tools import FunctionTool, ToolMetadata
import math
from typing import Optional, List, Dict
import numpy_financial as npf

class RentalHelper:
    """
    A class that encapsulates rental calculation logic, mirroring the VB.NET RentalHelper.
    """
    def __init__(self):
        self.CostNoVAT = 0.0
        self.CostWithVAT = 0.0
        self.VATRate = 0.0
        self.IntRate = 0.0
        self.RVRate = 0.0
        self.InsuranceRate = 0.0
        self.Tenure = 0
        self.TenureType = 1  # Assuming 1 for months, can be adjusted based on ddlTenureType values
        self.PaymentsPerYear = 12 # Default to monthly payments
        self.ArrearsAdvance = "Arrears" # Default value
        self.term = 0.0
        self.resdValue = 0.0
        self.bankIntRate = 0.0
        self.periodInYears = 0.0
        self.rvInvestment = 0.0
        self.bankCostWithVAT = 0.0
        self.bankCostNoVAT = 0.0

    def GetBankRentalAmount(self):
        """
        Calculates the bank rental amount using the PMT formula.
        """
        if self.bankIntRate == 0 or self.term == 0:
            return 0.0

        pv = -self.bankCostWithVAT  # Present value (negative as it's an outflow)
        r = self.bankIntRate / self.PaymentsPerYear # Rate per period
        nper = self.term * self.PaymentsPerYear # Total number of periods

        if r == 0:
            return pv / nper

        pmt = (r * pv) / (1 - math.pow(1 + r, -nper))
        return pmt

    def GetRentalAmount(self):
        """
        Calculates the rental amount using the PMT formula (similar to bank rental but with potentially different interest rate).
        """
        if self.IntRate == 0 or self.term == 0:
            return 0.0

        pv = -self.CostWithVAT  # Present value
        r = self.IntRate / self.PaymentsPerYear # Rate per period
        nper = self.term * self.PaymentsPerYear # Total number of periods

        if r == 0:
            return pv / nper

        pmt = (r * pv) / (1 - math.pow(1 + r, -nper))
        return pmt





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
            description="Use this tool to search for information within our company's documents.  You can ask questions about our fintech products, and this tool will look through documents like BRDs, user manuals, release notes, and technical guides to find the answer.  It's great for finding specific details about product features, how to use them, and troubleshooting tips.",
        )


        def verify_rentals(
            item_count: Optional[int] = None,
            principal: Optional[float] = None,
            capital: Optional[float] = None,
            vat_rate: Optional[float] = None,
            bank_margin_rate: Optional[float] = None,
            residual_value_rate: Optional[float] = None,
            insurance_rate: Optional[float] = None,
            tenure: Optional[int] = None,
            tenure_type: Optional[str] = None, # e.g., "Months", "Years"
            payment_frequency: Optional[str] = None, # e.g., "Monthly", "Quarterly", "Yearly"
            due_factor: Optional[str] = None, # "Arrears" or "Advance"
            interest_computation_method: Optional[str] = None, # e.g., "Simple", "Compound", "6" for step
            bank_residual_value: Optional[float] = None,
            bank_interest_rate: Optional[float] = None,
            bank_period_in_years: Optional[float] = None,
            maintenance_rental: Optional[float] = None,
            charges_armotization: Optional[float] = None,
            rental_vat_rate_setup: Optional[float] = None
        ) -> Dict:
            """
            Verifies lease rental calculations, specifically designed for use
            when calculating lease rentals within LeasePac.
        
            Args:
                item_count (int, optional): The quantity of items being leased.
                principal (float, optional): The principal amount (without VAT).
                capital (float, optional): The total capital amount (with VAT).
                vat_rate (float, optional): The VAT rate (in percentage).
                bank_margin_rate (float, optional): The bank margin rate (in percentage).
                residual_value_rate (float, optional): The residual value rate (in percentage).
                insurance_rate (float, optional): The insurance rate (in percentage).
                tenure (int, optional): The number of repayment periods.
                tenure_type (str, optional): The type of tenure ('Months' or 'Years').
                payment_frequency (str, optional): The payment frequency ('Monthly', 'Quarterly', or 'Yearly').
                due_factor (str, optional): Indicates if payments are due in 'Arrears' or 'Advance'.
                interest_computation_method (str, optional): The interest computation method (e.g., 'Simple', 'Compound', '6').
                bank_residual_value (float, optional): The bank's residual value.
                bank_interest_rate (float, optional): The bank's interest rate (in percentage).
                bank_period_in_years (float, optional): The bank's period in years.
                maintenance_rental (float, optional): The monthly maintenance rental cost.
                charges_armotization (float, optional): The total charges for amortization.
                rental_vat_rate_setup (float, optional): The VAT rate for the rental calculation (in percentage).
        
            Returns:
                Dict: A dictionary containing the computed rental values, or a message
                      indicating missing parameters.
            """
            if any(arg is None for arg in [item_count, principal, capital, vat_rate,
                                           bank_margin_rate, residual_value_rate,
                                           insurance_rate, tenure, tenure_type,
                                           payment_frequency, due_factor,
                                           interest_computation_method,
                                           bank_residual_value, bank_interest_rate,
                                           bank_period_in_years, maintenance_rental,
                                           charges_armotization, rental_vat_rate_setup]):
                return {
                    "error": "Missing parameters. Please provide the following:",
                    "parameters_needed": [
                        "item_count (integer)",
                        "principal (float)",
                        "capital (float)",
                        "vat_rate (float)",
                        "bank_margin_rate (float)",
                        "residual_value_rate (float)",
                        "insurance_rate (float)",
                        "tenure (integer)",
                        "tenure_type (string: 'Months' or 'Years')",
                        "payment_frequency (string: 'Monthly', 'Quarterly', or 'Yearly')",
                        "due_factor (string: 'Arrears' or 'Advance')",
                        "interest_computation_method (string: e.g., 'Simple', 'Compound', '6')",
                        "bank_residual_value (float)",
                        "bank_interest_rate (float)",
                        "bank_period_in_years (float)",
                        "maintenance_rental (float)",
                        "charges_armotization (float)",
                        "rental_vat_rate_setup (float)"
                    ]
                }
        
            quantity = item_count
            vat_setup = rental_vat_rate_setup / 100.0
        
            rt = RentalHelper()
            rt.CostNoVAT = principal * quantity
            rt.CostWithVAT = capital * quantity
            rt.VATRate = vat_rate / 100.0
            rt.IntRate = bank_margin_rate / 100.0
            rt.RVRate = residual_value_rate / 100.0
            rt.InsuranceRate = insurance_rate / 100.0
            rt.Tenure = tenure
        
            if tenure_type.lower() == "years":
                rt.TenureType = 12
            elif tenure_type.lower() == "months":
                rt.TenureType = 1
            else:
                raise ValueError(f"Invalid tenure type: {tenure_type}")
        
            if payment_frequency.lower() == "monthly":
                rt.PaymentsPerYear = 12
                month_period = 1
            elif payment_frequency.lower() == "quarterly":
                rt.PaymentsPerYear = 4
                month_period = 3
            elif payment_frequency.lower() == "yearly":
                rt.PaymentsPerYear = 1
                month_period = 12
            else:
                raise ValueError(f"Invalid payment frequency: {payment_frequency}")
        
            rt.ArrearsAdvance = due_factor
        
            if interest_computation_method == "6":
                rt.term = tenure # Assuming for step, tenure directly represents the number of periods
            else:
                rt.term = (rt.Tenure * rt.TenureType) / month_period
        
            # Bank Details
            rt.resdValue = bank_residual_value
            rt.bankIntRate = bank_interest_rate / 100.0
            rt.periodInYears = bank_period_in_years
        
            rt.rvInvestment = rt.resdValue / (math.pow((1 + rt.bankIntRate), rt.periodInYears)) if rt.bankIntRate != 0 else rt.resdValue
            rt.bankCostWithVAT = (rt.CostWithVAT) - rt.rvInvestment
            rt.bankCostNoVAT = rt.bankCostWithVAT / (1 + rt.VATRate) if (1 + rt.VATRate) != 0 else rt.bankCostWithVAT
            bank_rental = rt.GetBankRentalAmount()
        
            rental = rt.GetRentalAmount()
            rental_monthly = rental / month_period
            insurance = rt.CostWithVAT * rt.InsuranceRate
            insurance_monthly = insurance / 12
            insurance_total = insurance_monthly * month_period
        
            charges_total = charges_armotization + (maintenance_rental * quantity)
            monthly_cost = rental_monthly + charges_total + insurance_monthly
            wet_lease_rental = monthly_cost * month_period
            wet_lease_rental_vat = wet_lease_rental * ((100 + rental_vat_rate_setup) / 100)
            monthly_cost_vat = monthly_cost * ((100 + rental_vat_rate_setup) / 100)
            effective_vat_rate = vat_rate / 100.0
        
            if interest_computation_method == "6":
                rental_margin = monthly_cost - bank_rental
            else:
                rental_margin = wet_lease_rental - bank_rental
        
            all_costs = rt.CostNoVAT
            all_costs_vat = all_costs * (1 + effective_vat_rate) if effective_vat_rate != 0 else all_costs
            service_fees = (maintenance_rental * month_period) * quantity
        
            return {
                "bank_rental": bank_rental,
                "rv_investment": rt.rvInvestment,
                "bank_cost_vat": rt.bankCostWithVAT,
                "bank_cost_less_vat": rt.bankCostNoVAT,
                "rental_amount": rental,
                "rental_per_month": rental_monthly,
                "insurance_amount": insurance_total,
                "insurance_per_month": insurance_monthly,
                "monthly_cost": monthly_cost,
                "monthly_cost_vat": monthly_cost_vat,
                "wet_lease_rental_cost": wet_lease_rental,
                "wet_lease_rental_cost_vat": wet_lease_rental_vat,
                "all_costs": all_costs,
                "all_costs_vat": all_costs_vat,
                "rental_margin": rental_margin,
                "service_fees": service_fees,
                "term": rt.term
            }
        
        
        verify_leasepac_rentals_tool = FunctionTool.from_defaults(
            name="verify_leasepac_rentals",
            fn=verify_rentals,
            description="""
            Use this tool to verify lease rental calculations, specifically when working
            with LeasePac. This tool requires various financial inputs such as item count,
            principal, interest rates, tenure, and other charges to compute and return
            key rental figures for verification against LeasePac outputs. If no parameters
            are provided, the tool will indicate the necessary parameters.
            The parameters are as follows;
            param_descriptions={
                "item_count": "The quantity of items being leased (integer).",
                "principal": "The principal amount (without VAT) as a float.",
                "capital": "The total capital amount (with VAT) as a float.",
                "vat_rate": "The VAT rate as a percentage (float).",
                "bank_margin_rate": "The bank margin rate as a percentage (float).",
                "residual_value_rate": "The residual value rate as a percentage (float).",
                "insurance_rate": "The insurance rate as a percentage (float).",
                "tenure": "The number of repayment periods (integer).",
                "tenure_type": "The type of tenure ('Months' or 'Years' as a string).",
                "payment_frequency": "The payment frequency ('Monthly', 'Quarterly', or 'Yearly' as a string).",
                "due_factor": "Indicates if payments are due in 'Arrears' or 'Advance' (string).",
                "interest_computation_method": "The interest computation method (e.g., 'Simple', 'Compound', '6' as a string).",
                "bank_residual_value": "The bank's residual value as a float.",
                "bank_interest_rate": "The bank's interest rate as a percentage (float).",
                "bank_period_in_years": "The bank's period in years as a float.",
                "maintenance_rental": "The monthly maintenance rental cost as a float.",
                "charges_armotization": "The total charges for amortization as a float.",
                "rental_vat_rate_setup": "The VAT rate for the rental calculation as a percentage (float)."
            }
            """,
        
        )

        def calculate_lease_rentals(
            unit_cost_asset: float,
            number_of_assets: int,
            leasing_rate: float,
            tenor_months: int,
            residual_value_rate: float,
            leasing_margin: float,
            arrears_advance: int = 0  # 0 for arrears, 1 for advance
        ) -> Dict:
            """
            Calculates both the client and bank lease rentals and returns a summary
            including the inputs for both calculations.
        
            Args:
                unit_cost_asset (float): The unit cost of the asset (VAT inclusive).
                number_of_assets (int): The number of assets.
                leasing_rate (float): The annual leasing rate (as a decimal).
                tenor_months (int): The tenor of the lease in months.
                residual_value_rate (float): The residual value rate (as a decimal).
                leasing_margin (float): The annual leasing margin (as a decimal).
                arrears_advance (int): 0 for arrears, 1 for advance. Defaults to 0.
        
            Returns:
                Dict: A dictionary containing the client rental summary and the bank rental summary,
                      each including their respective inputs and calculated rental.
            """
            total_cost_vat_inclusive_client = round(unit_cost_asset * number_of_assets, 2)
            total_cost_less_vat_client = round(total_cost_vat_inclusive_client / 1.16, 2)
        
            client_rate_per_period = (leasing_rate/100) / 12
            client_number_of_periods = tenor_months
            client_present_value = total_cost_less_vat_client
            client_future_value = -(round(residual_value_rate/100, 2) * total_cost_vat_inclusive_client)
        
            # print(f"client_rate_per_period: {client_rate_per_period}")
            # print(total_cost_less_vat_client)
            # print(f"client_number_of_periods: {client_number_of_periods}")
            # print(f"client_present_value: {client_present_value}")
            # print(f"client_future_value: {client_future_value}")
            # print(f"arrears_advance: {arrears_advance}")
            # print(f"leasing_margin: {leasing_margin}")
            # print(f"total_cost_vat_inclusive_client: {total_cost_vat_inclusive_client}")
        
            client_rental = -npf.pmt(client_rate_per_period, client_number_of_periods, client_present_value, client_future_value, when=('begin' if arrears_advance == 1 else 'end'))
            client_rental_rounded = round(client_rental, 2)
        
            client_rental_summary = {
                "inputs": {
                    "unit_cost_asset": unit_cost_asset,
                    "number_of_assets": number_of_assets,
                    "leasing_rate": leasing_rate,
                    "tenor_months": tenor_months,
                    "residual_value_rate": residual_value_rate,
                    "arrears_advance": arrears_advance,
                },
                "calculated_rental": client_rental_rounded,
            }
        
            effective_bank_rate = leasing_rate - leasing_margin
            bank_periods_in_year = 12
            discount_factor_bank = pow((1 + effective_bank_rate/100),(tenor_months / bank_periods_in_year))
            npv_bank = ((residual_value_rate/100) * total_cost_vat_inclusive_client) / discount_factor_bank
            bank_cost_with_vat = round(total_cost_vat_inclusive_client - npv_bank, 2)
            bank_cost_less_vat = round(bank_cost_with_vat / 1.16, 2)
        
        
            # print(effective_bank_rate)
            # print(bank_periods_in_year)
            # print(discount_factor_bank)
            # print((residual_value_rate/100) * total_cost_vat_inclusive_client)
            # print(pow((1 + effective_bank_rate/100),(tenor_months / bank_periods_in_year)))
            # print(npv_bank)
            # print(bank_cost_with_vat)
            # print(bank_cost_less_vat)
        
            bank_rental = -npf.pmt((effective_bank_rate/100) / bank_periods_in_year, tenor_months, bank_cost_less_vat, 0, when=('begin' if arrears_advance == 1 else 'end'))
            bank_rental_rounded = round(bank_rental, 2)
        
            bank_rental_summary = {
                "inputs": {
                    "bank_rate": effective_bank_rate,
                    "leasing_margin": leasing_margin,
                    "total_cost_asset_vat": total_cost_vat_inclusive_client,
                    "tenor_months": tenor_months,
                    "residual_value": round(residual_value_rate * total_cost_vat_inclusive_client, 2),
                    "arrears_advance": arrears_advance,
                    "npv_bank": npv_bank,
                    "bank_cost_with_vat": bank_cost_with_vat,
                    "bank_cost_less_vat": bank_cost_less_vat,
                },
                "calculated_rental": bank_rental_rounded,
            }
        
            return {
                "client_rental_summary": client_rental_summary,
                "bank_rental_summary": bank_rental_summary,
            }
        
        

        calculate_lease_rentals_tool = FunctionTool.from_defaults(
            name="calculate_client_and_bank_rentals",
            fn=calculate_lease_rentals,
            description="""
            Calculates lease payments for both the customer and the bank.  Tell me about the lease, and I'll work out the monthly payments.
            You'll need to provide these details:
        
            - "unit_cost_asset": How much does one item being leased cost? (including VAT)
            - "number_of_assets": How many items are being leased?
            - "leasing_rate": What's the yearly interest rate for the lease? (e.g., 0.10 for 10%)
            - "tenor_months": How many months will the lease last?
            - "residual_value_rate": What percentage of the item's value will it be worth at the end of the lease? (e.g., 0.15 for 15%)
            - "leasing_margin": What's the bank's profit margin on the lease?
            - "arrears_advance": When are payments made?  Use 0 for the end of the month, or 1 for the beginning.
        
            I'll calculate the monthly payment for both the customer and the bank.
            """,
        )




                


        

        llm = Settings.llm  # Make sure Settings.llm is correctly initialized
        agent_worker = FunctionCallingAgentWorker.from_tools(
                tools=[vector_query_tool, wikipedia_tool, verify_leasepac_rentals_tool, calculate_lease_rentals_tool],
                llm=llm,
                  system_prompt="""
                    You are a highly specialized AI assistant designed to answer user queries related to fintech and lease rentals. You have access to the following tools:
                    
                    1.  'document_retrieval':  This is your go-to tool for questions about our company's specific fintech products.  It searches our internal documents like BRDs, user manuals, and tech guides to find details on product features, requirements, and technical information.  Use this first for questions about how our products work.
                    
                    2.  'wikipedia_search':  Use this tool to search Wikipedia for general knowledge.  It's helpful for definitions, background information, and broader industry context on fintech topics when the answer isn't in our company documents and is NOT about the specific functionality or setup of our company's products.
                    
                    3.  'verify_leasepac_rentals':  This tool checks lease rental calculations, especially for LeasePac.  Use it to confirm the accuracy of a lease payment calculation when you have the specific details used in that calculation.
                    
                    4.  'calculate_client_and_bank_rentals': This tool calculates lease payments for both the customer and the bank, given the lease terms.  Use this when the user wants to know the actual payment amounts, and provides the cost of the asset, number of assets, interest rate, lease length, residual value, bank margin, and payment timing.
                    
                    When a user asks a question, follow this process:
                    
                    1.  If the user asks for a lease payment calculation and provides all the necessary details (asset cost, number of assets, interest rate, lease length, residual value, bank margin, payment timing), use 'calculate_client_and_bank_rentals'.  Provide the calculated amounts.
                    
                    2.  If the user asks to verify a lease calculation, especially for LeasePac, and provides the calculation details, use 'verify_leasepac_rentals'.  Confirm the accuracy.
                    
                    3.  For questions about our company's fintech products, their features, how they work, or their setup (e.g., system requirements, database), start with 'document_retrieval'.
                    
                    4.  For general fintech information, definitions, or industry context that is NOT about our company's products, use 'wikipedia_search'.
                    
                    5.  If none of the tools provide relevant information to answer the user's query, respond with: "I could not find the answer to your question in the available resources."
                    
                    Provide accurate and concise answers based on the information from the tools.  Always mention which tool you used.
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




