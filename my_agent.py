
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from langchain_community.utilities.sql_database import SQLDatabase
import os
import re
import logging
from typing import List, Dict, Any, Optional
from functools import lru_cache
from collections import defaultdict
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
# from langchain.agents import create_agent
# from langchain.agents import initialize_agent, Tool  # instead of AgentExecutor/ZeroShotAgent
# from langchain.agents.agent_executor import AgentExecutor
from langchain_openai import ChatOpenAI
from fuzzywuzzy import fuzz
from sqlalchemy import text
from langchain.chains import LLMChain
from urllib.parse import quote_plus
import pyodbc
from langchain_core.tools import Tool
from connection import DatabaseManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
SIMILARITY_THRESHOLD = 75
MAX_CONTEXT_LENGTH = 4000
db=DatabaseManager.get_db()

# class Config:
#     """Configuration management"""
#     def __init__(self):
#         self.openai_api_key = self._get_required_env("API_KEY_OPENAI")
#         self.active_company_id = int(os.getenv("ACTIVE_COMPANY_ID", 74))
        
#     @staticmethod
#     def _get_required_env(key: str) -> str:
#         value = os.getenv(key)
#         if not value:
#             raise ValueError(f"Required environment variable {key} not found")
#         return value

import os
import streamlit as st
from typing import Optional

class Config:
    """Configuration management with Streamlit Cloud support"""
    def __init__(self):
        # Try multiple sources for API key
        self.openai_api_key = self._get_api_key()
        self.active_company_id = int(self._get_env("ACTIVE_COMPANY_ID", "74"))
        
    @staticmethod
    def _get_api_key() -> str:
        """Get API key from multiple possible sources"""
        # 1. Try Streamlit Secrets (for deployment)
        try:
            import streamlit as st
            if hasattr(st, 'secrets'):
                # Try different possible key names in secrets
                api_key = (
                    st.secrets.get("API_KEY_OPENAI") or
                    st.secrets.get("OPENAI_API_KEY") or
                    st.secrets.get("openai_api_key")
                )
                if api_key:
                    return api_key
        except:
            pass
        
        # 2. Try environment variables (for local development)
        api_key = (
            os.getenv("API_KEY_OPENAI") or
            os.getenv("OPENAI_API_KEY") or
            os.getenv("openai_api_key")
        )
        
        if api_key:
            return api_key
        
        # 3. If nothing found, raise error
        raise ValueError(
            "OpenAI API key not found! "
            "Please add it to Streamlit Secrets or environment variables.\n"
            "For Streamlit Cloud: Go to Settings → Secrets and add:\n"
            "API_KEY_OPENAI = 'your-key-here'\n"
            "For local development: Set environment variable API_KEY_OPENAI"
        )
    
    @staticmethod
    def _get_env(key: str, default: Optional[str] = None) -> str:
        """Get environment variable from multiple sources"""
        # Try Streamlit Secrets first
        try:
            import streamlit as st
            if hasattr(st, 'secrets'):
                value = st.secrets.get(key)
                if value:
                    return value
        except:
            pass
        
        # Try environment variables
        value = os.getenv(key)
        if value:
            return value
        
        # Return default or raise error
        if default is not None:
            return default
        raise ValueError(f"Required variable {key} not found")
class ConversationContext:
    """
    Intelligent context tracking
    """
    
    def __init__(self):
        self.current_franchises: List[str] = []
        self.current_time_frame: str = "all"
        self.last_query_type: Optional[str] = None
        self.entity_mentions: Dict[str, int] = defaultdict(int)
        
    def update_context(self, query: str, results: Any = None, query_type: str = None):
        """Update conversation context"""
        # Extract franchises
        franchises = self._extract_franchises(query)
        if franchises:
            self.current_franchises = franchises[:3]  # Keep last 3
        
        # Extract time frame
        time_frame = self._extract_time_frame(query)
        if time_frame:
            self.current_time_frame = time_frame
        
        # Store query type
        if query_type:
            self.last_query_type = query_type
        else:
            self.last_query_type = self._infer_query_type(query)
        
        logger.info(f"Context updated - Franchises: {self.current_franchises}, "
                f"Time: {self.current_time_frame}, Type: {self.last_query_type}")
    
    def resolve_references(self, query: str) -> str:
        """Resolve pronouns and references"""
        original_query = query
        query_lower = query.lower()
        
        # Resolve "it" - refers to single entity
        if re.search(r'\bit\b', query_lower) and len(self.current_franchises) >= 1:
            query = re.sub(
                r'\bit\b', 
                self.current_franchises[0], 
                query, 
                flags=re.IGNORECASE
            )
        
        # Resolve "them/those/these" - refers to multiple entities
        them_pattern = r'\b(them|those|these)\b'
        if re.search(them_pattern, query_lower) and len(self.current_franchises) > 0:
            franchises_str = ', '.join(self.current_franchises)
            query = re.sub(
                them_pattern,
                franchises_str,
                query,
                flags=re.IGNORECASE
            )
        

        if not self._has_time_reference(query) and self.current_time_frame != "all":
            query += f" for {self.current_time_frame}"
        
        if query != original_query:
            logger.info(f"Query resolved: '{original_query}' → '{query}'")
        
        return query
    
    def _extract_franchises(self, query: str) -> List[str]:
        """Extract franchise names from query"""
        franchises = []

        quoted = re.findall(r'["\']([^"\']+)["\']', query)
        franchises.extend(quoted)
        

        json_match = re.search(r'\[([^\]]+)\]', query)
        if json_match:
            try:
                items = json_match.group(1).split(',')
                franchises.extend([item.strip(' "\'') for item in items])
            except:
                pass
        
        # Clean and deduplicate
        seen = set()
        unique_franchises = []
        for f in franchises:
            f_clean = f.strip()
            if f_clean and f_clean not in seen:
                seen.add(f_clean)
                unique_franchises.append(f_clean)
        
        return unique_franchises
    
    def _extract_time_frame(self, query: str) -> Optional[str]:
        """Extract time frame from query"""
        query_lower = query.lower()
        
        time_patterns = {
            'last_month': ['last month', 'past month', 'previous month'],
            'last_quarter': ['last quarter', 'past quarter', 'last 3 months'],
            'last_year': ['last year', 'past year', 'previous year'],
            'all': ['all time', 'overall', 'total']
        }
        
        for time_frame, patterns in time_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                return time_frame
        
        return None
    
    def _has_time_reference(self, query: str) -> bool:
        """Check if query has a time reference"""
        time_words = ['month', 'quarter', 'year', 'week', 'day', 'time', 'period']
        return any(word in query.lower() for word in time_words)
    
    def _infer_query_type(self, query: str) -> str:
        """Infer the type of query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['compare', 'comparison', 'vs', 'versus']):
            return "comparison"
        elif any(word in query_lower for word in ['performance', 'ranking', 'best', 'worst']):
            return "performance"
        elif any(word in query_lower for word in ['list', 'show', 'get']):
            return "list"
        else:
            return "general"
    
    def clear_context(self):
        """Clear all context"""
        self.current_franchises = []
        self.current_time_frame = "all"
        self.last_query_type = None
        logger.info("Context cleared")


class SmartBusinessAssistant:
    def __init__(self, db: SQLDatabase):
        self.config = Config()
        self.db = db
        


        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            # model="gpt-4o",
            api_key=self.config.openai_api_key,
            temperature=0.1,
            max_retries=2
        )

        self._schema_used=False
        self.schema = self.schema_lookup_func()
        # Initialize context
        self.context = ConversationContext()

    
        
        # Get company name
        self.company_name = self._get_company_name()
        
        # Get table names
        try:
            self.tables_names = db.get_usable_table_names()
        except Exception as e:
            logger.warning(f"Could not get table names: {e}")
            self.tables_names = []
        
        # Initialize components
        self.performance_evaluator = self.FranchisePerformanceEvaluator(self.db, self)
        self.comparator = self.FranchiseComparison(self.db, self, self.performance_evaluator)
        
        # Setup tools and agent
        self._setup_tools()
        self._setup_agent()

    # def schema_lookup_func(self, input_str: str = "") -> str:
    #     """Return database schema. Input is ignored but required for tool interface."""
    #     table_info = self.db.get_table_info()
    #     self._schema_used = True
    #     return str(table_info)
    def schema_lookup_func(self,input_str: str = "") -> str:
        """Return concise database schema."""
        full_info = self.db.get_table_info()
        

        import re

        table_matches = re.findall(r'CREATE TABLE \[(.*?)\] \(', full_info)
        

        schema_lines = []
        for table_name in table_matches:
            # Find this table's definition
            table_pattern = rf'CREATE TABLE \[{table_name}\].*?\)\n\n'
            table_match = re.search(table_pattern, full_info, re.DOTALL)
            
            if table_match:
                table_def = table_match.group(0)

                column_lines = []
                for line in table_def.split('\n'):
                    if line.strip() and not line.strip().startswith('(') and not line.strip().startswith(')') and not line.strip().startswith('CREATE'):
                        col_match = re.match(r'\s*\[(.*?)\]', line.strip())
                        if col_match:
                            column_lines.append(col_match.group(1))
                
                if column_lines:
                    schema_lines.append(f"{table_name}: {', '.join(column_lines)}")
        
        return "\n".join(schema_lines)

    
    def _get_company_name(self) -> str:
        """Get company name from database"""
        try:
            query = "SELECT Com_Name FROM tab_company WHERE Com_ID = :company_id"
            rows = self._execute_query(query, {"company_id": self.config.active_company_id})
            if rows and len(rows) > 0:
                return rows[0].get("Com_Name", "Company")
            return "Company"
        except Exception as e:
            logger.error(f"Failed to get company name: {e}")
            return "Company"
    
    def _execute_query(self, query: str, params: Dict = None) -> List[Dict]:
        """Execute SQL query and return results as list of dictionaries"""
        try:
            if params is None:
                params = {}
            
            logger.info(f"Executing query: {query[:100]}... with params: {params}")
            connection = self.db._engine.connect()
            # with self.db._engine.connect().execution_options
            result = connection.execute(text(query), params)
            
            # Get column names
            columns = list(result.keys())
            
            # Convert rows to dictionaries
            rows = []
            for row in result:
                row_dict = {}
                for i, col in enumerate(columns):
                    row_dict[col] = row[i]
                rows.append(row_dict)
            
            connection.close()
            return rows
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def _setup_tools(self):
        """Setup all available tools"""
        
        # SQL Query Tool - SIMPLIFIED
        def sql_query_func(query: str) -> str:
            """Execute SQL queries"""
            alias_map = {}
            for tbl, alias in re.findall(r"(dbo\.\w+)\s+(\w+)", query, re.I):
                alias_map[alias] = tbl

            if re.search(r"\bSELECT\b", query, re.I):
                schema = self.schema
                for match in re.findall(r"(\w+)\.(\w+)", query):
                    table_alias, col = match
                    real_table = alias_map.get(table_alias)

                    if not real_table:
                        continue

                    if col not in schema[real_table]:
                        return f"Invalid column: {real_table}.{col}"

            try:
                # Clean query
                query = query.strip()
                if query.startswith('"') and query.endswith('"'):
                    query = query[1:-1]
                elif query.startswith("'") and query.endswith("'"):
                    query = query[1:-1]
                
                # Remove backticks if present
                query = query.strip().replace('`', '')
                
                logger.info(f"Executing SQL: {query[:100]}...")
                
                # Execute the query
                rows = self._execute_query(query)
                
                if rows:
                    # Format the results simply
                    if len(rows) == 0:
                        return "No results found."
                    
                    result_lines = []
                    for i, row in enumerate(rows[:10]):  # Limit to 10 rows
                        result_lines.append(f"Row {i+1}: {row}")
                    
                    if len(rows) > 10:
                        result_lines.append(f"... and {len(rows) - 10} more rows")
                    
                    return "\n".join(result_lines)
                else:
                    return "Query executed successfully but returned no results."
                    
            except Exception as e:
                return f"SQL Error: {str(e)}"
        
        sql_tool = Tool(
            name="sql_db_query",
            func=sql_query_func,
            description="""
            Execute SQL queries to get raw data from the database.
            Input: A SQL SELECT query WITHOUT surrounding quotes.
            Example: SELECT TOP 1 CustomerName FROM Customer ORDER BY Customer_ID DESC
            """
        )
        
        # Performance Tool - SIMPLIFIED
        def perf_tool_func(input_data: str = "") -> str:
            """Analyze franchise performance"""
            try:
                # Simple parsing
                input_data = input_data.strip().strip('"\'')
                
                if not input_data:
                    franchise_names = []
                elif "," in input_data:
                    franchise_names = [name.strip() for name in input_data.split(",")]
                else:
                    franchise_names = [input_data]
                
                logger.info(f"Performance tool called with: {franchise_names}")
                result = self.performance_evaluator.evaluate_franchise_performance(
                    franchise_names=franchise_names
                )
                
                if "error" in result:
                    return f"Error: {result['error']}"
                
                # Simple formatting
                rows = result.get("performance_data", {}).get("rows", [])
                if not rows:
                    return "No performance data found"
                
                response = ["Performance Analysis:"]
                for row in rows:
                    if isinstance(row, dict):
                        name = row.get("Franchise_Name", "Unknown")
                        revenue = row.get("Total_Revenue", 0)
                        orders = row.get("Total_Orders", 0)
                        response.append(f"{name}: ${revenue:,.2f} revenue, {orders} orders")
                
                return "\n".join(response)
                
            except Exception as e:
                logger.error(f"Performance tool error: {e}")
                return f"Error: {str(e)}"
        
        perf_tool = Tool(
            name="FranchisePerformanceEvaluation",
            func=perf_tool_func,
            description="Analyze franchise performance. Input: Franchise name or comma-separated names."
        )
        
        # Comparison Tool - SIMPLIFIED
        def comp_tool_func(input_data: str = "") -> str:
            """Compare franchises"""
            try:
                # Simple parsing
                input_data = input_data.strip().strip('"\'')
                
                if not input_data:
                    return "Error: Need franchise names to compare"
                
                if "," in input_data:
                    franchise_names = [name.strip() for name in input_data.split(",")]
                else:
                    franchise_names = [input_data]
                
                logger.info(f"Comparison tool called with: {franchise_names}")
                result = self.comparator.compare_franchises(franchise_names)
                
                if "error" in result:
                    return f"Error: {result['error']}"
                
                # Simple formatting
                data = result.get("comparison_data", [])
                if not data:
                    return "No comparison data found"
                
                response = ["Comparison Results:"]
                for item in data:
                    if isinstance(item, dict):
                        name = item.get("Franchise_Name", "Unknown")
                        revenue = item.get("Total_Revenue", 0)
                        orders = item.get("Total_Orders", 0)
                        response.append(f"{name}: ${revenue:,.2f} revenue, {orders} orders")
                
                return "\n".join(response)
                
            except Exception as e:
                logger.error(f"Comparison tool error: {e}")
                return f"Error: {str(e)}"
        
        comp_tool = Tool(
            name="FranchiseComparison",
            func=comp_tool_func,
            description="Compare franchises. Input: Comma-separated franchise names."
        )
        
        # Branch Sites Tool - SIMPLIFIED
        def sites_tool_func(branch_name: str) -> str:
            """Get sites for a branch"""
            try:
                # Clean input
                branch_name = branch_name.strip().strip('"\'')
                
                logger.info(f"Sites tool called for: {branch_name}")
                result = self.get_branch_sites(branch_name)
                
                if isinstance(result, dict) and "error" in result:
                    return f"Error: {result['error']}"
                
                if isinstance(result, list):
                    response = [f"Sites for {branch_name}:"]
                    for site in result[:5]:  # Limit to 5 sites
                        if isinstance(site, dict):
                            site_name = site.get("Sname", "Unknown")
                            orders = site.get("total_orders", 0)
                            revenue = site.get("total_revenue", 0)
                            response.append(f"- {site_name}: {orders} orders, ${revenue:,.2f} revenue")
                    
                    if len(result) > 5:
                        response.append(f"... and {len(result) - 5} more sites")
                    
                    return "\n".join(response) if len(response) > 1 else "No sites found"
                
                return str(result)
                    
            except Exception as e:
                logger.error(f"Sites tool error: {e}")
                return f"Error: {str(e)}"
        
        sites_tool = Tool(
            name="BranchSitesInfo",
            func=sites_tool_func,
            description="Get information about sites within a franchise. Input: Franchise name."
        )
        schema_lookup_tool = Tool(
            name="schema_lookup",
            func=self.schema_lookup_func,
            description="Return table->columns mapping for database schema. Input: table name or empty for all."
        )
        
        
        self.tools = [sql_tool, perf_tool, comp_tool, sites_tool,schema_lookup_tool]
    
    def _setup_agent(self):
        """Setup agent with improved prompt"""

        prefix = f"""
You are a professional SQL assistant for {self.company_name} Answer in professional tone.
You have access to a SQL database containing business data.
You should be very careful to follow the rules below.

STATE RULES (HARD):

You MUST call schema_lookup for table structure if unsure
- If a column does not exist, you MUST stop and say so
- Do Not provide sql data in the final answer unless asked for raw data
VALID ACTIONS:

- sql_db_query
- FranchisePerformanceEvaluation
- FranchiseComparison
- BranchSitesInfo

INVALID BEHAVIOR:
- Guessing table names
- Guessing column names


If you violate rules, the answer is invalid.

FILTERING RULES (MANDATORY)

- The agent must always try to filter results by company or branch when the user requests company/franchise/site-scoped data.
-Don't take look at another company data other than the active company id in the config
- Column name variants to recognize (case-insensitive):
  • Company keys: Com_ID, CompanyId, Company_ID
  • Franchise keys: Fren_ID, FranchiseId, Franchise_ID, FRANCHISE_ID
  • Site keys: Site_ID, SiteId, site_id
- Before generating SQL, INSPECT the table schema (INFORMATION_SCHEMA.COLUMNS or SELECT TOP 1 *) and use the exact column name(s) found. Do NOT guess names.
- Fallback logic for applying a Company filter (Com_ID = :company_id):
  1. If the selected table contains a Company key (e.g., Com_ID), filter directly: `WHERE table.Com_ID = :company_id`.
  2. ELSE if the table contains a Franchise key (Fren_ID / FranchiseId), join to `tab_frenchise` and filter:  
     `FROM table t JOIN tab_frenchise f ON t.Fren_ID = f.Fren_ID WHERE f.Com_ID = :company_id`
  3. ELSE if the table contains only Site_ID, join to `tab_site` then to `tab_frenchise`:  
     `FROM table t JOIN tab_site s ON t.Site_ID = s.Site_ID JOIN tab_frenchise f ON s.Fren_ID = f.Fren_ID WHERE f.Com_ID = :company_id`
  4. If none of the above columns exist, respond: "Cannot apply company filter — no Company/Franchise/Site key found for this table."
- Fallback logic for applying a Franchise filter (Fren_ID or FranchiseId = :franchise_id):
  1. If the table contains a Franchise key, filter directly: `WHERE table.Fren_ID = :franchise_id` (use exact column name).
  2. ELSE if the table has Site_ID, join to `tab_site`: `JOIN tab_site s ON t.Site_ID = s.Site_ID WHERE s.Fren_ID = :franchise_id`.
  3. ELSE respond: "Cannot apply franchise filter — missing Franchise or Site key."
- Fallback logic for applying a Site filter (Site_ID = :site_id):
  1. If the table contains Site_ID, filter directly.
  2. ELSE if the table contains a Franchise key, optionally: `JOIN tab_site s ON s.Fren_ID = t.Fren_ID WHERE s.Site_ID = :site_id`.
  3. ELSE respond: "Cannot apply site filter — missing Site or Franchise key."
- Always prefer direct-column filtering (no join) when the column exists; otherwise use the shortest safe join path shown above.
- Use exact schema-qualified table names (e.g., dbo.tab_frenchise, dbo.tab_site, dbo.Tab_Bill_detail) found from the schema inspection.
- Use parameterized filters (e.g., `:company_id`) — never inline user input.
- Examples (after confirming columns exist):
  • Direct company filter: `SELECT * FROM Tab_Bill WHERE Com_ID = :company_id`
  • Via franchise: `SELECT b.* FROM Tab_Bill_detail b JOIN tab_frenchise f ON b.Fren_ID = f.Fren_ID WHERE f.Com_ID = :company_id`
  • Via site: `SELECT s.Site_ID, s.SName FROM tab_site s JOIN Tab_Bill b ON b.Site_ID = s.Site_ID WHERE s.Fren_ID = :franchise_id`
- If schema inspection shows multiple variant column names, use the exact column name returned by INFORMATION_SCHEMA.
- NEVER guess, NEVER hallucinate. If a requested filter cannot be applied using available schema columns and joins, explicitly say which column(s) are missing and offer an alternative (e.g., "I can filter by Franchise but not by Company for this table").


CRITICAL RULES:
- Use ONLY tables and columns that exist in the schema
- NEVER invent table or column names
- ALWAYS use the appropriate tool when data is required
- For database questions, you MUST use sql_db_query
- SQL must be valid SQL Server syntax
- If the answer requires data, query the database first
- If the question is not answerable from the database, say so clearly

AVAILABLE TOOLS:
- schema_lookup → Get table structure (call with table name or empty for list)
- sql_db_query → Execute SQL queries
- FranchisePerformanceEvaluation → Analyze franchise performance
- FranchiseComparison → Compare franchises
- BranchSitesInfo → Get site information for a franchise

SQL RULES:
- Always verify table and column names from the schema
- Use explicit JOIN conditions
- Limit results when appropriate (TOP, WHERE)
- Do NOT guess missing data
-when asking about any information answer with the name of it not the ids like customer name not customer id

IMPORTANT:
- Do NOT explain what you would do
- Do NOT make assumptions
- Use tools to get real data
"""

        suffix = """Begin!

Question: {input}
{agent_scratchpad}"""
        
        # # Create prompt
        prompt = ZeroShotAgent.create_prompt(
            tools=self.tools,
            prefix=prefix,
            suffix=suffix,
            input_variables=["input", "agent_scratchpad"]
        )

        from langchain.chains import LLMChain
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        
        # # # Create agent
        agent = ZeroShotAgent(llm_chain=llm_chain, tools=self.tools, stop=["\nObservation:"])
        
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=False,
            max_iterations=3,  
            handle_parsing_errors=False,
            early_stopping_method="generate",
            return_intermediate_steps=True

        )
        
    @lru_cache(maxsize=100)
    def _get_franchise_names_from_db(self) -> List[str]:
        """Cached franchise name retrieval"""
        try:
            query = "SELECT DISTINCT FName FROM tab_frenchise WHERE Com_ID = :company_id"
            rows = self._execute_query(query, {"company_id": self.config.active_company_id})
            names = [row.get("FName") for row in rows if row.get("FName")]
            return names
        except Exception as e:
            logger.error(f"Error fetching franchise names: {e}")
            return []
    
    def similar_branch(self, input_branch_names: List[str]) -> List[Optional[str]]:
        """Match input branch names to database names with fuzzy matching"""
        if not input_branch_names:
            return []
            
        try:
            db_names = self._get_franchise_names_from_db()
            if not db_names:
                return [None] * len(input_branch_names)
            
            results = []
            for branch_name in input_branch_names:
                if not branch_name or not isinstance(branch_name, str):
                    results.append(None)
                    continue
                
                branch_clean = branch_name.strip().lower()
                
                # Try exact match
                for db_name in db_names:
                    if db_name and db_name.lower() == branch_clean:
                        results.append(db_name)
                        break
                else:
                    # Try fuzzy
                    best_match = None
                    best_score = 0
                    for db_name in db_names:
                        if not db_name:
                            continue
                        score = fuzz.ratio(branch_clean, db_name.lower())
                        if score > best_score and score > SIMILARITY_THRESHOLD:
                            best_score = score
                            best_match = db_name
                    results.append(best_match)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in similar_branch: {e}")
            return [None] * len(input_branch_names)
    
    def get_branch_sites(self, branch_name: str):
        """Get sites for a branch"""
        try:
            matched = None
            db_names = self._get_franchise_names_from_db()
            
            for db_name in db_names:
                if db_name and db_name.lower() == branch_name.strip().lower():
                    matched = db_name
                    break
            
            if not matched:
                matches = self.similar_branch([branch_name])
                matched = matches[0] if matches else None
                
            if not matched:
                available = self._get_franchise_names_from_db()
                return {"error": f"No matching franchise found for: {branch_name}"}
            
            query = """SELECT 
                s.Site_ID,
                s.Sname,
                COUNT(DISTINCT b.Bill_ID) AS total_orders,
                SUM(b.Total) AS total_revenue,
                SUM(b.QUANTITY) AS total_items_sold,
                CASE 
                    WHEN COUNT(DISTINCT b.Bill_ID) > 0 
                    THEN SUM(b.Total) * 1.0 / COUNT(DISTINCT b.Bill_ID)
                    ELSE 0 
                END AS avg_order_value
            FROM tab_site s
            INNER JOIN tab_frenchise f ON s.Fren_ID = f.Fren_ID
            LEFT JOIN tab_Bill_detail b ON s.Site_ID = b.Site_ID
            WHERE f.Com_ID = :company_id AND f.FName = :branch_name
            GROUP BY s.Site_ID, s.Sname
            ORDER BY total_revenue DESC;"""
            
            rows = self._execute_query(query, {
                "company_id": self.config.active_company_id,
                "branch_name": matched
            })
            
            return rows
                
        except Exception as e:
            logger.error(f"Error in get_branch_sites: {e}", exc_info=True)
            return {"error": str(e)}
    
    class FranchisePerformanceEvaluator:
        def __init__(self, db: SQLDatabase, main_assistant):
            self.db = db
            self.assistant = main_assistant
        
        def evaluate_franchise_performance(
            self, 
            time_frame: str = "all", 
            franchise_names: Optional[List[str]] = None
        ) -> Dict[str, Any]:
            """Comprehensive franchise performance analysis"""
            franchise_names = franchise_names or []
            
            matched_names = self.assistant.similar_branch(franchise_names)
            valid_names = [name for name in matched_names if name is not None]
            
            if franchise_names and not valid_names:
                return {
                    "error": f"Could not find matching franchises for: {franchise_names}"
                }
            
            try:
                performance_data = self._get_basic_franchise_metrics(time_frame, valid_names)
                
                return {
                    "time_frame": time_frame,
                    "franchises": valid_names,
                    "performance_data": performance_data,
                }
            except Exception as e:
                logger.error(f"Performance evaluation failed: {e}", exc_info=True)
                return {
                    "error": str(e),
                    "time_frame": time_frame,
                    "franchises": valid_names
                }
        
        def _get_basic_franchise_metrics(
            self, 
            timeframe: str, 
            franchise_names: List[str]
        ) -> Dict[str, Any]:
            """Retrieve basic franchise metrics"""
            try:
                params = {"company_id": self.assistant.config.active_company_id}
                franchise_filter = ""
                
                if franchise_names:
                    placeholders = ", ".join([f":fname{i}" for i in range(len(franchise_names))])
                    franchise_filter = f" AND f.FName IN ({placeholders})"
                    for i, name in enumerate(franchise_names):
                        params[f"fname{i}"] = name
                
                time_filters = {
                    "last_month": "b.Bill_Date >= DATEADD(month, -1, GETDATE())",
                    "last_quarter": "b.Bill_Date >= DATEADD(month, -3, GETDATE())",
                    "last_year": "b.Bill_Date >= DATEADD(year, -1, GETDATE())",
                    "all": "1=1"
                }
                time_filter = time_filters.get(timeframe, time_filters["all"])
                
                query = f"""
                SELECT 
                    f.Fren_ID,
                    f.FName as Franchise_Name,
                    COUNT(DISTINCT b.Bill_ID) as Total_Orders,
                    COALESCE(SUM(b.Bill_Total), 0) as Total_Revenue,
                    COALESCE(AVG(b.Bill_Total), 0) AS Average_Order_Value,
                    COUNT(DISTINCT s.Site_ID) AS Number_of_Sites,
                    CASE 
                        WHEN COUNT(DISTINCT s.Site_ID) > 0 
                        THEN COALESCE(SUM(b.Bill_Total), 0) / COUNT(DISTINCT s.Site_ID) 
                        ELSE 0 
                    END AS Revenue_Per_Site
                FROM tab_frenchise f
                LEFT JOIN tab_site s ON f.Fren_ID = s.Fren_ID
                LEFT JOIN tab_Bill b ON s.Site_ID = b.Site_ID
                WHERE f.Com_ID = :company_id 
                    AND {time_filter}
                    {franchise_filter}
                GROUP BY f.Fren_ID, f.FName
                ORDER BY Total_Revenue DESC
                """
                
                rows = self.assistant._execute_query(query, params)
                return {"rows": rows}
                    
            except Exception as e:
                logger.error(f"Metrics query failed: {e}", exc_info=True)
                return {"error": str(e)}
    
    class FranchiseComparison:
        def __init__(self, db: SQLDatabase, main_assistant, evaluator):
            self.db = db
            self.assistant = main_assistant
            self.evaluator = evaluator
        
        def compare_franchises(
            self, 
            franchise_names: Optional[List[str]] = None
        ) -> Dict[str, Any]:
            """Compare multiple franchises"""
            if not franchise_names or len(franchise_names) < 2:
                return {"error": "Need at least 2 franchises to compare"}
            
            matched_names = self.assistant.similar_branch(franchise_names)
            valid_names = [name for name in matched_names if name is not None]
            
            if len(valid_names) < 2:
                return {
                    "error": f"Could not find enough matching franchises. Found: {valid_names}"
                }
            
            try:
                comparison_data = self._fetch_comparison_data(valid_names)
                
                if not comparison_data:
                    return {"error": f"No data found for franchises: {valid_names}"}
                
                return {
                    "franchises": valid_names,
                    "comparison_data": comparison_data,
                }
                
            except Exception as e:
                logger.error(f"Comparison failed: {e}", exc_info=True)
                return {"error": f"Comparison failed: {str(e)}"}
        
        def _fetch_comparison_data(self, franchise_names: List[str]) -> List[Dict]:
            """Fetch comparison data"""
            try:
                params = {"company_id": self.assistant.config.active_company_id}
                placeholders = ", ".join([f":fname{i}" for i in range(len(franchise_names))])
                for i, name in enumerate(franchise_names):
                    params[f"fname{i}"] = name
                
                query = f"""
                SELECT 
                    f.Fren_ID,
                    f.FName as Franchise_Name,
                    COUNT(DISTINCT b.Bill_ID) AS Total_Orders,
                    COALESCE(SUM(b.Bill_Total), 0) AS Total_Revenue,
                    COALESCE(AVG(b.Bill_Total), 0) AS Avg_Order_Value,
                    COUNT(DISTINCT s.Site_ID) AS Site_Count,
                    CASE 
                        WHEN COUNT(DISTINCT s.Site_ID) = 0 THEN 0
                        ELSE COALESCE(SUM(b.Bill_Total), 0) / COUNT(DISTINCT s.Site_ID)
                    END AS Revenue_Per_Site
                FROM tab_frenchise f
                LEFT JOIN tab_site s ON f.Fren_ID = s.Fren_ID
                LEFT JOIN tab_bill b ON s.Site_ID = b.Site_ID
                WHERE f.Com_ID = :company_id 
                    AND f.FName IN ({placeholders})
                GROUP BY f.Fren_ID, f.FName
                ORDER BY Total_Revenue DESC;
                """
                
                rows = self.assistant._execute_query(query, params)
                return rows
                    
            except Exception as e:
                logger.error(f"Fetch comparison data failed: {e}", exc_info=True)
                return []
    
    def smart_run(self, user_input: str) -> str:
        """
        Main entry point for processing user queries
        """
        try:
            user_input = user_input.strip()
            if not user_input:
                return "Please provide a question or request."
            
            user_input_lower = user_input.lower()
            
            # Handle simple queries
            if user_input_lower in ["hi", "hello", "hey"]:
                return f"Hello! I'm your business intelligence assistant for {self.company_name}. How can I help you today?"
            
            if user_input_lower in ["help", "what can you do"]:
                return f"""I can help you with {self.company_name} business intelligence:

                    1. Answer questions about customers, sales, and data
                    2. Analyze franchise performance
                    3. Compare different franchises
                    4. Get site information
                    5. Run SQL queries on the database
                    """
            logger.info(f"Processing query: {user_input}")
            
            try:
                result = self.agent_executor.invoke({"input": user_input})
                
                if isinstance(result, dict):
                    response = result.get("output", str(result))
                else:
                    response = str(result)
                

                if "Agent stopped" in response or "Final Answer:" in response:

                    if "Final Answer:" in response:
                        parts = response.split("Final Answer:")
                        if len(parts) > 1:
                            response = parts[1].strip()
                    elif "Thought:" in response:

                        thoughts = response.split("Thought:")
                        for thought in reversed(thoughts):
                            if "final answer" in thought.lower():
                                lines = thought.split("\n")
                                for i, line in enumerate(lines):
                                    if "final answer" in line.lower() and i + 1 < len(lines):
                                        response = lines[i + 1].strip()
                                        break
                                break
                
                return response
                
            except Exception as agent_error:
                logger.error(f"Agent execution failed: {agent_error}")
                return "I had trouble processing that request. Please try rephrasing your question or ask for help."
                
        except Exception as e:
            logger.error(f"smart_run failed: {e}", exc_info=True)
            return "I encountered an error. Please try again."
