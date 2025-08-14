# main.py - FastAPI Backend for Finance Copilot
from fastapi import FastAPI, HTTPException, Depends, Security, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
import asyncio
import pyodbc
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import logging
from contextlib import asynccontextmanager
import uuid
from collections import defaultdict
import jwt
from jwt import PyJWTError

# app/main.py (very top, before class Config)
from pathlib import Path
from dotenv import load_dotenv

# Azure imports
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from openai import AzureOpenAI
import msal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load ../.env relative to this file
load_dotenv()

# ===============================
# CONFIGURATION
# ===============================

class Config:
    """Application configuration"""
    
    # Azure AD Configuration - MUST BE UPDATED WITH YOUR VALUES
    AZURE_CLIENT_ID = os.getenv("AZURE_CLIENT_ID", "YOUR_CLIENT_ID_HERE")  # TODO: Replace
    AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID", "YOUR_TENANT_ID_HERE")  # TODO: Replace
    AZURE_CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET", "")  # For backend auth if needed
    
    # Azure OpenAI Configuration
    AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
    AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-35-turbo")
    AZURE_OPENAI_EMBEDDING_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    
    # SQL Database Configuration
    SQL_SERVER = os.getenv("SQL_SERVER")
    SQL_DATABASE = os.getenv("SQL_DATABASE")
    SQL_USERNAME = os.getenv("SQL_USERNAME")
    SQL_PASSWORD = os.getenv("SQL_PASSWORD")
    SQL_DRIVER = os.getenv("SQL_DRIVER", "{ODBC Driver 17 for SQL Server}")
    
    # Power BI Configuration
    POWER_BI_WORKSPACE_ID = os.getenv("POWER_BI_WORKSPACE_ID")
    POWER_BI_REPORT_ID = os.getenv("POWER_BI_REPORT_ID")
    POWER_BI_DEFAULT_PAGE = os.getenv("POWER_BI_DEFAULT_PAGE")
    
    # Key Vault Configuration (for production)
    # KEY_VAULT_NAME = os.getenv("KEY_VAULT_NAME", "your-keyvault-name")
    # KEY_VAULT_URL = f"https://{KEY_VAULT_NAME}.vault.azure.net/"
    
    # Application Settings
    ENABLE_AUTH = os.getenv("ENABLE_AUTH", "false").lower() == "true"  # Set to true in production
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Session Management
    SESSION_TIMEOUT_MINUTES = int(os.getenv("SESSION_TIMEOUT_MINUTES", "30"))
    MAX_CONVERSATION_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY", "50"))

config = Config()

# ===============================
# AZURE OPENAI CLIENT
# ===============================

try:
    openai_client = AzureOpenAI(
        api_key=config.AZURE_OPENAI_KEY,
        api_version=config.AZURE_OPENAI_API_VERSION,
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT
    )
    logger.info("Azure OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Azure OpenAI client: {str(e)}")
    openai_client = None

# ===============================
# SECURITY & AUTHENTICATION
# ===============================

security = HTTPBearer(auto_error=False)

class UserContext(BaseModel):
    """User context from Azure AD token"""
    user_id: str
    name: str
    email: Optional[str] = None
    roles: List[str] = Field(default_factory=list)
    permissions: Dict[str, Any] = Field(default_factory=dict)

# In-memory session storage (use Redis in production)
class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.conversation_history: Dict[str, List[Dict]] = defaultdict(list)
    
    def create_session(self, user_id: str) -> str:
        """Create new session for user"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "user_id": user_id,
            "created_at": datetime.utcnow(),
            "last_activity": datetime.utcnow()
        }
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session if valid"""
        session = self.sessions.get(session_id)
        if session:
            # Check timeout
            if datetime.utcnow() - session["last_activity"] > timedelta(minutes=config.SESSION_TIMEOUT_MINUTES):
                del self.sessions[session_id]
                return None
            session["last_activity"] = datetime.utcnow()
        return session
    
    def add_to_history(self, user_id: str, message: str, response: str, intent: str):
        """Add conversation to history"""
        history = self.conversation_history[user_id]
        history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "message": message,
            "response": response,
            "intent": intent
        })
        # Keep only last N messages
        if len(history) > config.MAX_CONVERSATION_HISTORY:
            self.conversation_history[user_id] = history[-config.MAX_CONVERSATION_HISTORY:]
    
    def get_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get conversation history for user"""
        return self.conversation_history[user_id][-limit:]

session_manager = SessionManager()

# ===============================
# PYDANTIC MODELS
# ===============================

class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None
    include_history: bool = False

class ChatResponse(BaseModel):
    intent: Literal["visual_update", "sql_query", "rag_lookup", "mixed", "needs_clarification"]
    response: str
    suggestions: List[str]
    meta: Optional[Any] = None 
    data: Optional[Dict[str, Any]] = None
    sql_query: Optional[str] = None
    session_id: Optional[str] = None
    confidence: float = Field(ge=0, le=1)
    show_table: Optional[bool] = False  # NEW: Signal to show table
    table_data: Optional[List[Dict[str, Any]]] = None  # NEW: Structured table data
    table_title: Optional[str] = None

class SQLQueryRequest(BaseModel):
    query: str
    parameters: Optional[Dict[str, Any]] = None

class ConversationHistory(BaseModel):
    messages: List[Dict[str, Any]]
    total_count: int
    session_id: str

# ===============================
# DATABASE MANAGER
# ===============================

class DatabaseManager:
    """Manages database connections with connection pooling and RLS support"""
    
    def __init__(self):
        # Build connection string directly without template formatting
        self.default_connection_string = (
            f"DRIVER={config.SQL_DRIVER};"
            f"SERVER={config.SQL_SERVER};"
            f"DATABASE={config.SQL_DATABASE};"
            f"UID={config.SQL_USERNAME};"
            f"PWD={config.SQL_PASSWORD};"
            "Encrypt=yes;"
            "TrustServerCertificate=no;"
            "Connection Timeout=30;"
            "ApplicationIntent=ReadOnly;"  # For read replicas if available
        )
    
    def get_connection_string(self, user_context: Optional[UserContext] = None) -> str:
        """Get connection string based on user context"""
        # In production with proper SSO, you could build user-specific connection strings
        # For now, use the default service account connection
        if user_context and config.ENABLE_AUTH and user_context.email:
            # Example of building user-specific connection (not used currently)
            return (
                f"DRIVER={config.SQL_DRIVER};"
                f"SERVER={config.SQL_SERVER};"
                f"DATABASE={config.SQL_DATABASE};"
                f"UID={config.SQL_USERNAME};"
                f"PWD={config.SQL_PASSWORD};"
                "Encrypt=yes;"
                "TrustServerCertificate=no;"
                "Connection Timeout=30;"
            )
        return self.default_connection_string
    
    async def execute_query(
        self, 
        query: str, 
        params: Dict = None,
        user_context: Optional[UserContext] = None
    ) -> pd.DataFrame:
        """Execute SQL query with optional RLS filtering"""
        try:
            # Add RLS filtering if user context provided
            if user_context and config.ENABLE_AUTH:
                query = self._apply_rls(query, user_context)
            
            connection_string = self.get_connection_string(user_context)
            
            # Use context manager for connection
            with pyodbc.connect(connection_string) as conn:
                if params:
                    df = pd.read_sql(query, conn, params=params)
                else:
                    df = pd.read_sql(query, conn)
                
                logger.info(f"Query executed successfully, returned {len(df)} rows")
                return df
                
        except pyodbc.Error as e:
            logger.error(f"Database error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error executing query: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Query execution error: {str(e)}")
    
    def _apply_rls(self, query: str, user_context: UserContext) -> str:
        """Apply Row-Level Security based on user context"""
        # This is a simplified example - implement based on your RLS requirements
        # You might want to:
        # 1. Add user context as session context in SQL
        # 2. Use database roles
        # 3. Add WHERE clauses based on user permissions
        
        # Example: Add user context as a filter
        # This would need to be customized based on your data model
        rls_filter = ""
        
        if "admin" not in user_context.roles:
            # Non-admin users see filtered data
            # Example: Filter by user's region or department
            if user_context.permissions.get("region"):
                rls_filter = f" /* RLS: Region = '{user_context.permissions['region']}' */"
        
        return query + rls_filter
    
    async def test_connection(self, user_context: Optional[UserContext] = None) -> bool:
        """Test database connection"""
        try:
            test_query = "SELECT 1 as test, @@VERSION as version"
            result = await self.execute_query(test_query, user_context=user_context)
            logger.info(f"Database connection successful: {result['version'].iloc[0][:50]}...")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {str(e)}")
            return False
    
    async def get_table_schema(self, table_name: str = None) -> pd.DataFrame:
        """Get database schema information"""
        schema_query = """
        SELECT 
            TABLE_SCHEMA,
            TABLE_NAME,
            COLUMN_NAME,
            DATA_TYPE,
            IS_NULLABLE,
            COLUMN_DEFAULT
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = 'SalesLT'
        """
        if table_name:
            schema_query += f" AND TABLE_NAME = '{table_name}'"
        schema_query += " ORDER BY TABLE_NAME, ORDINAL_POSITION"
        
        return await self.execute_query(schema_query)

# ===============================
# FINANCE QUERIES
# ===============================

class FinanceQueries:
    """Centralized repository of financial analytics queries"""
    
    @staticmethod
    def get_query_templates() -> Dict[str, str]:
        """Get all predefined query templates"""
        return {
            # === REVENUE & SALES ANALYTICS ===
            
            "top_products_by_revenue": """
                SELECT TOP 20
                    p.Name as ProductName,
                    p.ProductNumber,
                    pc.Name as CategoryName,
                    FORMAT(SUM(sod.OrderQty * sod.UnitPrice), 'C', 'en-US') as TotalRevenue,
                    SUM(sod.OrderQty) as TotalQuantitySold,
                    FORMAT(AVG(sod.UnitPrice), 'C', 'en-US') as AvgSellingPrice,
                    COUNT(DISTINCT soh.SalesOrderID) as NumberOfOrders,
                    COUNT(DISTINCT soh.CustomerID) as UniqueCustomers,
                    CONVERT(varchar, MAX(soh.OrderDate), 107) as LastSaleDate
                FROM SalesLT.SalesOrderDetail sod
                JOIN SalesLT.SalesOrderHeader soh ON sod.SalesOrderID = soh.SalesOrderID
                JOIN SalesLT.Product p ON sod.ProductID = p.ProductID
                LEFT JOIN SalesLT.ProductCategory pc ON p.ProductCategoryID = pc.ProductCategoryID
                GROUP BY p.ProductID, p.Name, p.ProductNumber, pc.Name
                ORDER BY SUM(sod.OrderQty * sod.UnitPrice) DESC
            """,
            
            "top_customers_by_value": """
                SELECT TOP 20
                    c.CustomerID,
                    CONCAT(c.FirstName, ' ', c.LastName) as CustomerName,
                    c.CompanyName,
                    FORMAT(SUM(soh.TotalDue), 'C', 'en-US') as TotalRevenue,
                    COUNT(DISTINCT soh.SalesOrderID) as TotalOrders,
                    FORMAT(AVG(soh.TotalDue), 'C', 'en-US') as AvgOrderValue,
                    CONVERT(varchar, MIN(soh.OrderDate), 107) as FirstOrderDate,
                    CONVERT(varchar, MAX(soh.OrderDate), 107) as LastOrderDate,
                    DATEDIFF(day, MIN(soh.OrderDate), MAX(soh.OrderDate)) as CustomerLifespanDays,
                    FORMAT(SUM(soh.TotalDue) / NULLIF(COUNT(DISTINCT soh.SalesOrderID), 0), 'C', 'en-US') as RevenuePerOrder
                FROM SalesLT.Customer c
                JOIN SalesLT.SalesOrderHeader soh ON c.CustomerID = soh.CustomerID
                GROUP BY c.CustomerID, c.FirstName, c.LastName, c.CompanyName
                ORDER BY SUM(soh.TotalDue) DESC
            """,
            
            "monthly_revenue_trends": """
                SELECT 
                    YEAR(OrderDate) as Year,
                    MONTH(OrderDate) as Month,
                    DATENAME(month, OrderDate) + ' ' + CAST(YEAR(OrderDate) as VARCHAR) as MonthYear,
                    FORMAT(SUM(TotalDue), 'C', 'en-US') as MonthlyRevenue,
                    COUNT(DISTINCT SalesOrderID) as OrderCount,
                    COUNT(DISTINCT CustomerID) as UniqueCustomers,
                    FORMAT(AVG(TotalDue), 'C', 'en-US') as AvgOrderValue,
                    FORMAT(SUM(TotalDue) / NULLIF(COUNT(DISTINCT CustomerID), 0), 'C', 'en-US') as RevenuePerCustomer
                FROM SalesLT.SalesOrderHeader 
                GROUP BY YEAR(OrderDate), MONTH(OrderDate), DATENAME(month, OrderDate)
                ORDER BY Year DESC, Month DESC
            """,
            
            "quarterly_performance": """
                WITH QuarterlyData AS (
                    SELECT 
                        YEAR(OrderDate) as Year,
                        DATEPART(quarter, OrderDate) as Quarter,
                        SUM(TotalDue) as QuarterlyRevenue,
                        COUNT(DISTINCT SalesOrderID) as OrderCount,
                        COUNT(DISTINCT CustomerID) as UniqueCustomers,
                        AVG(TotalDue) as AvgOrderValue
                    FROM SalesLT.SalesOrderHeader 
                    GROUP BY YEAR(OrderDate), DATEPART(quarter, OrderDate)
                )
                SELECT 
                    Year,
                    Quarter,
                    'Q' + CAST(Quarter as VARCHAR) + ' ' + CAST(Year as VARCHAR) as QuarterLabel,
                    FORMAT(QuarterlyRevenue, 'C', 'en-US') as QuarterlyRevenue,
                    OrderCount,
                    UniqueCustomers,
                    FORMAT(AvgOrderValue, 'C', 'en-US') as AvgOrderValue,
                    FORMAT(LAG(QuarterlyRevenue) OVER (ORDER BY Year, Quarter), 'C', 'en-US') as PreviousQuarterRevenue,
                    CASE 
                        WHEN LAG(QuarterlyRevenue) OVER (ORDER BY Year, Quarter) IS NOT NULL 
                        THEN CAST(ROUND(((QuarterlyRevenue - LAG(QuarterlyRevenue) OVER (ORDER BY Year, Quarter)) * 100.0 / 
                             NULLIF(LAG(QuarterlyRevenue) OVER (ORDER BY Year, Quarter), 0)), 2) as VARCHAR) + '%'
                        ELSE 'N/A'
                    END as QoQ_Growth
                FROM QuarterlyData
                ORDER BY Year DESC, Quarter DESC
            """,
            
            "product_profitability_analysis": """
                SELECT TOP 25
                    p.Name as ProductName,
                    p.ProductNumber,
                    pc.Name as CategoryName,
                    FORMAT(p.StandardCost, 'C', 'en-US') as StandardCost,
                    FORMAT(AVG(sod.UnitPrice), 'C', 'en-US') as AvgSellingPrice,
                    FORMAT(AVG(sod.UnitPrice - p.StandardCost), 'C', 'en-US') as AvgProfitPerUnit,
                    CAST(CASE 
                        WHEN p.StandardCost > 0 
                        THEN ROUND((AVG(sod.UnitPrice - p.StandardCost) * 100.0 / p.StandardCost), 2)
                        ELSE 0 
                    END as VARCHAR) + '%' as ProfitMargin,
                    FORMAT(SUM(sod.OrderQty * (sod.UnitPrice - p.StandardCost)), 'C', 'en-US') as TotalProfit,
                    FORMAT(SUM(sod.OrderQty * sod.UnitPrice), 'C', 'en-US') as TotalRevenue,
                    SUM(sod.OrderQty) as TotalQuantitySold
                FROM SalesLT.Product p
                JOIN SalesLT.SalesOrderDetail sod ON p.ProductID = sod.ProductID
                JOIN SalesLT.SalesOrderHeader soh ON sod.SalesOrderID = soh.SalesOrderID
                LEFT JOIN SalesLT.ProductCategory pc ON p.ProductCategoryID = pc.ProductCategoryID
                WHERE p.StandardCost > 0
                GROUP BY p.ProductID, p.Name, p.ProductNumber, p.StandardCost, pc.Name
                HAVING SUM(sod.OrderQty) > 0
                ORDER BY SUM(sod.OrderQty * (sod.UnitPrice - p.StandardCost)) DESC
            """,
            
            "customer_segmentation_analysis": """
                WITH CustomerTiers AS (
                    SELECT 
                        c.CustomerID,
                        CONCAT(c.FirstName, ' ', c.LastName) as CustomerName,
                        c.CompanyName,
                        SUM(soh.TotalDue) as TotalRevenue,
                        COUNT(DISTINCT soh.SalesOrderID) as OrderCount,
                        AVG(soh.TotalDue) as AvgOrderValue,
                        CASE 
                            WHEN SUM(soh.TotalDue) >= 10000 THEN 'Platinum'
                            WHEN SUM(soh.TotalDue) >= 5000 THEN 'Gold'
                            WHEN SUM(soh.TotalDue) >= 2000 THEN 'Silver'
                            ELSE 'Bronze'
                        END as CustomerTier
                    FROM SalesLT.Customer c
                    JOIN SalesLT.SalesOrderHeader soh ON c.CustomerID = soh.CustomerID
                    GROUP BY c.CustomerID, c.FirstName, c.LastName, c.CompanyName
                )
                SELECT 
                    CustomerTier,
                    COUNT(*) as CustomerCount,
                    FORMAT(SUM(TotalRevenue), 'C', 'en-US') as TierTotalRevenue,
                    FORMAT(AVG(TotalRevenue), 'C', 'en-US') as AvgRevenuePerCustomer,
                    CAST(AVG(OrderCount) as INT) as AvgOrdersPerCustomer,
                    FORMAT(AVG(AvgOrderValue), 'C', 'en-US') as AvgOrderValue,
                    CAST(ROUND((SUM(TotalRevenue) * 100.0 / (SELECT SUM(TotalRevenue) FROM CustomerTiers)), 2) as VARCHAR) + '%' as RevenueShare
                FROM CustomerTiers
                GROUP BY CustomerTier
                ORDER BY 
                    CASE CustomerTier 
                        WHEN 'Platinum' THEN 1 
                        WHEN 'Gold' THEN 2 
                        WHEN 'Silver' THEN 3 
                        ELSE 4 
                    END
            """,
            
            "yearly_comparison_analysis": """
                WITH YearlyData AS (
                    SELECT 
                        YEAR(OrderDate) as Year,
                        SUM(TotalDue) as YearlyRevenue,
                        COUNT(DISTINCT SalesOrderID) as OrderCount,
                        COUNT(DISTINCT CustomerID) as UniqueCustomers,
                        AVG(TotalDue) as AvgOrderValue
                    FROM SalesLT.SalesOrderHeader 
                    GROUP BY YEAR(OrderDate)
                )
                SELECT 
                    Year,
                    FORMAT(YearlyRevenue, 'C', 'en-US') as YearlyRevenue,
                    OrderCount,
                    UniqueCustomers,
                    FORMAT(AvgOrderValue, 'C', 'en-US') as AvgOrderValue,
                    FORMAT(LAG(YearlyRevenue) OVER (ORDER BY Year), 'C', 'en-US') as PreviousYearRevenue,
                    CASE 
                        WHEN LAG(YearlyRevenue) OVER (ORDER BY Year) IS NOT NULL 
                        THEN CAST(ROUND(((YearlyRevenue - LAG(YearlyRevenue) OVER (ORDER BY Year)) * 100.0 / 
                             NULLIF(LAG(YearlyRevenue) OVER (ORDER BY Year), 0)), 2) as VARCHAR) + '%'
                        ELSE 'N/A'
                    END as YoY_Growth,
                    FORMAT(YearlyRevenue / NULLIF(OrderCount, 0), 'C', 'en-US') as RevenuePerOrder,
                    FORMAT(YearlyRevenue / NULLIF(UniqueCustomers, 0), 'C', 'en-US') as RevenuePerCustomer
                FROM YearlyData
                ORDER BY Year DESC
            """,
            
            "executive_dashboard_kpis": """
                WITH BaseOrders AS (
                    SELECT soh.*
                    FROM SalesLT.SalesOrderHeader AS soh
                    WHERE 1=1 /*TIME_FILTERS*/
                ),     

                OverallMetrics AS (
                    SELECT 
                        SUM(TotalDue) as TotalRevenue,
                        COUNT(DISTINCT SalesOrderID) as TotalOrders,
                        COUNT(DISTINCT CustomerID) as TotalCustomers,
                        AVG(TotalDue) as AvgOrderValue,
                        MIN(OrderDate) as EarliestOrder,
                        MAX(OrderDate) as LatestOrder
                    FROM BaseOrders
                ),
                ProductMetrics AS (
                    SELECT 
                        COUNT(DISTINCT p.ProductID) as TotalProducts,
                        COUNT(DISTINCT pc.ProductCategoryID) as TotalCategories
                    FROM SalesLT.Product p
                    LEFT JOIN SalesLT.ProductCategory pc ON p.ProductCategoryID = pc.ProductCategoryID
                ),
                RecentMetrics AS (
                    SELECT 
                        SUM(TotalDue) as Last30DaysRevenue,
                        COUNT(DISTINCT SalesOrderID) as Last30DaysOrders
                    FROM SalesLT.SalesOrderHeader
                    WHERE OrderDate >= DATEADD(day, -30, GETDATE())
                )
                SELECT 
                    FORMAT(om.TotalRevenue, 'C', 'en-US') as TotalRevenue,
                    om.TotalOrders,
                    om.TotalCustomers,
                    pm.TotalProducts,
                    pm.TotalCategories,
                    FORMAT(om.AvgOrderValue, 'C', 'en-US') as AvgOrderValue,
                    FORMAT(om.TotalRevenue / NULLIF(om.TotalCustomers, 0), 'C', 'en-US') as RevenuePerCustomer,
                    FORMAT(rm.Last30DaysRevenue, 'C', 'en-US') as Last30DaysRevenue,
                    rm.Last30DaysOrders as Last30DaysOrders,
                    DATEDIFF(day, om.EarliestOrder, om.LatestOrder) as BusinessDays,
                    FORMAT(om.TotalRevenue / NULLIF(DATEDIFF(day, om.EarliestOrder, om.LatestOrder), 0), 'C', 'en-US') as AvgDailyRevenue
                FROM OverallMetrics om, ProductMetrics pm, RecentMetrics rm
            """
        }
    
    @staticmethod
    def get_dynamic_query(query_type: str, filters: Dict[str, Any] = None) -> str:
        """Generate dynamic queries based on filters"""
        base_queries = FinanceQueries.get_query_templates()
        
        if query_type not in base_queries:
            raise ValueError(f"Unknown query type: {query_type}")
        
        query = base_queries[query_type]
        
        # Add dynamic filters if provided
        if filters:
            where_clauses = []
            
            if filters.get("start_date"):
                where_clauses.append(f"OrderDate >= '{filters['start_date']}'")
            
            if filters.get("end_date"):
                where_clauses.append(f"OrderDate <= '{filters['end_date']}'")
            
            if filters.get("category"):
                where_clauses.append(f"pc.Name = '{filters['category']}'")
            
            if filters.get("customer_id"):
                where_clauses.append(f"c.CustomerID = {filters['customer_id']}")
            
            # Add WHERE clause to query if filters exist
            if where_clauses:
                # This is simplified - in production, properly parse and inject WHERE clauses
                where_clause = " AND ".join(where_clauses)
                # You'd need more sophisticated query parsing here
                logger.info(f"Dynamic filters applied: {where_clause}")
        
        return query

# ===============================
# INTENT ANALYSIS
# ===============================

class IntentAnalyzer:
    """Analyzes user intent using Azure OpenAI"""
    
    def __init__(self):
        self.intent_system_prompt = """You are a financial analytics intent classifier for a Finance Copilot system.

Classify user queries into EXACTLY ONE of these intents:

1. **sql_query**: User wants data, analytics, calculations, metrics, KPIs, or any numerical analysis
   - Examples: "top 5 customers", "monthly revenue", "profit margins", "sales trends", "compare Q1 vs Q2"
   - Keywords: top, best, highest, lowest, average, total, sum, count, trends, analysis, compare, show, display, list

2. **visual_update**: User wants to filter or update Power BI dashboard visuals
   - Examples: "filter dashboard to 2024", "show only US region", "update chart to Q3"
   - Keywords: filter, update dashboard, change view (specifically for dashboards)

3. **rag_lookup**: User asks about documents, reports, or meeting notes
   - Examples: "what did the board say", "find in Q2 report", "search meeting notes"
   - Keywords: document, report, meeting, notes, said, mentioned, wrote

4. **mixed**: Combination of visual update AND data query
   - Examples: "filter to 2024 and show top products", "update dashboard and compare regions"

5. **needs_clarification**: Query is too vague or unclear
   - Examples: "help", "what can you do", "hi", unclear requests

IMPORTANT: Respond with ONLY the format: intent|confidence
Do NOT include the word "Intent:" or any other text.
Example responses:
- sql_query|0.95
- visual_update|0.90
- needs_clarification|0.70

Most queries about data should be classified as sql_query."""
    
    async def analyze_intent(self, message: str, context: List[Dict] = None) -> tuple[str, float]:
        """Analyze user message intent"""
        if not openai_client:
            logger.error("OpenAI client not initialized")
            return "needs_clarification", 0.5
        
        try:
            # Include conversation context if available
            messages = [{"role": "system", "content": self.intent_system_prompt}]
            
            if context:
                # Add last 3 messages for context
                for ctx in context[-3:]:
                    messages.append({"role": "user", "content": ctx.get("message", "")})
                    messages.append({"role": "assistant", "content": f"Intent: {ctx.get('intent', '')}"})
            
            messages.append({"role": "user", "content": message})
            
            response = openai_client.chat.completions.create(
                model=config.AZURE_OPENAI_DEPLOYMENT,
                messages=messages,
                max_tokens=50,
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip()
            logger.info(f"Raw intent result: '{result}'")
            
            # Aggressive cleaning of the result
            # Remove all variations of "Intent:" prefix
            cleaned_result = result
            for prefix in ["Intent:", "intent:", "Intent", "intent"]:
                if prefix in cleaned_result:
                    # Find the position and remove everything before and including it
                    pos = cleaned_result.find(prefix)
                    if pos >= 0:
                        cleaned_result = cleaned_result[pos + len(prefix):].strip()
                        break
            
            # Remove any leading colons or spaces
            cleaned_result = cleaned_result.lstrip(': ').strip()
            
            logger.info(f"Cleaned intent result: '{cleaned_result}'")
            
            # Try to parse the intent and confidence
            if "|" in cleaned_result:
                parts = cleaned_result.split("|")
                intent = parts[0].strip().lower()
                
                # Additional cleaning
                intent = intent.replace("sql_query", "sql_query")  # Normalize
                intent = intent.strip()
                
                try:
                    confidence = float(parts[1].strip())
                except (ValueError, IndexError):
                    confidence = 0.8
                
                # Validate intent
                valid_intents = ["sql_query", "visual_update", "rag_lookup", "mixed", "needs_clarification"]
                if intent in valid_intents:
                    logger.info(f"Successfully parsed - Intent: {intent}, Confidence: {confidence}")
                    return intent, confidence
                else:
                    logger.warning(f"Invalid intent '{intent}' not in valid list")
                    # Try to find a valid intent in the original response
                    for valid_intent in valid_intents:
                        if valid_intent in result.lower():
                            logger.info(f"Found valid intent '{valid_intent}' in original response")
                            return valid_intent, 0.7
                    return "needs_clarification", 0.5
            else:
                # Try to extract just the intent without confidence
                intent = cleaned_result.lower().strip()
                
                valid_intents = ["sql_query", "visual_update", "rag_lookup", "mixed", "needs_clarification"]
                if intent in valid_intents:
                    logger.info(f"Found intent without confidence: {intent}")
                    return intent, 0.8
                else:
                    # Check if any valid intent is in the response
                    for valid_intent in valid_intents:
                        if valid_intent in cleaned_result.lower():
                            logger.info(f"Found partial match for intent: {valid_intent}")
                            return valid_intent, 0.7
                    
                    logger.warning(f"Could not parse any valid intent from: '{result}'")
                    return "needs_clarification", 0.5
                
        except Exception as e:
            logger.error(f"Intent analysis error: {str(e)}", exc_info=True)
            return "needs_clarification", 0.5

# ===============================
# RESPONSE GENERATION
# ===============================

class ResponseGenerator:
    """Generates natural language responses"""
    
    def __init__(self):
        self.response_system_prompt = """You are a professional Finance Copilot assistant providing financial analytics insights.

Guidelines:
- Be concise but comprehensive
- Use proper financial terminology
- Format currency values appropriately
- Highlight key insights and trends
- Provide actionable business intelligence
- For lists/rankings, clearly enumerate each item
- Include relevant percentages and comparisons
- Maintain professional yet conversational tone

When presenting data:
- For top N queries: List each item with its key metrics
- For trends: Highlight growth rates and patterns
- For comparisons: Show differences and percentages
- For KPIs: Explain what the numbers mean for the business

Do not include suggestions in your response - focus only on answering the question with data insights."""
    
    async def generate_response(
        self, 
        intent: str, 
        user_message: str, 
        data: pd.DataFrame = None, 
        context: str = "",
        error: str = None
    ) -> Dict[str, Any]:
        """Generate natural language response based on intent and data"""
        
        if not openai_client:
            return {
                "response": "I'm having trouble connecting to the AI service. Please try again later.",
                "suggestions": self.get_default_suggestions(intent)
            }
        
        try:
            # Handle error cases
            if error:
                return {
                    "response": f"I encountered an issue processing your request: {error}. Please try rephrasing your question or selecting from the suggestions below.",
                    "suggestions": self.get_default_suggestions("needs_clarification")
                }
            
            # Check if this is a query that should return tabular data
            should_show_table = (
                intent == "sql_query" and 
                data is not None and 
                not data.empty and
                len(data) > 1  # More than 1 row suggests tabular data
            )
            
            if should_show_table:
                # For tabular data, provide a concise summary and let the table speak for itself
                summary_prompt = f"""
                Intent: {intent}
                User Query: "{user_message}"
                Context: {context}
                
                Data Summary: {len(data)} records returned from the database.
                Sample columns: {', '.join(data.columns[:5].tolist())}
                
                Provide a BRIEF (2-3 sentences max) summary of what this data shows. 
                Do NOT list individual rows - the user will see them in a table.
                Focus on high-level insights like trends, totals, or key findings.
                Keep it conversational and professional.
                """
                
                response = openai_client.chat.completions.create(
                    model=config.AZURE_OPENAI_DEPLOYMENT,
                    messages=[
                        {"role": "system", "content": self.response_system_prompt},
                        {"role": "user", "content": summary_prompt}
                    ],
                    max_tokens=200,  # Keep it brief
                    temperature=0.3
                )
                
                brief_summary = response.choices[0].message.content.strip()
                
                return {
                    "response": brief_summary,
                    "suggestions": self.get_contextual_suggestions(intent, user_message, data)[:4],
                    "show_table": True,  # Signal to frontend to show the table
                    "table_data": data.to_dict('records'),  # Structured data for the table
                    "table_title": self._get_table_title(user_message, intent)
                }
            else:
                # For non-tabular responses, use the existing logic
                data_summary = ""
                if data is not None and not data.empty:
                    data_summary = f"Query Results Summary:\n{data.head(5).to_string(index=False)}\n"
                    if len(data) > 5:
                        data_summary += f"\n... and {len(data) - 5} more records"
                
                prompt = f"""Intent: {intent}
User Query: "{user_message}"
Context: {context}

Data Results:
{data_summary if data_summary else "No data available"}

Please provide a comprehensive response that directly answers the user's question.
Keep the response professional but conversational."""
                
                response = openai_client.chat.completions.create(
                    model=config.AZURE_OPENAI_DEPLOYMENT,
                    messages=[
                        {"role": "system", "content": self.response_system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=800,
                    temperature=0.3
                )
                
                response_text = response.choices[0].message.content.strip()
                suggestions = self.get_contextual_suggestions(intent, user_message, data)
                
                return {
                    "response": response_text,
                    "suggestions": suggestions[:4],
                    "show_table": False
                }
        except Exception as e:
            logger.error(f"Response generation error: {str(e)}")
            return {
                "response": self.get_fallback_response(intent, data),
                "suggestions": self.get_default_suggestions(intent),
                "show_table": False
            }

    def _get_table_title(self, user_message: str, intent: str) -> str:
        """Generate appropriate table title based on user query"""
        message_lower = user_message.lower()
        
        if "customer" in message_lower:
            if "top" in message_lower:
                return "Top Customers by Revenue"
            return "Customer Analysis"
        elif "product" in message_lower:
            if "top" in message_lower:
                return "Top Products by Revenue"
            elif "profit" in message_lower:
                return "Product Profitability Analysis"
            return "Product Analysis"
        elif "monthly" in message_lower:
            return "Monthly Performance"
        elif "quarterly" in message_lower:
            return "Quarterly Performance"
        elif "yearly" in message_lower or "annual" in message_lower:
            return "Annual Performance"
        else:
            return "Analysis Results"
    
    def get_fallback_response(self, intent: str, data: pd.DataFrame = None) -> str:
        """Generate fallback response when AI fails"""
        if intent == "sql_query" and data is not None and not data.empty:
            return f"I've retrieved {len(data)} records based on your query. The data is displayed below."
        elif intent == "visual_update":
            return "Dashboard filtering is currently in development. Please manually adjust the Power BI filters for now."
        elif intent == "rag_lookup":
            return "Document search functionality will be available soon. Currently, I can help with data analytics and reporting."
        else:
            return "I'm here to help with your financial analytics needs. Please try asking about sales trends, top customers, or revenue analysis."
    
    def get_contextual_suggestions(self, intent: str, message: str, data: pd.DataFrame = None) -> List[str]:
        """Generate contextual suggestions based on the current query"""
        suggestions = []
        message_lower = message.lower()
        
        if intent == "sql_query":
            if "customer" in message_lower:
                suggestions.extend([
                    "Show customer profitability analysis",
                    "View customer retention metrics",
                    "Analyze customer segmentation",
                    "Compare customer lifetime values"
                ])
            elif "product" in message_lower:
                suggestions.extend([
                    "Analyze product profitability",
                    "Show products by category",
                    "View product sales trends",
                    "Compare product performance"
                ])
            elif "revenue" in message_lower or "sales" in message_lower:
                suggestions.extend([
                    "Show quarterly performance",
                    "Analyze yearly comparisons",
                    "View revenue by region",
                    "Display monthly trends"
                ])
            elif "month" in message_lower or "quarter" in message_lower or "year" in message_lower:
                suggestions.extend([
                    "Compare with previous period",
                    "Show growth rates",
                    "Analyze seasonal patterns",
                    "View trend analysis"
                ])
            else:
                suggestions.extend([
                    "Show executive dashboard KPIs",
                    "View top products by revenue",
                    "Analyze customer segments",
                    "Display monthly revenue trends"
                ])
        
        elif intent == "visual_update":
            suggestions.extend([
                "Filter by different time period",
                "Show specific product category",
                "Focus on top customers",
                "Change to regional view"
            ])
        
        elif intent == "mixed":
            suggestions.extend([
                "Get detailed breakdown",
                "Export analysis to Excel",
                "Compare with benchmarks",
                "Show underlying data"
            ])
        
        else:  # needs_clarification or rag_lookup
            suggestions.extend([
                "Show top 10 customers by revenue",
                "Display monthly sales trends",
                "Analyze product profitability",
                "View executive dashboard"
            ])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for s in suggestions:
            if s not in seen:
                seen.add(s)
                unique_suggestions.append(s)
        
        return unique_suggestions
    
    def get_default_suggestions(self, intent: str) -> List[str]:
        """Get default suggestions based on intent"""
        suggestions_map = {
            "sql_query": [
                "Show top products by revenue",
                "Analyze monthly trends",
                "View customer segments",
                "Display profitability metrics"
            ],
            "visual_update": [
                "Filter to current year",
                "Show different regions",
                "Focus on specific category",
                "Update time period"
            ],
            "mixed": [
                "Get detailed analysis",
                "Compare periods",
                "Export to Excel",
                "View full report"
            ],
            "rag_lookup": [
                "Search for financial reports",
                "Find meeting notes",
                "Look up policy documents",
                "View historical analyses"
            ],
            "needs_clarification": [
                "Show executive dashboard",
                "List top 10 customers",
                "Display revenue trends",
                "Analyze product performance"
            ]
        }
        return suggestions_map.get(intent, ["Ask about sales", "View analytics", "Get insights"])

# ===============================
# QUERY PROCESSOR
# ===============================

class QueryProcessor:
    """Main query processing engine"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.intent_analyzer = IntentAnalyzer()
        self.response_generator = ResponseGenerator()
        self.finance_queries = FinanceQueries()
    
    async def process_message(
        self, 
        message: str, 
        session_id: str = None,
        user_context: Optional[UserContext] = None
    ) -> ChatResponse:
        """Process chat message and return response"""
        try:
            # Get conversation history for context
            history = []
            if user_context and user_context.user_id:
                history = session_manager.get_history(user_context.user_id, limit=5)
            
            # Analyze intent with context
            intent, confidence = await self.intent_analyzer.analyze_intent(message, history)
            logger.info(f"Intent: {intent} (confidence: {confidence:.2f})")
            
            # Route to appropriate handler
            response = None
            if intent == "sql_query":
                response = await self.handle_sql_query(message, intent, user_context)
            elif intent == "visual_update":
                response = await self.handle_visual_update(message, intent)
            elif intent == "mixed":
                response = await self.handle_mixed_query(message, intent, user_context)
            elif intent == "rag_lookup":
                response = await self.handle_rag_lookup(message, intent)
            else:  # needs_clarification
                response = await self.handle_clarification(message, intent)
            
            # Add confidence to response
            response.confidence = confidence
            
            # Store in conversation history
            if user_context and user_context.user_id:
                session_manager.add_to_history(
                    user_context.user_id,
                    message,
                    response.response,
                    intent
                )
            
            # Add session ID if provided
            if session_id:
                response.session_id = session_id
            
            return response
                
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}", exc_info=True)
            
            # Return error response
            error_response = await self.response_generator.generate_response(
                "needs_clarification",
                message,
                error=str(e)
            )
            
            return ChatResponse(
                intent="needs_clarification",
                response=error_response["response"],
                suggestions=error_response["suggestions"],
                confidence=0.0
            )
    
    async def handle_sql_query(
        self, 
        message: str, 
        intent: str,
        user_context: Optional[UserContext] = None
    ) -> ChatResponse:
        """Handle SQL query intent"""
        try:
            # Match query pattern
            query_key = self.match_query_pattern(message)
            
            if not query_key:
                # Try to generate a dynamic query based on the message
                query_key = "executive_dashboard_kpis"  # Default fallback
            
            logger.info(f"Matched query pattern: {query_key}")
            
            # Get the SQL query
            query_templates = self.finance_queries.get_query_templates()
            query = query_templates.get(query_key)
            
            if not query:
                raise ValueError(f"Query template '{query_key}' not found")
            
            # NEW: parse filters from the user’s message
            filters = self.parse_filter_request(message)

            # NEW: inject time WHERE conditions + collect params
            query, params = self._inject_time_filters(query, filters)

            # NEW: honor Top-N if requested
            query = self._apply_topn(query, filters)
                
            # Execute query with user context for RLS
            data = await self.db_manager.execute_query(query, params=params, user_context=user_context)
            
            logger.info(f"Query returned {len(data)} rows")
            
            # Generate natural language response
            response_data = await self.response_generator.generate_response(
                intent, 
                message, 
                data,
                context=f"Query type: {query_key.replace('_', ' ').title()}"
            )
            
            chat_response = ChatResponse(
                intent=intent,
                response=response_data["response"],
                suggestions=response_data["suggestions"],
                # data={
                #     "records": data.head(50).to_dict('records'),  # Limit data sent to frontend
                #     "total_count": len(data),
                #     "query_type": query_key
                # } if not data.empty else None,
                # sql_query=query if not config.ENABLE_AUTH else None,  # Hide query in production
                confidence=0.95
            )

            if response_data.get("show_table", False):
                chat_response.show_table = True
                chat_response.table_data = response_data.get("table_data", [])
                chat_response.table_title = response_data.get("table_title", "Results")
            else:
             # Fallback: include data in the old format for backward compatibility
                chat_response.data = {
                    "records": data.head(50).to_dict('records'),
                    "total_count": len(data),
                    "query_type": query_key
                } if not data.empty else None
        
            if not config.ENABLE_AUTH:
                chat_response.sql_query = query
            
            return chat_response
            
        except Exception as e:
            logger.error(f"SQL query handling error: {str(e)}", exc_info=True)
            
            error_response = await self.response_generator.generate_response(
                intent,
                message,
                error=f"Database query failed: {str(e)}"
            )
            
            return ChatResponse(
                intent=intent,
                response=error_response["response"],
                suggestions=error_response["suggestions"],
                confidence=0.5
            )

    async def handle_visual_update(self, message: str, intent: str) -> ChatResponse:
        """Handle visual update intent"""
        # Parse the filter request
        filters = self.parse_filter_request(message)
        
        response_data = await self.response_generator.generate_response(
            intent, 
            message, 
            context=f"Dashboard filter request: {filters}"
        )
        
        # For production with Power BI Embedded SDK
        embed_config = {
            "workspace_id": config.POWER_BI_WORKSPACE_ID,
            "report_id": config.POWER_BI_REPORT_ID,
            "page_name": config.POWER_BI_DEFAULT_PAGE,
            "filters": filters,
            "message": "Note: Automatic dashboard filtering is currently in development. Please use the Power BI filter pane to apply these filters manually."
        }
        
        return ChatResponse(
            intent=intent,
            response=f"I understand you want to filter the dashboard. {response_data['response']}\n\n" +
                    "**Note:** Automatic dashboard filtering is coming soon. For now, please use the Power BI filter options to apply your desired filters.",
            suggestions=response_data["suggestions"],
            data=embed_config,
            confidence=0.85
        )
    
    async def handle_mixed_query(
        self, 
        message: str, 
        intent: str,
        user_context: Optional[UserContext] = None
    ) -> ChatResponse:
        """Handle mixed intent (both SQL and visual update)"""
        # Execute SQL query part
        sql_response = await self.handle_sql_query(message, "sql_query", user_context)
        
        # Add visual update note
        sql_response.intent = intent
        sql_response.response += "\n\nFor the dashboard filtering aspect of your request, please use the Power BI filter pane to apply the relevant filters."
        
        return sql_response
    
    async def handle_rag_lookup(self, message: str, intent: str) -> ChatResponse:
        """Handle RAG lookup intent"""
        response_data = await self.response_generator.generate_response(
            intent,
            message,
            context="RAG/Document search functionality"
        )
        
        return ChatResponse(
            intent=intent,
            response="📄 **Document Search Coming Soon**\n\n" +
                    "The ability to search through reports, meeting notes, and other documents will be available in the next phase. " +
                    "Currently, I can help you with:\n" +
                    "• Real-time data analytics\n" +
                    "• Financial KPIs and metrics\n" +
                    "• Sales and revenue analysis\n" +
                    "• Customer and product insights\n\n" +
                    "What data analysis can I help you with today?",
            suggestions=[
                "Show executive dashboard",
                "Analyze monthly revenue trends",
                "View top customers",
                "Display product profitability"
            ],
            data={"rag_available": False, "coming_soon": True},
            confidence=0.9
        )
    
    async def handle_clarification(self, message: str, intent: str) -> ChatResponse:
        """Handle clarification needed intent"""
        
        # Provide helpful guidance
        intro_response = """I'm your Finance Copilot assistant. I can help you with:

📊 Data Analytics
• Top customers, products, and categories by revenue
• Profitability analysis and margins
• Sales trends and comparisons

📈 Time-Based Analysis
• Monthly, quarterly, and yearly performance
• Growth rates and trend analysis
• Period-over-period comparisons

👥 Customer Insights
• Customer segmentation and tiers
• Retention and lifetime value metrics
• Geographic performance analysis

💰 Financial KPIs
• Executive dashboard metrics
• Revenue and profit analysis
• Operational efficiency indicators

How can I help you analyze your financial data today?"""
        
        return ChatResponse(
            intent=intent,
            response=intro_response,
            suggestions=[
                "Show executive dashboard KPIs",
                "List top 10 customers by revenue",
                "Display monthly revenue trends",
                "Analyze product profitability"
            ],
            confidence=0.7
        )
    
    def match_query_pattern(self, message: str) -> Optional[str]:
        """Enhanced pattern matching for query selection"""
        message_lower = message.lower()
        
        # Comprehensive keyword patterns
        patterns = {
            # Executive/Overview queries - HIGHEST PRIORITY
            
            # Top/Ranking queries
            "top_products_by_revenue": [
                "top product", "best product", "product revenue", "product ranking",
                "highest selling product", "product performance", "product sales",
                "products by revenue", "leading products"
            ],
            "top_customers_by_value": [
                "top customer", "best customer", "customer revenue", "valuable customer",
                "customer ranking", "vip customer", "key account", "customer value",
                "customers by revenue", "list customers", "10 customers"
            ],
            "top_categories_by_performance": [
                "top categor", "best categor", "category performance", "category revenue",
                "categories", "product categories"
            ],
            
            # Profitability queries
            "product_profitability_analysis": [
                "product profit", "product margin", "profitable product",
                "product profitability", "analyze product profitability",
                "profitability analysis"
            ],
            "customer_profitability_ranking": [
                "customer profit", "customer margin", "profitable customer",
                "customer profitability", "customer roi"
            ],
            
            # Time-based queries
            "monthly_revenue_trends": [
                "monthly revenue", "monthly trend", "monthly sales",
                "revenue by month", "monthly performance", "month over month",
                "monthly analysis", "display monthly", "show monthly"
            ],
            "quarterly_performance": [
                "quarter", "quarterly", "q1", "q2", "q3", "q4",
                "quarter over quarter", "qoq", "quarterly revenue"
            ],
            "yearly_comparison_analysis": [
                "yearly", "annual", "year", "yoy", "year over year",
                "yearly comparison", "annual performance", "yearly revenue"
            ],

            "executive_dashboard_kpis": [
                "executive dashboard", "dashboard kpi", "executive kpi", "kpi", "dashboard", 
                "overview", "summary", "executive", "key metrics", "business metrics", 
                "performance overview", "main metrics", "show executive", "executive summary"
            ],
            
            # Customer analytics
            "customer_segmentation_analysis": [
                "segment", "customer segment", "customer tier", "customer group",
                "segmentation", "customer classification"
            ],
            "customer_retention_metrics": [
                "retention", "customer retention", "loyalty", "repeat customer",
                "returning customer", "churn"
            ]
        }
        
        # Check for executive dashboard first (highest priority)
        if any(keyword in message_lower for keyword in ["executive", "dashboard", "kpi"]):
            return "executive_dashboard_kpis"
        
        # Score each pattern based on keyword matches
        best_match = None
        best_score = 0
        
        for query_key, keywords in patterns.items():
            score = sum(1 for keyword in keywords if keyword in message_lower)
            if score > best_score:
                best_score = score
                best_match = query_key
        
        # If no good match, try to infer from common terms
        if best_score == 0:
            if "customer" in message_lower and ("top" in message_lower or "list" in message_lower):
                return "top_customers_by_value"
            elif "product" in message_lower:
                if "profit" in message_lower:
                    return "product_profitability_analysis"
                else:
                    return "top_products_by_revenue"
            elif any(term in message_lower for term in ["revenue", "sales", "money", "earnings"]):
                if "month" in message_lower:
                    return "monthly_revenue_trends"
                elif "year" in message_lower:
                    return "yearly_comparison_analysis"
                elif "quarter" in message_lower:
                    return "quarterly_performance"
                else:
                    return "executive_dashboard_kpis"
        
        return best_match if best_score > 0 else "executive_dashboard_kpis"
    
    # def parse_filter_request(self, message: str) -> Dict[str, Any]:
    #     """Parse filter request from natural language"""
    #     filters = {}
    #     message_lower = message.lower()
        
    #     # Year detection
    #     import re
    #     year_match = re.search(r'\b(20\d{2})\b', message)
    #     if year_match:
    #         filters["year"] = int(year_match.group(1))
        
    #     # Quarter detection
    #     quarter_match = re.search(r'\b(q[1-4])\b', message_lower)
    #     if quarter_match:
    #         filters["quarter"] = quarter_match.group(1).upper()
        
    #     # Month detection
    #     months = ["january", "february", "march", "april", "may", "june",
    #              "july", "august", "september", "october", "november", "december"]
    #     for i, month in enumerate(months):
    #         if month in message_lower:
    #             filters["month"] = i + 1
    #             filters["month_name"] = month.capitalize()
    #             break
        
    #     # Region/Country detection
    #     regions = ["us", "usa", "europe", "asia", "americas", "emea", "apac"]
    #     for region in regions:
    #         if region in message_lower:
    #             filters["region"] = region.upper()
    #             break
        
    #     # Category detection
    #     if "category" in message_lower or "categories" in message_lower:
    #         filters["category_filter"] = True
        
    #     return filters

    def parse_filter_request(self, message: str) -> Dict[str, Any]:
        """
        Parse filter request from natural language: year, quarter, month, last-N ranges, top-N.
        """
        import re
        from datetime import datetime, timedelta
        filters: Dict[str, Any] = {}
        msg = message.lower().strip()

        # Year (e.g., 2008, 2024)
        m = re.search(r'\b(19|20)\d{2}\b', msg)
        if m:
            filters["year"] = int(m.group(0))

        # Quarter (Q1..Q4)
        m = re.search(r'\bq([1-4])\b', msg)
        if m:
            filters["quarter"] = int(m.group(1))

        # Month name/abbr (Jan, January, Sept, Sep)
        month_map = {
            "jan":1,"january":1,"feb":2,"february":2,"mar":3,"march":3,"apr":4,"april":4,
            "may":5,"jun":6,"june":6,"jul":7,"july":7,"aug":8,"august":8,"sep":9,"sept":9,"september":9,
            "oct":10,"october":10,"nov":11,"november":11,"dec":12,"december":12
        }
        for token, mnum in month_map.items():
            if re.search(rf'\b{token}\b', msg):
                filters["month"] = mnum
                break

        # Last N period (days/weeks/months/quarters/years)
        m = re.search(r'\blast\s+(\d+)\s*(day|days|week|weeks|month|months|quarter|quarters|year|years)\b', msg)
        if m:
            n = int(m.group(1))
            unit = m.group(2)
            now = datetime.utcnow()
            if "day" in unit:
                start = now - timedelta(days=n)
            elif "week" in unit:
                start = now - timedelta(weeks=n)
            elif "month" in unit:
                # approx 30d per month for dev simplicity
                start = now - timedelta(days=30*n)
            elif "quarter" in unit:
                start = now - timedelta(days=90*n)
            else:
                start = now - timedelta(days=365*n)
            filters["date_from"] = start.date().isoformat()
            filters["date_to"] = now.date().isoformat()

        # Top-N (top 5, top-10...)
        m = re.search(r'\btop[-\s]?(\d{1,3})\b', msg)
        if m:
            n = int(m.group(1))
            if 1 <= n <= 100:
                filters["top_n"] = n

        return filters
    
    def _detect_orderdate_column(self, query: str) -> str:
        """Best-effort detection of the OrderDate column with alias (soh.OrderDate, h.OrderDate, OrderDate)."""
        import re
        m = re.search(r'([A-Za-z_][A-Za-z0-9_]*)\.OrderDate', query)
        if m:
            return f"{m.group(1)}.OrderDate"
        return "OrderDate"
    
    def _replace_time_anchor(self, query: str, where_snippet: str) -> str:
        """
        Replaces /*TIME_FILTERS*/ with the provided WHERE-AND snippet.
        The anchor should sit inside a WHERE clause (e.g., WHERE 1=1 /*TIME_FILTERS*/).
        """
        return query.replace("/*TIME_FILTERS*/", f" {where_snippet} ")

    def _inject_time_filters(self, query: str, filters: dict) -> tuple[str, list]:
        """
        Prefer replacing /*TIME_FILTERS*/ inside CTE/base tables.
        Fallback: append WHERE/AND in simple SELECT queries that expose OrderDate in outer scope.
        Returns (query, params_list).
        """
        import re
        conditions = []
        params = []

        # Build conditions in a fixed order: date range → year → quarter → month
        # Use the alias 'soh' consistently when the anchor path is used.
        # Fallback path will detect the date column alias from the query text.
        # 1) Explicit range
        if "date_from" in filters and "date_to" in filters:
            conditions.append("AND soh.OrderDate BETWEEN ? AND ?")
            params.extend([filters["date_from"], filters["date_to"]])

        # 2) Year / Quarter / Month
        if filters.get("year"):
            conditions.append("AND YEAR(soh.OrderDate) = ?")
            params.append(int(filters["year"]))
        if filters.get("quarter"):
            conditions.append("AND DATEPART(quarter, soh.OrderDate) = ?")
            params.append(int(filters["quarter"]))
        if filters.get("month"):
            conditions.append("AND MONTH(soh.OrderDate) = ?")
            params.append(int(filters["month"]))

        # If no filters, return unchanged
        if not conditions:
            return query, []

        # --- Path A: Anchor replacement ---
        if "/*TIME_FILTERS*/" in query:
            new_query = self._replace_time_anchor(query, " ".join(conditions))
            return new_query, params

        # --- Path B: Fallback injection for simple queries ---
        # Guard: if query looks like a CTE and no anchor, skip injection to avoid syntax errors.
        if re.search(r'^\s*WITH\s+', query, flags=re.IGNORECASE | re.DOTALL):
            # No safe place to inject; return unchanged (let caller proceed all-time or handle message).
            return query, []

        # Detect date column alias (best-effort)
        date_col = self._detect_orderdate_column(query)  # you already have this helper; keep it.
        fb_conditions = []
        fb_params = []

        # Rebuild conditions using detected date column
        if "date_from" in filters and "date_to" in filters:
            fb_conditions.append(f"{date_col} BETWEEN ? AND ?")
            fb_params.extend([filters["date_from"], filters["date_to"]])
        if filters.get("year"):
            fb_conditions.append(f"YEAR({date_col}) = ?")
            fb_params.append(int(filters["year"]))
        if filters.get("quarter"):
            fb_conditions.append(f"DATEPART(quarter, {date_col}) = ?")
            fb_params.append(int(filters["quarter"]))
        if filters.get("month"):
            fb_conditions.append(f"MONTH({date_col}) = ?")
            fb_params.append(int(filters["month"]))

        if not fb_conditions:
            return query, []

        # Insert before GROUP BY / ORDER BY / HAVING if present; else append WHERE/AND
        cut = len(query)
        for kw in ["\nGROUP BY", "\nORDER BY", "\nHAVING"]:
            m = re.search(kw, query, flags=re.IGNORECASE)
            if m:
                cut = min(cut, m.start())
        head, tail = query[:cut], query[cut:]

        if re.search(r'\bWHERE\b', head, flags=re.IGNORECASE):
            head = head + "\n  AND " + " AND ".join(fb_conditions)
        else:
            head = head + "\nWHERE " + " AND ".join(fb_conditions)

        return head + tail, fb_params

    def _apply_topn(self, query: str, filters: dict) -> str:
        """
        Honor top_n if present. Replace existing TOP number or insert TOP n after SELECT
        only when an ORDER BY exists (to preserve meaning). Avoids regex backref issues.
        """
        import re
        n = filters.get("top_n")
        if not n:
            return query

        # 1) Replace an existing TOP <num>
        q2 = re.sub(r'(?i)^\s*SELECT\s+TOP\s+\d+\s+', f"SELECT TOP {n} ", query, count=1)
        if q2 != query:
            return q2

        # 2) Insert TOP n if ORDER BY exists and no TOP already
        if re.search(r'(?i)\bORDER\s+BY\b', query):
            q3 = re.sub(r'(?i)^\s*SELECT\s+(?!TOP\s+\d+\s+)', f"SELECT TOP {n} ", query, count=1)
            return q3

        # 3) Otherwise leave it (caller can slice rows in Python if needed)
        return query


# ===============================
# DATABASE & QUERY INITIALIZATION
# ===============================

db_manager = DatabaseManager()
query_processor = QueryProcessor(db_manager)

# ===============================
# AUTHENTICATION HELPERS
# ===============================

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> UserContext:
    """Verify Azure AD token and extract user context"""
    
    # For development without auth
    if not config.ENABLE_AUTH:
        return UserContext(
            user_id="dev_user",
            name="Development User",
            email="dev@example.com",
            roles=["admin"],
            permissions={"all": True}
        )
    
    # Production: Verify Azure AD token
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        # TODO: Implement actual Azure AD token verification
        # This would involve:
        # 1. Decoding the JWT token
        # 2. Verifying the signature with Azure AD public keys
        # 3. Checking token expiration
        # 4. Extracting user claims
        
        # Placeholder for token verification
        token = credentials.credentials
        
        # In production, decode and verify the token properly
        # decoded_token = jwt.decode(token, options={"verify_signature": False})
        
        # For now, return mock user context
        return UserContext(
            user_id="authenticated_user",
            name="Authenticated User",
            email="user@company.com",
            roles=["user"],
            permissions={"region": "US"}
        )
        
    except Exception as e:
        logger.error(f"Token verification failed: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

# ===============================
# FASTAPI APPLICATION
# ===============================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 Starting Finance Copilot Backend...")
    
    # Test database connection
    db_connected = await db_manager.test_connection()
    if db_connected:
        logger.info("✅ Database connection successful")
    else:
        logger.error("❌ Database connection failed")
    
    # Test OpenAI connection
    if openai_client:
        logger.info("✅ Azure OpenAI client initialized")
    else:
        logger.warning("⚠️ Azure OpenAI client not initialized")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Finance Copilot Backend...")

# Create FastAPI app
app = FastAPI(
    title="Finance Copilot Backend",
    description="AI-powered financial analytics chatbot with Azure integration",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# ===============================
# CORS CONFIGURATION
# ===============================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        # Add your Azure Static Web App URL here
        # "https://your-app.azurestaticapps.net",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# API ENDPOINTS
# ===============================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Finance Copilot API",
        "version": "2.0.0",
        "status": "operational",
        "docs": "/api/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    db_status = await db_manager.test_connection()
    
    return {
        "status": "healthy" if db_status else "degraded",
        "services": {
            "database": "connected" if db_status else "disconnected",
            "openai": "connected" if openai_client else "disconnected",
            "auth": "enabled" if config.ENABLE_AUTH else "disabled"
        },
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0"
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    message: ChatMessage,
    user: UserContext = Depends(verify_token)
):
    """Main chat endpoint for processing user messages"""
    logger.info(f"Chat request from {user.user_id}: {message.message[:100]}...")
    
    # Create or get session
    if not message.session_id:
        message.session_id = session_manager.create_session(user.user_id)
    
    response = await query_processor.process_message(
        message.message,
        message.session_id,
        user
    )
    
    return response

@app.get("/api/history")
async def get_history(
    limit: int = 10,
    user: UserContext = Depends(verify_token)
):
    """Get conversation history for the current user"""
    history = session_manager.get_history(user.user_id, limit)
    
    return ConversationHistory(
        messages=history,
        total_count=len(history),
        session_id=session_manager.create_session(user.user_id)
    )

@app.post("/api/sql")
async def execute_sql(
    request: SQLQueryRequest,
    user: UserContext = Depends(verify_token)
):
    """Direct SQL query execution (admin only in production)"""
    
    # Check permissions
    if config.ENABLE_AUTH and "admin" not in user.roles:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    logger.warning(f"Direct SQL execution by {user.user_id}")
    
    try:
        data = await db_manager.execute_query(
            request.query, 
            request.parameters,
            user
        )
        
        return {
            "success": True,
            "data": {
                "records": data.head(100).to_dict('records'),
                "total_count": len(data)
            },
            "columns": list(data.columns) if not data.empty else []
        }
    except Exception as e:
        logger.error(f"SQL execution error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/queries")
async def get_available_queries(user: UserContext = Depends(verify_token)):
    """Get available predefined query templates"""
    templates = FinanceQueries.get_query_templates()
    
    return {
        "categories": {
            "Overview": ["executive_dashboard_kpis"],
            "Revenue Analysis": [
                "top_products_by_revenue",
                "top_customers_by_value", 
                "top_categories_by_performance",
                "monthly_revenue_trends",
                "quarterly_performance",
                "yearly_comparison_analysis"
            ],
            "Profitability": [
                "product_profitability_analysis",
                "customer_profitability_ranking"
            ],
            "Customer Analytics": [
                "customer_segmentation_analysis",
                "customer_retention_metrics"
            ]
        },
        "total_queries": len(templates)
    }

@app.get("/api/schema")
async def get_database_schema(
    table: Optional[str] = None,
    user: UserContext = Depends(verify_token)
):
    """Get database schema information"""
    
    # Check permissions
    if config.ENABLE_AUTH and "admin" not in user.roles:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    schema = await db_manager.get_table_schema(table)
    
    return {
        "schema": schema.to_dict('records'),
        "tables": schema['TABLE_NAME'].unique().tolist() if not table else [table]
    }

@app.get("/api/powerbi/config")
async def get_powerbi_config(user: UserContext = Depends(verify_token)):
    """Get Power BI configuration for embedding"""
    
    # For production with Power BI Embedded
    # This would include token generation and embed URLs
    
    return {
        "embed_type": "public",  # Change to "secure" for production
        "workspace_id": config.POWER_BI_WORKSPACE_ID,
        "report_id": config.POWER_BI_REPORT_ID,
        "default_page": config.POWER_BI_DEFAULT_PAGE,
        "embed_url": f"https://app.powerbi.com/reportEmbed?reportId={config.POWER_BI_REPORT_ID}&autoAuth=true&ctid={config.AZURE_TENANT_ID}",
        
        # For production with Power BI Embedded SDK
        # "access_token": "GENERATE_POWERBI_EMBED_TOKEN_HERE",
        # "token_type": "Embed",
        # "token_expiry": "TOKEN_EXPIRY_TIME",
        
        "note": "Using public embed for prototype. Production will use secure embedding with Azure AD authentication."
    }

# ===============================
# ERROR HANDLERS
# ===============================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return {
        "error": "An unexpected error occurred",
        "detail": str(exc) if not config.ENABLE_AUTH else None,
        "status_code": 500,
        "timestamp": datetime.utcnow().isoformat()
    }

# ===============================
# MAIN ENTRY POINT
# ===============================

if __name__ == "__main__":
    import uvicorn
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Set to False in production
        log_level=config.LOG_LEVEL.lower()
    )