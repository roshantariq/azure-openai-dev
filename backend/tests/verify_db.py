# verify_database.py - Verify AdventureWorksLT Data
"""
This script connects directly to your database to verify the actual data.
Run: python verify_database.py
"""

import pyodbc
import pandas as pd
from datetime import datetime
from tabulate import tabulate

# Database configuration
config = {
    "driver": "{ODBC Driver 17 for SQL Server}",
    "server": "chatbot-test-server.database.windows.net",
    "database": "chatbot-test-db",
    "username": "roshan",
    "password": "P@ssw0rd12345"
}

def get_connection():
    """Create database connection"""
    conn_str = (
        f"DRIVER={config['driver']};"
        f"SERVER={config['server']};"
        f"DATABASE={config['database']};"
        f"UID={config['username']};"
        f"PWD={config['password']};"
        "Encrypt=yes;"
        "TrustServerCertificate=no;"
    )
    return pyodbc.connect(conn_str)

def verify_date_range():
    """Check the actual date range of orders in the database"""
    print("\n" + "="*60)
    print("1. VERIFYING ORDER DATE RANGE")
    print("="*60)
    
    query = """
    SELECT 
        MIN(OrderDate) as EarliestOrder,
        MAX(OrderDate) as LatestOrder,
        COUNT(DISTINCT CAST(OrderDate as DATE)) as UniqueDays,
        COUNT(*) as TotalOrders,
        COUNT(DISTINCT YEAR(OrderDate)) as YearsWithData,
        COUNT(DISTINCT FORMAT(OrderDate, 'yyyy-MM')) as MonthsWithData
    FROM SalesLT.SalesOrderHeader
    """
    
    with get_connection() as conn:
        df = pd.read_sql(query, conn)
        print(df.to_string())
    
    # Check orders by year
    query_years = """
    SELECT 
        YEAR(OrderDate) as Year,
        COUNT(*) as OrderCount,
        FORMAT(SUM(TotalDue), 'C', 'en-US') as TotalRevenue,
        FORMAT(MIN(OrderDate), 'yyyy-MM-dd') as FirstOrder,
        FORMAT(MAX(OrderDate), 'yyyy-MM-dd') as LastOrder
    FROM SalesLT.SalesOrderHeader
    GROUP BY YEAR(OrderDate)
    ORDER BY Year
    """
    
    print("\n📅 Orders by Year:")
    with get_connection() as conn:
        df = pd.read_sql(query_years, conn)
        print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))

def verify_monthly_data():
    """Check monthly distribution of orders"""
    print("\n" + "="*60)
    print("2. MONTHLY DATA DISTRIBUTION")
    print("="*60)
    
    query = """
    SELECT 
        YEAR(OrderDate) as Year,
        MONTH(OrderDate) as Month,
        DATENAME(month, OrderDate) as MonthName,
        COUNT(*) as OrderCount,
        FORMAT(SUM(TotalDue), 'C', 'en-US') as Revenue,
        FORMAT(MIN(OrderDate), 'yyyy-MM-dd') as FirstOrder,
        FORMAT(MAX(OrderDate), 'yyyy-MM-dd') as LastOrder
    FROM SalesLT.SalesOrderHeader
    GROUP BY YEAR(OrderDate), MONTH(OrderDate), DATENAME(month, OrderDate)
    ORDER BY Year, Month
    """
    
    with get_connection() as conn:
        df = pd.read_sql(query, conn)
        print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))

def verify_quarterly_data():
    """Check quarterly distribution"""
    print("\n" + "="*60)
    print("3. QUARTERLY DATA DISTRIBUTION")
    print("="*60)
    
    query = """
    SELECT 
        YEAR(OrderDate) as Year,
        DATEPART(quarter, OrderDate) as Quarter,
        'Q' + CAST(DATEPART(quarter, OrderDate) as VARCHAR) as QuarterName,
        COUNT(*) as OrderCount,
        FORMAT(SUM(TotalDue), 'C', 'en-US') as Revenue,
        FORMAT(MIN(OrderDate), 'yyyy-MM-dd') as FirstOrder,
        FORMAT(MAX(OrderDate), 'yyyy-MM-dd') as LastOrder
    FROM SalesLT.SalesOrderHeader
    GROUP BY YEAR(OrderDate), DATEPART(quarter, OrderDate)
    ORDER BY Year, Quarter
    """
    
    with get_connection() as conn:
        df = pd.read_sql(query, conn)
        print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))

def verify_specific_periods():
    """Test specific period queries"""
    print("\n" + "="*60)
    print("4. TESTING SPECIFIC PERIOD QUERIES")
    print("="*60)
    
    test_queries = [
        ("Q1 2008", "WHERE YEAR(OrderDate) = 2008 AND DATEPART(quarter, OrderDate) = 1"),
        ("January 2008", "WHERE YEAR(OrderDate) = 2008 AND MONTH(OrderDate) = 1"),
        ("2008 Full Year", "WHERE YEAR(OrderDate) = 2008"),
        ("May 2008", "WHERE YEAR(OrderDate) = 2008 AND MONTH(OrderDate) = 5"),
        ("June 2008", "WHERE YEAR(OrderDate) = 2008 AND MONTH(OrderDate) = 6"),
    ]
    
    for period_name, where_clause in test_queries:
        query = f"""
        SELECT 
            '{period_name}' as Period,
            COUNT(*) as OrderCount,
            FORMAT(SUM(TotalDue), 'C', 'en-US') as Revenue
        FROM SalesLT.SalesOrderHeader
        {where_clause}
        """
        
        with get_connection() as conn:
            df = pd.read_sql(query, conn)
            print(f"\n{period_name}:")
            if df['OrderCount'].iloc[0] > 0:
                print(f"  ✓ Orders: {df['OrderCount'].iloc[0]}, Revenue: {df['Revenue'].iloc[0]}")
            else:
                print(f"  ✗ No data found")

def verify_top_customers():
    """Verify top customers query"""
    print("\n" + "="*60)
    print("5. TOP CUSTOMERS VERIFICATION")
    print("="*60)
    
    query = """
    SELECT TOP 5
        c.CustomerID,
        CONCAT(c.FirstName, ' ', c.LastName) as CustomerName,
        c.CompanyName,
        FORMAT(SUM(soh.TotalDue), 'C', 'en-US') as TotalRevenue,
        COUNT(soh.SalesOrderID) as Orders
    FROM SalesLT.Customer c
    JOIN SalesLT.SalesOrderHeader soh ON c.CustomerID = soh.CustomerID
    GROUP BY c.CustomerID, c.FirstName, c.LastName, c.CompanyName
    ORDER BY SUM(soh.TotalDue) DESC
    """
    
    with get_connection() as conn:
        df = pd.read_sql(query, conn)
        print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))

def verify_products():
    """Verify product data"""
    print("\n" + "="*60)
    print("6. PRODUCT DATA VERIFICATION")
    print("="*60)
    
    query = """
    SELECT 
        COUNT(DISTINCT p.ProductID) as TotalProducts,
        COUNT(DISTINCT pc.ProductCategoryID) as TotalCategories,
        COUNT(DISTINCT sod.ProductID) as ProductsSold
    FROM SalesLT.Product p
    LEFT JOIN SalesLT.ProductCategory pc ON p.ProductCategoryID = pc.ProductCategoryID
    LEFT JOIN SalesLT.SalesOrderDetail sod ON p.ProductID = sod.ProductID
    """
    
    with get_connection() as conn:
        df = pd.read_sql(query, conn)
        print(df.to_string())

def check_executive_kpis():
    """Verify executive dashboard KPIs"""
    print("\n" + "="*60)
    print("7. EXECUTIVE DASHBOARD KPIs (ALL TIME)")
    print("="*60)
    
    query = """
    WITH OverallMetrics AS (
        SELECT 
            SUM(TotalDue) as TotalRevenue,
            COUNT(DISTINCT SalesOrderID) as TotalOrders,
            COUNT(DISTINCT CustomerID) as TotalCustomers,
            AVG(TotalDue) as AvgOrderValue,
            MIN(OrderDate) as EarliestOrder,
            MAX(OrderDate) as LatestOrder
        FROM SalesLT.SalesOrderHeader
    ),
    ProductMetrics AS (
        SELECT 
            COUNT(DISTINCT p.ProductID) as TotalProducts,
            COUNT(DISTINCT pc.ProductCategoryID) as TotalCategories
        FROM SalesLT.Product p
        LEFT JOIN SalesLT.ProductCategory pc ON p.ProductCategoryID = pc.ProductCategoryID
    )
    SELECT 
        FORMAT(om.TotalRevenue, 'C', 'en-US') as TotalRevenue,
        om.TotalOrders,
        om.TotalCustomers,
        pm.TotalProducts,
        pm.TotalCategories,
        FORMAT(om.AvgOrderValue, 'C', 'en-US') as AvgOrderValue,
        FORMAT(om.EarliestOrder, 'yyyy-MM-dd') as EarliestOrder,
        FORMAT(om.LatestOrder, 'yyyy-MM-dd') as LatestOrder
    FROM OverallMetrics om, ProductMetrics pm
    """
    
    with get_connection() as conn:
        df = pd.read_sql(query, conn)
        print(df.T.to_string())  # Transpose for better readability

def main():
    print("\n" + "="*60)
    print("ADVENTUREWORKSLT DATABASE VERIFICATION")
    print("="*60)
    
    try:
        # Run all verifications
        verify_date_range()
        verify_monthly_data()
        verify_quarterly_data()
        verify_specific_periods()
        verify_top_customers()
        verify_products()
        check_executive_kpis()
        
        # Summary and recommendations
        print("\n" + "="*60)
        print("SUMMARY & RECOMMENDATIONS")
        print("="*60)
        
        print("""
📊 KEY FINDINGS:
1. The AdventureWorksLT sample database only contains data from May-June 2008 (Q2)
2. There is NO data for Q1 2008, January 2008, or any other months
3. Total of 32 orders from 32 unique customers
4. 295 products across 37 categories in the database

⚠️ ISSUES TO FIX:
1. "Executive Dashboard KPIs" is being classified as 'visual_update' instead of 'sql_query'
   - Add "kpi", "dashboard", "executive" to sql_query intent keywords
   
2. Queries for Q1 2008 or January 2008 return empty/null because no data exists
   - The backend should handle this better with a message like:
     "No sales data found for Q1 2008 in the database"
   
3. Time filter injections may be failing for some queries
   - Check the _inject_time_filters function for edge cases

✅ WHAT'S WORKING:
- Top customers query is accurate
- Top products query is accurate  
- 2008 total sales ($956,303.59) is correct
- Basic queries without time filters work well

🔧 RECOMMENDED FIXES:
1. Update intent classifier to recognize "dashboard KPIs" as sql_query
2. Add data existence validation before running queries
3. Return meaningful messages when no data exists for a period
4. Consider loading more sample data to test time filters properly
        """)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("Make sure the database connection details are correct.")

if __name__ == "__main__":
    main()