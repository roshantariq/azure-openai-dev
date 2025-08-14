# quick_test.py - Quick manual test for Finance Copilot
"""
Simple test script to verify the backend is working correctly.
Run: python quick_test.py
"""

import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"

def test_query(query):
    """Test a single query and display results"""
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print('-'*60)
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/chat",
            json={"message": query},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✓ Intent: {data.get('intent')}")
            print(f"✓ Confidence: {data.get('confidence', 0):.2f}")
            
            if data.get('data'):
                record_count = data['data'].get('total_count', 0)
                print(f"✓ Records Retrieved: {record_count}")
                
                # Show sample data if available
                if data['data'].get('records') and len(data['data']['records']) > 0:
                    print(f"\nFirst 3 records:")
                    for i, record in enumerate(data['data']['records'][:3], 1):
                        print(f"  {i}. {json.dumps(record, indent=2)[:200]}...")
            
            # Show response preview
            response_text = data.get('response', '')
            print(f"\nResponse Preview:")
            print(response_text[:500] + "..." if len(response_text) > 500 else response_text)
            
            return True
        else:
            print(f"✗ Error: HTTP {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False

def main():
    print("Finance Copilot Backend Test")
    print("="*60)
    
    # Check health
    print("\n1. Checking backend health...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            health = response.json()
            print(f"✓ Backend is {health['status']}")
            print(f"  - Database: {health['services']['database']}")
            print(f"  - OpenAI: {health['services']['openai']}")
        else:
            print("✗ Backend health check failed")
            return
    except Exception as e:
        print(f"✗ Cannot connect to backend: {str(e)}")
        print("Make sure the backend is running on http://localhost:8000")
        return
    
    # Test queries
    test_queries = [
        # Basic queries
        "Show executive dashboard KPIs",
        "List top 10 customers by revenue",
        
        # Time-filtered queries
        "Show total sales in 2008",
        "What was the revenue in Q1 2008?",
        "Show sales in January 2008",
        "Display last 30 days revenue",
        
        # Top-N queries
        "Show top 5 products by revenue",
        "List top 3 customers in 2008",
        
        # Complex queries
        "Show top 5 customers by revenue in Q1 2008",
        "Compare Q1 and Q2 2008 revenue",
    ]
    
    print("\n2. Testing queries...")
    success_count = 0
    
    for query in test_queries:
        if test_query(query):
            success_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Test Summary:")
    print(f"  Total: {len(test_queries)}")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {len(test_queries) - success_count}")
    print(f"  Success Rate: {(success_count/len(test_queries)*100):.1f}%")
    
    if success_count == len(test_queries):
        print("\n✓ All tests passed! Backend is working correctly.")
    elif success_count > len(test_queries) * 0.8:
        print("\n⚠ Most tests passed, but some issues detected.")
    else:
        print("\n✗ Many tests failed. Check the backend logs.")

if __name__ == "__main__":
    main()