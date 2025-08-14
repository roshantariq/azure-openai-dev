// src/App.tsx - Finance Copilot TypeScript Version
import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './App.css';
import { POWERBI_CONFIG } from './config';
// use POWERBI_CONFIG.workspaceId / reportId / defaultPage


// Types
interface Message {
  id: number;
  content: string | 
  { 
    response: string; 
    data?: any;
    show_table?: boolean;
    table_data?: any[];
    table_title?: string;
  };
  isUser: boolean;
  intent?: string;
  suggestions?: string[];
}

interface User {
  displayName: string;
  mail: string;
  id: string;
}

// REPLACE THESE PLACEHOLDER VALUES AFTER CREATING AZURE AD APP REGISTRATION
const AZURE_CONFIG = {
  clientId: 'YOUR_AZURE_CLIENT_ID_HERE', // TODO: REPLACE WITH ACTUAL CLIENT ID
  authority: 'https://login.microsoftonline.com/YOUR_TENANT_ID_HERE', // TODO: REPLACE WITH ACTUAL TENANT ID
  redirectUri: 'http://localhost:3000', // TODO: UPDATE FOR PRODUCTION
};

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Mock authentication service (replace with actual MSAL in production)
class MockAuthService {
  private isAuthenticated: boolean = false;
  private accessToken: string | null = null;
  private user: User | null = null;

  async login(): Promise<User> {
    // TODO: REPLACE WITH ACTUAL MSAL AUTHENTICATION
    // Simulate login for development
    return new Promise((resolve) => {
      setTimeout(() => {
        this.isAuthenticated = true;
        this.accessToken = 'mock-access-token-' + Date.now();
        this.user = {
          displayName: 'Development User',
          mail: 'dev.user@company.com',
          id: 'dev-user-123'
        };
        resolve(this.user!);
      }, 1000);
    });
  }

  async logout(): Promise<void> {
    this.isAuthenticated = false;
    this.accessToken = null;
    this.user = null;
  }

  getAccessToken(): string | null {
    return this.accessToken;
  }

  getUser(): User | null {
    return this.user;
  }

  isUserAuthenticated(): boolean {
    return this.isAuthenticated;
  }
}

// Initialize auth service
const authService = new MockAuthService();

// Configure axios with auth interceptor
axios.interceptors.request.use((config) => {
  const token = authService.getAccessToken();
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// ---------- helpers ----------
const isNumericLike = (v: unknown) => {
  if (v === null || v === undefined) return false;
  if (typeof v === 'number') return true;
  if (typeof v !== 'string') return false;
  // allow "1,234.56", "$12.34"
  return /^[\$\s]*-?\d{1,3}(,\d{3})*(\.\d+)?$/.test(v) || /^-?\d+(\.\d+)?$/.test(v);
};

const formatCurrencyIfNeeded = (key: string, v: any) => {
  // display as-is if backend already formatted with "$"
  if (typeof v === 'string' && v.trim().startsWith('$')) return v;
  // heuristic: keys with revenue/amount/value/price
  const k = key.toLowerCase();
  const shouldFormat = ['revenue', 'amount', 'value', 'price', 'avgorder', 'avg_selling'].some(s => k.includes(s));
  if (!shouldFormat || v === null || v === undefined || v === '') return v ?? '';
  const num = typeof v === 'number' ? v : Number(String(v).replace(/[^\d.-]/g, ''));
  if (isNaN(num)) return v;
  return new Intl.NumberFormat(undefined, { style: 'currency', currency: 'USD', maximumFractionDigits: 2 }).format(num);
};

const toCSV = (rows: any[]) => {
  if (!rows.length) return '';
  const cols = Array.from(new Set(rows.flatMap(r => Object.keys(r))));
  const esc = (s: any) => {
    const str = s === null || s === undefined ? '' : String(s);
    return /[",\n]/.test(str) ? `"${str.replace(/"/g, '""')}"` : str;
  };
  const header = cols.map(esc).join(',');
  const body = rows.map(r => cols.map(c => esc(r[c])).join(',')).join('\n');
  return `${header}\n${body}`;
};

const downloadCSV = (rows: any[], filename = 'results.csv') => {
  const csv = toCSV(rows);
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
};

// ---------- ResultsTable ----------
interface ResultsTableProps {
  rows: any[];
  title?: string;
  maxRows?: number; // to avoid giant payloads; show top N then export for full set
}

const ResultsTable: React.FC<ResultsTableProps> = ({ rows, title, maxRows = 20 }) => {
  if (!rows || !rows.length) return null;

  // choose columns from first row (stable order), then add any extras found later
  const firstKeys = Object.keys(rows[0]);
  const extraKeys = Array.from(
    new Set(rows.slice(1).flatMap(r => Object.keys(r)).filter(k => !firstKeys.includes(k)))
  );
  const columns = [...firstKeys, ...extraKeys];

  // derive alignment: right-align numeric-ish
  const alignRight = (key: string) => rows.some(r => isNumericLike(r[key]));

  const limited = rows.slice(0, maxRows);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(toCSV(rows));
      alert('Copied table as CSV to clipboard.');
    } catch {
      alert('Copy failed. Try Download instead.');
    }
  };

  return (
    <div className="result-block">
      <div className="result-header">
        <div className="result-title">
          {title || 'Results'} <span className="muted">• {rows.length} rows</span>
        </div>
        <div className="result-actions">
          <button className="btn btn-secondary" onClick={() => downloadCSV(rows)}>Download CSV</button>
          <button className="btn" onClick={handleCopy}>Copy</button>
        </div>
      </div>

      <div className="table-wrap">
        <table className="result-table">
          <thead>
            <tr>
              {columns.map(col => (
                <th key={col} className={alignRight(col) ? 'ta-right' : ''}>{col}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {limited.map((r, idx) => (
              <tr key={idx}>
                {columns.map(col => {
                  const val = r[col];
                  const display = alignRight(col) ? formatCurrencyIfNeeded(col, val) : (val ?? '');
                  return (
                    <td key={col} className={alignRight(col) ? 'ta-right' : ''}>
                      {display}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {rows.length > maxRows && (
        <div className="result-footer muted">
          Showing first {maxRows} rows. Download CSV to see all.
        </div>
      )}
    </div>
  );
};

// Message component
const Message: React.FC<{
  message: string | 
  { 
    response: string; 
    data?: any;
    show_table?: boolean;
    table_data?: any[];
    table_title?: string;
  };
  isUser: boolean;
  intent?: string;
  suggestions?: string[];
  onSuggestionClick: (suggestion: string) => void;
}> = ({ message, isUser, intent, suggestions, onSuggestionClick }) => {
  const intentColors: { [key: string]: string } = {
    visual_update: 'bg-blue-100 text-blue-800',
    sql_query: 'bg-green-100 text-green-800',
    rag_lookup: 'bg-yellow-100 text-yellow-800',
    mixed: 'bg-purple-100 text-purple-800',
    needs_clarification: 'bg-red-100 text-red-800',
    system: 'bg-gray-100 text-gray-800',
  };

  return (
  <div className={`message ${isUser ? 'user-message' : 'bot-message'}`}>
    {!isUser && intent && (
      <div className={`intent-badge ${intentColors[intent] || 'bg-gray-100 text-gray-800'}`}>
        {intent.replace('_', ' ').toUpperCase()}
        {typeof message !== 'string' && message.table_data && (
            <span className="badge-meta"> • {message.table_data.length} rows</span>
          )}
      </div>
    )}
    <div className="message-content">
      {typeof message === 'string' ? (
        <div style={{ whiteSpace: 'pre-line' }}>{message}</div>
      ) : (
        <div>
          {/* Short narrative first (kept concise) */}
          {message.response && <div style={{ whiteSpace: 'pre-line' }}>{message.response}</div>}

          {/* If the backend returns tabular data, render it nicely */}
          {message.show_table && message.table_data && message.table_data.length > 0 && (
            <ResultsTable
              // rows={message.data}
              // title={
              //   intent === 'sql_query' ? 'Query Results' :
              //   intent === 'visual_update' ? 'Applied Filters' :
              //   'Results'
              // }
              rows={message.table_data}
              title={message.table_title || 'Results'}
            />
          )}

          {/* Fallback: show data if available but not in table format */}
            {!message.show_table && Array.isArray(message.data) && message.data.length > 0 && (
              <ResultsTable
                rows={message.data}
                title={
                  intent === 'sql_query' ? 'Query Results' :
                  intent === 'visual_update' ? 'Applied Filters' :
                  'Results'
                }
              />
            )}

          {/* If non-array data (e.g., status), keep a small preview */}
          {!Array.isArray(message.data) && message.data && !message.show_table && (
            <div className="data-preview">
              <small className="text-gray-600">Data updated</small>
            </div>
          )}
        </div>
      )}

      {suggestions && suggestions.length > 0 && (
        <div className="suggestions">
          {suggestions.map((suggestion, index) => (
            <button
              key={index}
              className="suggestion-btn"
              onClick={() => onSuggestionClick(suggestion)}
            >
              {suggestion}
            </button>
          ))}
        </div>
      )}
    </div>
  </div>
);
};

// Power BI Embed Component
const PowerBIEmbed: React.FC<{ isAuthenticated: boolean }> = ({ isAuthenticated }) => {
  const [embedLoaded, setEmbedLoaded] = useState(false);

  useEffect(() => {
    if (isAuthenticated && !embedLoaded) {
      // Simulate Power BI embed loading
      setTimeout(() => {
        setEmbedLoaded(true);
      }, 2000);
    }
  }, [isAuthenticated, embedLoaded]);

  if (!isAuthenticated) {
    return (
      <div className="powerbi-signin">
        <div className="signin-content">
          <div className="signin-icon">🔐</div>
          <h3>Sign in to view your dashboard</h3>
          <p>Please authenticate to access your Power BI reports</p>
        </div>
      </div>
    );
  }

  if (!embedLoaded) {
    return (
      <div className="powerbi-loading">
        <div className="loading-spinner"></div>
        <p>Loading Power BI Dashboard...</p>
      </div>
    );
  }

  // In production, replace this with actual Power BI embed
  const embedUrl = `https://app.powerbi.com/reportEmbed?reportId=${POWERBI_CONFIG.reportId}&groupId=${POWERBI_CONFIG.workspaceId}&w=2&config=eyJjbHVzdGVyVXJsIjoiaHR0cHM6Ly9XQUJJLVdFU1QtRVVST1BFLXJlZGlyZWN0LmFuYWx5c2lzLndpbmRvd3MubmV0IiwiZW1iZWRGZWF0dXJlcyI6eyJ1c2FnZU1ldHJpY3NWTmV4dCI6dHJ1ZX19`;

  return (
    <div className="powerbi-container">
      <iframe
        className="powerbi-frame"
        src={embedUrl}
        frameBorder="0"
        allowFullScreen={true}
        title="Power BI Dashboard"
      />
      <div className="powerbi-overlay">
        <div className="status-indicator">
          <span className="status-dot"></span>
          Dashboard Connected
        </div>
      </div>
    </div>
  );
};

// Loading component
const LoadingDots: React.FC = () => (
  <div className="loading-message">
    <div className="loading-content">
      <span>Analyzing your request</span>
      <div className="loading-dots">
        <span></span>
        <span></span>
        <span></span>
      </div>
    </div>
  </div>
);

// Main App Component
const App: React.FC = () => {
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(false);
  const [user, setUser] = useState<User | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isSigningIn, setIsSigningIn] = useState<boolean>(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom when new messages are added
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Initialize with welcome message
  useEffect(() => {
    const introResponse = `I'm your Finance Copilot assistant. I can help you with:

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

How can I help you analyze your financial data today?`;

    setMessages([{
      id: Date.now(),
      content: introResponse,
      isUser: false,
      intent: 'system',
      suggestions: [
        'Show executive dashboard KPIs',
        'List top 10 customers by revenue',
        'Display monthly revenue trends',
        'Analyze product profitability'
      ]
    }]);
  }, []);

  // Authentication functions
  const handleSignIn = async () => {
    setIsSigningIn(true);
    try {
      const user = await authService.login();
      setIsAuthenticated(true);
      setUser(user);
      
      // Add authentication success message
      setMessages(prev => [...prev, {
        id: Date.now(),
        content: `Welcome Development User! You're now connected to your Power BI workspace. Your dashboard is loading...`,
        isUser: false,
        intent: 'system'
      }]);
    } catch (error) {
      console.error('Sign-in failed:', error);
    } finally {
      setIsSigningIn(false);
    }
  };

  const handleSignOut = async () => {
    await authService.logout();
    setIsAuthenticated(false);
    setUser(null);
    setMessages(prev => [...prev, {
      id: Date.now(),
      content: "You've been signed out. Sign in again to access your dashboard and data.",
      isUser: false,
      intent: 'system'
    }]);
  };

  // Chat functions
  const sendMessage = async (messageText: string = inputMessage) => {
    if (!messageText.trim()) return;

    const userMessage: Message = {
      id: Date.now(),
      content: messageText,
      isUser: true
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await axios.post(`${API_BASE_URL}/api/chat`, {
        message: messageText,
        session_id: `session_${Date.now()}`
      });

      const botMessage: Message = {
        id: Date.now() + 1,
        content: {
          response: response.data.response,
          data: response.data.data,
          show_table: response.data.show_table,
          table_data: response.data.table_data,
          table_title: response.data.table_title
        },
        isUser: false,
        intent: response.data.intent,
        suggestions: response.data.suggestions
      };

      setMessages(prev => [...prev, botMessage]);

    } catch (error) {
      console.error('Chat error:', error);
      
      const errorMessage: Message = {
        id: Date.now() + 1,
        content: "I apologize, but I encountered an error processing your request. Please try again or rephrase your question.",
        isUser: false,
        intent: 'system',
        suggestions: ['Try again', 'Get help', 'Show examples']
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    setInputMessage(suggestion);
    sendMessage(suggestion);
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="header-content">
          <h1 className="app-title">
            <span className="title-icon">📊</span>
            Finance Copilot
          </h1>
          <div className="auth-section">
            {isAuthenticated ? (
              <div className="user-info">
                <span className="user-name">{user?.displayName}</span>
                <button className="auth-btn signout-btn" onClick={handleSignOut}>
                  Sign Out
                </button>
              </div>
            ) : (
              <button 
                className="auth-btn signin-btn" 
                onClick={handleSignIn}
                disabled={isSigningIn}
              >
                {isSigningIn ? 'Signing In...' : 'Sign In'}
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="main-content">
        {/* Dashboard Section */}
        <section className="dashboard-section">
          <div className="dashboard-header">
            <h2>Financial Dashboard</h2>
            <div className="dashboard-controls">
              {isAuthenticated && (
                <div className="connection-status connected">
                  <span className="status-dot"></span>
                  Connected
                </div>
              )}
            </div>
          </div>
          <div className="dashboard-content">
            <PowerBIEmbed isAuthenticated={isAuthenticated} />
          </div>
        </section>

        {/* Chat Section */}
        <section className="chat-section">
          <div className="chat-header">
            <h2>AI Assistant</h2>
            <div className="chat-status">
              <span className={`status-indicator ${isAuthenticated ? 'online' : 'offline'}`}></span>
              <span className="status-text">
                {isAuthenticated ? 'Ready to help' : 'Sign in required'}
              </span>
            </div>
          </div>

          <div className="chat-messages">
            {messages.map((message) => (
              <Message
                key={message.id}
                message={message.content}
                isUser={message.isUser}
                intent={message.intent}
                suggestions={message.suggestions}
                onSuggestionClick={handleSuggestionClick}
              />
            ))}
            {isLoading && <LoadingDots />}
            <div ref={messagesEndRef} />
          </div>

          <div className="chat-input-section">
            <div className="input-container">
              <textarea
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder={isAuthenticated ? 
                  "Ask me about your financial data..." : 
                  "Sign in to start chatting..."
                }
                disabled={!isAuthenticated || isLoading}
                rows={1}
                className="message-input"
              />
              <button
                onClick={() => sendMessage()}
                disabled={!isAuthenticated || isLoading || !inputMessage.trim()}
                className="send-button"
              >
                {isLoading ? (
                  <div className="button-spinner"></div>
                ) : (
                  <span>Send</span>
                )}
              </button>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
};

export default App;