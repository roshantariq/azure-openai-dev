// App.js - Main React Component for Finance Copilot
import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './App.css';

// REPLACE THESE PLACEHOLDER VALUES AFTER CREATING AZURE AD APP REGISTRATION
const AZURE_CONFIG = {
  clientId: 'YOUR_AZURE_CLIENT_ID_HERE', // TODO: REPLACE WITH ACTUAL CLIENT ID
  authority: 'https://login.microsoftonline.com/YOUR_TENANT_ID_HERE', // TODO: REPLACE WITH ACTUAL TENANT ID
  redirectUri: 'http://localhost:3000', // TODO: UPDATE FOR PRODUCTION
};

const POWERBI_CONFIG = {
  workspaceId: '6b668465-a48a-44f5-9f62-497fbd636e72',
  reportId: '2f9547f9-9f5c-465c-a181-d6d31df39d6a',
  defaultPage: '2ed3ae8ddb60d87a2bea',
};

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Mock authentication service (replace with actual MSAL in production)
class MockAuthService {
  constructor() {
    this.isAuthenticated = false;
    this.accessToken = null;
    this.user = null;
  }

  async login() {
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
        resolve(this.user);
      }, 1000);
    });
  }

  async logout() {
    this.isAuthenticated = false;
    this.accessToken = null;
    this.user = null;
  }

  getAccessToken() {
    return this.accessToken;
  }

  getUser() {
    return this.user;
  }

  isUserAuthenticated() {
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

// Message component
const Message = ({ message, isUser, intent, suggestions, onSuggestionClick }) => {
  const intentColors = {
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
        </div>
      )}
      <div className="message-content">
        {typeof message === 'string' ? (
          <p>{message}</p>
        ) : (
          <div>
            <p>{message.response || message.text}</p>
            {message.data && (
              <div className="data-preview">
                <small className="text-gray-600">
                  {Array.isArray(message.data) 
                    ? `${message.data.length} records found` 
                    : 'Data updated'}
                </small>
              </div>
            )}
          </div>
        )}
      </div>
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
  );
};

// Power BI Embed Component
const PowerBIEmbed = ({ isAuthenticated }) => {
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
const LoadingDots = () => (
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
const App = () => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [user, setUser] = useState(null);
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isSigningIn, setIsSigningIn] = useState(false);
  const messagesEndRef = useRef(null);

  // Scroll to bottom when new messages are added
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Initialize with welcome message
  useEffect(() => {
    setMessages([{
      id: Date.now(),
      content: "Welcome! I'm your Finance Copilot. I can help you analyze financial data, filter dashboards, and provide insights. Try asking me something like 'Show me monthly revenue trends' or 'Top 5 customers by value'.",
      isUser: false,
      intent: 'system',
      suggestions: [
        'Monthly revenue analysis',
        'Top products by revenue',
        'Customer retention analysis',
        'Quarterly growth trends'
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
        content: `Welcome ${user.displayName}! You're now connected to your Power BI workspace. Your dashboard is loading...`,
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
  const sendMessage = async (messageText = inputMessage) => {
    if (!messageText.trim()) return;

    const userMessage = {
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

      const botMessage = {
        id: Date.now() + 1,
        content: {
          response: response.data.response,
          data: response.data.data
        },
        isUser: false,
        intent: response.data.intent,
        suggestions: response.data.suggestions
      };

      setMessages(prev => [...prev, botMessage]);

    } catch (error) {
      console.error('Chat error:', error);
      
      const errorMessage = {
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

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const handleSuggestionClick = (suggestion) => {
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