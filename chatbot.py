import finnhub
from datetime import datetime, timedelta
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import yfinance as yf

import os
from dotenv import load_dotenv

# Ye line .env file load karegi
load_dotenv()

# Ab keys direct string ki jagah os.getenv se lo
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

# Check karne ke liye ki key load hui ya nahi (Optional)
if not GOOGLE_API_KEY:
    print("Error: Google API Key not found in .env file")
# Configuration


llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.3)
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Common ticker mappings
TICKER_MAP = {
    "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL", "amazon": "AMZN",
    "meta": "META", "facebook": "META", "tesla": "TSLA", "nvidia": "NVDA",
    "netflix": "NFLX", "intel": "INTC", "amd": "AMD", "disney": "DIS"
}

def get_ticker(query):
    """Extract ticker from query using mapping + LLM fallback"""
    query_lower = query.lower()
    for company, ticker in TICKER_MAP.items():
        if company in query_lower:
            return ticker
    
    prompt = f"Extract ONLY the stock ticker symbol (e.g., Apple->AAPL). Return 'UNKNOWN' if none found.\nQuery: {query}\nTicker:"
    response = llm.invoke(prompt)
    return response.content.strip().upper()

def fetch_price_data(ticker, days=30):
    """Get historical price data from yfinance"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(start=datetime.now() - timedelta(days=days), end=datetime.now())
        if hist.empty:
            return ""
        
        change = ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
        return f"""PRICE DATA (Last 30 Days):
Current: ${hist['Close'].iloc[-1]:.2f}
Start: ${hist['Close'].iloc[0]:.2f}
Change: {change:+.2f}%
High: ${hist['High'].max():.2f}
Low: ${hist['Low'].min():.2f}
"""
    except Exception as e:
        print(f"Price data error: {e}")
        return ""

def fetch_finnhub_news(ticker, days=21):
    """Fetch news from Finnhub"""
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        articles = finnhub_client.company_news(ticker, _from=start, to=today)
        
        if not articles:
            return ""
        
        news_text = "FINNHUB NEWS:\n"
        for article in articles[:12]:
            date = datetime.fromtimestamp(article.get('datetime', 0)).strftime('%Y-%m-%d')
            news_text += f"\n[{date}] {article.get('headline', 'No Title')}\n"
            if article.get('summary'):
                news_text += f"{article['summary']}\n"
        
        return news_text
    except Exception as e:
        print(f"Finnhub error: {e}")
        return ""

def fetch_yfinance_news(ticker):
    """Fetch news from Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news:
            return ""
        
        news_text = "\nYAHOO FINANCE NEWS:\n"
        for item in news[:10]:
            date = datetime.fromtimestamp(item.get('providerPublishTime', 0)).strftime('%Y-%m-%d')
            news_text += f"\n[{date}] {item.get('title', 'No Title')}\n"
        
        return news_text
    except Exception as e:
        print(f"Yahoo Finance error: {e}")
        return ""

def get_company_info(ticker):
    """Get basic company information"""
    try:
        info = yf.Ticker(ticker).info
        return f"""COMPANY INFO:
Name: {info.get('longName', ticker)}
Sector: {info.get('sector', 'N/A')}
Industry: {info.get('industry', 'N/A')}
Market Cap: ${info.get('marketCap', 0):,.0f}
"""
    except:
        return ""

def build_context(ticker):
    """Aggregate all context sources"""
    context_parts = [
        get_company_info(ticker),
        fetch_price_data(ticker),
        fetch_finnhub_news(ticker),
        fetch_yfinance_news(ticker)
    ]
    
    full_context = "\n".join([part for part in context_parts if part])
    print(f"Context length: {len(full_context)} chars")
    return full_context

def run_chatbot(user_query):
    """Main chatbot logic with enhanced RAG"""
    
    # Extract ticker
    ticker = get_ticker(user_query)
    if ticker == "UNKNOWN" or not ticker:
        return "Could not identify a stock ticker. Please mention a company name or ticker (e.g., 'Apple', 'AAPL')."
    
    print(f"\nAnalyzing {ticker}...")
    
    # Build comprehensive context
    context = build_context(ticker)
    
    if len(context) < 100:
        return f"Insufficient data for {ticker}. Possible reasons:\n- Invalid ticker\n- API rate limits\n- Limited news coverage"
    
    # Enhanced RAG with better chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = text_splitter.split_text(context)
    print(f"Created {len(chunks)} chunks")
    
    # Vector store with top-k retrieval
    vector_db = FAISS.from_texts(chunks, embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 6})
    
    # Enhanced prompt
    prompt = ChatPromptTemplate.from_template("""You are an expert financial analyst.

CONTEXT:
{context}

QUESTION: {question}

Provide a clear, evidence-based answer that:
1. References specific price movements (% changes)
2. Cites relevant news headlines with dates
3. Explains causation between news and price action
4. Uses concrete facts from the context

Answer:""")
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Generate answer
    print("Generating analysis...\n")
    return rag_chain.invoke(user_query)

# Main execution
if __name__ == "__main__":
    print("Stock Market Analytical Chatbot")
    print("Examples: 'Why did Apple drop?', 'What's happening with Tesla?'\n")
    
    while True:
        query = input("Question (or 'quit'): ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            break
        if query:
            print(f"\n{run_chatbot(query)}\n")