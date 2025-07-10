# Multi-Agent Travel Planning System

A sophisticated travel planning assistant built with LangGraph that uses specialized agents for flights, hotels, and itinerary planning. Features both a beautiful web UI (Chainlit) and console interface.

## Features

- **🎯 Smart Routing**: Automatically routes queries to the appropriate specialist agent
- **✈️ Flight Agent**: Searches and compares flight options using real-time data
- **🏨 Hotel Agent**: Finds and recommends accommodations based on preferences
- **🗓️ Itinerary Agent**: Creates detailed travel plans with attractions, timing, and tips
- **💬 Memory**: Maintains conversation context across multiple interactions
- **🔍 Real-time Search**: Uses Tavily and SERP API for current information
- **🌐 Web UI**: Beautiful Chainlit interface with real-time responses
- **📱 Console Mode**: Fallback terminal interface for testing

## Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo>
cd travel-planning-agent
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

Get your API keys:
- **Google AI Studio**: https://aistudio.google.com/apikey
- **Tavily**: https://app.tavily.com/home
- **SERP API**: https://serpapi.com/dashboard

### 3. Run the Application

#### Web UI (Recommended)
```bash
chainlit run app_with_ui.py
```
Then open your browser to `http://localhost:8000`

#### Console Mode
```bash
python app.py
```

## Usage Examples

### Flight Search
```
User: "Find flights from NYC to London on December 15th, 2025"
Agent: [Searches real-time flight data and presents options]
```

### Hotel Booking
```
User: "Find 4-star hotels in Paris for Jul 10-15, 2025 for 1 adult"
Agent: [Shows available hotels with prices and amenities]
```

### Itinerary Planning
```
User: "Plan a 3-day trip to Tokyo"
Agent: [Creates detailed day-by-day itinerary with attractions, timing, and tips]
```

### Multi-turn Conversations
```
User: "I want to plan a trip to Paris"
Agent: [Provides Paris travel information]
User: "Now find me flights from New York to Paris for July 15th"
Agent: [Searches flights, remembers Paris context]
User: "What about hotels near the Eiffel Tower?"
Agent: [Finds hotels, knows it's for the Paris trip]
```

## File Structure

```
├── app_with_ui.py       # Main application with Chainlit UI
├── travel_agent.py      # Console-only version (optional)
├── requirements.txt     # Python dependencies
├── langgraph.json      # LangGraph Cloud configuration
├── .env.example        # Environment variables template
└── README.md           # This file
```

## Key Components

### Agents
- **Router**: Analyzes queries and routes to appropriate specialist
- **Flight Agent**: Handles flight search and booking queries using SERP API
- **Hotel Agent**: Manages accommodation search and recommendations via Google Hotels
- **Itinerary Agent**: Creates comprehensive travel plans using Tavily search

### Technical Architecture
- **LangGraph**: Multi-agent workflow orchestration
- **Google Gemini**: Large language model for intelligent responses
- **Chainlit**: Modern web interface with real-time chat
- **Memory**: Persistent conversation history with InMemorySaver
- **Tools**: External APIs for real-time travel data

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_API_KEY` | Google Gemini API key | Yes |
| `TAVILY_API_KEY` | Tavily search API key | Yes |
| `SERPAPI_API_KEY` | SERP API key for Google search | Yes |

## Installation Details

### Core Dependencies
```bash
pip install chainlit
pip install langgraph
pip install langchain-google-genai
pip install langchain-tavily
pip install serpapi
pip install python-dotenv
```

### Complete Requirements
```bash
pip install -r requirements.txt
```

## Web UI Features

### Chainlit Interface
- **Real-time responses** with thinking indicators
- **Multi-turn conversation** memory
- **Beautiful formatting** with markdown support
- **Session management** for multiple users
- **Error handling** with user-friendly messages

### UI Elements
- 🤔 **Thinking indicators** while processing
- 🧳 **Welcome message** with feature overview
- ✅ **Success responses** with formatted results
- ❌ **Error messages** with helpful guidance

## Agent Routing Logic

The system intelligently routes queries based on keywords and context:

### Flight Agent Triggers
- Flight bookings, airlines, air travel
- Flight search, tickets, airports
- Departures, arrivals, airline prices
- Examples: "Book me a flight", "Find flights to Paris"

### Hotel Agent Triggers
- Hotels, accommodations, stays, rooms
- Hotel bookings, lodging, resorts
- Hotel search, hotel prices
- Examples: "Find hotels in Tokyo", "Where should I stay"

### Itinerary Agent Triggers
- Travel itineraries, trip planning
- Destinations, activities, attractions
- Travel advice, weather, culture, food
- Examples: "Plan my trip", "What to do in Rome"

## Troubleshooting

### Common Issues
1. **API Key Errors**: Ensure all API keys are set in `.env`
2. **Chainlit Import Error**: Run `pip install chainlit`
3. **Memory Issues**: Check LangGraph checkpoint configuration
4. **SERP API Limits**: Check your SERP API quota and billing




## Advanced Configuration

### Custom Agent Prompts
Modify agent prompts in the `_setup_agents()` method to customize behavior:
- Flight agent: Adjust date parsing and airport code handling
- Hotel agent: Modify search parameters and result formatting
- Itinerary agent: Customize planning approach and recommendations

### Memory Configuration
Adjust conversation memory settings:
```python
# In _build_workflow()
checkpointer = InMemorySaver()  # For development
# For production, consider persistent storage
```

### Tool Configuration
Customize API parameters:
```python
# Flight search parameters
params = {
    'currency': 'USD',  # Change currency
    'gl': 'us',         # Change country
    'hl': 'en',         # Change language
}
```

## Performance Tips

1. **API Rate Limits**: Monitor your API usage to avoid rate limiting
2. **Response Time**: SERP API calls may take 3-5 seconds


## Deployment Options

### Local Development
```bash
chainlit run app_with_ui.py
```
