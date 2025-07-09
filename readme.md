# Multi-Agent Travel Planning System

A sophisticated travel planning assistant built with LangGraph that uses specialized agents for flights, hotels, and itinerary planning.

## Features

- **🎯 Smart Routing**: Automatically routes queries to the appropriate specialist agent
- **✈️ Flight Agent**: Searches and compares flight options using real-time data
- **🏨 Hotel Agent**: Finds and recommends accommodations based on preferences
- **🗓️ Itinerary Agent**: Creates detailed travel plans with attractions, timing, and tips
- **💬 Memory**: Maintains conversation context across multiple interactions
- **🔍 Real-time Search**: Uses Tavily and SERP API for current information


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

### 3. Run Locally

```python
python travel_agent.py
```

### 4. Deploy to LangGraph Cloud

```bash
# Install LangGraph CLI
pip install langgraph-cli

# Deploy to LangGraph Cloud
langgraph deploy
```

## Usage Examples

### Flight Search
```
User: "Find flights from NYC to London on December 15th, 2025"
Agent: [Searches real-time flight data and presents options]
```

### Hotel Booking
```
User: "Find 4-star hotels in Paris for Jul, 10-15 2015 for 1 adult"
Agent: [Shows available hotels with prices and amenities]
```

### Itinerary Planning
```
User: "Plan a 3-day trip to Tokyo"
Agent: [Creates detailed day-by-day itinerary with attractions, timing, and tips]
```

## File Structure

```
├── travel_agent.py      # Main application file
├── requirements.txt     # Python dependencies
├── langgraph.json      # LangGraph Cloud configuration
├── .env.example        # Environment variables template
└── README.md           # This file
```

## Key Components

### Agents
- **Router**: Analyzes queries and routes to appropriate specialist
- **Flight Agent**: Handles flight search and booking queries
- **Hotel Agent**: Manages accommodation search and recommendations
- **Itinerary Agent**: Creates comprehensive travel plans


## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_API_KEY` | Google Gemini API key | Yes |
| `TAVILY_API_KEY` | Tavily search API key | Yes |
| `SERPAPI_API_KEY` | SERP API key for Google search | Yes |




## Troubleshooting

### Common Issues
1. **API Key Errors**: Ensure all API keys are set in `.env`
2. **Import Errors**: Run `pip install -r requirements.txt`
3. **Memory Issues**: Check LangGraph checkpoint configuration





