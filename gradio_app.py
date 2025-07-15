"""
Multi-Agent Travel Planning System
A LangGraph-based travel assistant with specialized agents for flights, hotels, and itineraries.
"""

import os
import json
from typing import TypedDict, Annotated, List, Optional, Union
import operator
from dotenv import load_dotenv
import gradio as gr
import uuid

# Load environment variables
load_dotenv()

# Core imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

# Tool imports
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
import serpapi


class TravelPlannerState(TypedDict):
    """State schema for travel multiagent system"""
    messages: Annotated[List[BaseMessage], operator.add]
    next_agent: Optional[str]
    user_query: Optional[str]


class TravelPlannerApp:
    """Main travel planner application class"""
    
    def __init__(self):
        # Check for required environment variables
        required_vars = ['GOOGLE_API_KEY', 'TAVILY_API_KEY', 'SERPAPI_API_KEY']
        missing_vars = [var for var in required_vars if not os.environ.get(var)]
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        self.llm = self._setup_llm()
        self.tools = self._setup_tools()
        self.agents = self._setup_agents()
        self.router = self._create_router()
        self.workflow = self._build_workflow()
        
    def _setup_llm(self):
        """Initialize the LLM"""
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.2,
            google_api_key=os.environ.get("GOOGLE_API_KEY")
        )
    
    def _setup_tools(self):
        """Setup external tools"""
        # Tavily search tool
        tavily_tool = TavilySearch(max_results=2)
        
        # Define SERP API tools using @tool decorator
        @tool
        def search_flights(departure_airport: str, arrival_airport: str, 
                          outbound_date: str, return_date: str = None, 
                          adults: int = 1, children: int = 0) -> str:
            """Search for flights using Google Flights engine via SERP API"""
            return self._search_flights(departure_airport, arrival_airport, 
                                      outbound_date, return_date, adults, children)
        
        @tool
        def search_hotels(location: str, check_in_date: str, check_out_date: str, 
                         adults: int = 1, children: int = 0, rooms: int = 1, 
                         hotel_class: str = None, sort_by: int = 8) -> str:
            """Search for hotels using Google Hotels engine via SERP API"""
            return self._search_hotels(location, check_in_date, check_out_date, 
                                     adults, children, rooms, hotel_class, sort_by)
        
        return {
            "tavily": tavily_tool,
            "search_flights": search_flights,
            "search_hotels": search_hotels
        }
    
    def _search_flights(self, departure_airport: str, arrival_airport: str, 
                       outbound_date: str, return_date: str = None, 
                       adults: int = 1, children: int = 0) -> str:
        """Search for flights using Google Flights engine via SERP API"""
        try:
            params = {
                'api_key': os.environ.get('SERPAPI_API_KEY'),
                'engine': 'google_flights',
                'hl': 'en',
                'gl': 'us',
                'departure_id': departure_airport,
                'arrival_id': arrival_airport,
                'outbound_date': outbound_date,
                'currency': 'USD',
                'adults': adults,
                'children': children,
            }
            
            # Set trip type based on return_date
            if return_date:
                params['return_date'] = return_date
                params['type'] = '1'  # Round trip
            else:
                params['type'] = '2'  # One way
            
            print(f"🔍 Searching flights with params: {params}")
            
            # Add timeout to prevent hanging
            import time
            start_time = time.time()
            
            search = serpapi.search(params)
            
            elapsed = time.time() - start_time
            print(f"⏱️ Search completed in {elapsed:.2f} seconds")
            
            if not search.data:
                return "No search results returned from SERP API"
            
            # Try different result keys depending on trip type
            possible_keys = ['best_flights', 'other_flights', 'flights']
            results = None
            
            for key in possible_keys:
                if key in search.data and search.data[key]:
                    results = search.data[key]
                    break
            
            if not results:
                available_keys = list(search.data.keys())
                return f"No flights found. Available data keys: {available_keys}"
            
            return json.dumps(results, indent=2)
            
        except Exception as e:
            error_msg = f"Flight search failed: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def _search_hotels(self, location: str, check_in_date: str, check_out_date: str, 
                      adults: int = 1, children: int = 0, rooms: int = 1, 
                      hotel_class: str = None, sort_by: int = 8) -> str:
        """Search for hotels using Google Hotels engine via SERP API"""
        try:
            adults = int(float(adults)) if adults else 1
            children = int(float(children)) if children else 0
            rooms = int(float(rooms)) if rooms else 1
            sort_by = int(float(sort_by)) if sort_by else 8
            
            params = {
                'api_key': os.environ.get('SERPAPI_API_KEY'),
                'engine': 'google_hotels',
                'hl': 'en',
                'gl': 'us',
                'q': location,
                'check_in_date': check_in_date,
                'check_out_date': check_out_date,
                'currency': 'USD',
                'adults': adults,
                'children': children,
                'rooms': rooms,
                'sort_by': sort_by
            }
            
            if hotel_class:
                params['hotel_class'] = hotel_class
            
            print(f"🔍 Searching hotels with params: {params}")
            
            # Add timeout to prevent hanging
            import time
            start_time = time.time()
            
            search = serpapi.search(params)
            
            elapsed = time.time() - start_time
            print(f"⏱️ Search completed in {elapsed:.2f} seconds")
            
            if not search.data:
                return "No search results returned from SERP API"
            
            properties = search.data.get('properties', [])
            
            if not properties:
                available_keys = list(search.data.keys())
                return f"No hotels found in results. Available data keys: {available_keys}"
            
            # Return formatted results
            results = []
            for hotel in properties[:5]:  # Top 5 results
                hotel_info = {
                    'name': hotel.get('name', 'Unknown'),
                    'price': hotel.get('rate_per_night', 'Price not available'),
                    'rating': hotel.get('overall_rating', 'No rating'),
                    'description': hotel.get('description', 'No description'),
                    'amenities': hotel.get('amenities', [])
                }
                results.append(hotel_info)
            
            return json.dumps(results, indent=2)
            
        except Exception as e:
            error_msg = f"Hotel search failed: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def _setup_agents(self):
        """Setup all specialized agents"""
        
        # Itinerary Agent
        itinerary_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert travel itinerary planner. ONLY respond to travel planning and itinerary-related questions.

IMPORTANT RULES:
- If asked about non-travel topics (weather, math, general questions), politely decline and redirect to travel planning
- Always provide complete, well-formatted itineraries with specific details
- Include timing, locations, transportation, and practical tips

Use the ReAct approach:
1. THOUGHT: Analyze what travel information is needed
2. ACTION: Search for current information about destinations, attractions, prices, hours
3. OBSERVATION: Process the search results
4. Provide a comprehensive, formatted response

Available tools:
- tavily_search_results_json: Search for current travel information

Format your itineraries with:
- Clear day-by-day breakdown
- Specific times and locations
- Transportation between locations
- Estimated costs when possible
- Practical tips and recommendations"""),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        # Flight Agent
        flight_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a flight booking expert. ONLY respond to flight-related queries.

IMPORTANT RULES:
- If asked about non-flight topics, politely decline and redirect to flight booking
- Always use the search_flights tool to find current flight information
- For one-way flights: only provide departure_airport, arrival_airport, and outbound_date
- For round-trip flights: include return_date parameter
- CRITICAL: When parsing dates, pay attention to the year mentioned by the user
- If no year is specified, assume the current year (2025)
- Format dates as YYYY-MM-DD (e.g., 2025-07-15 for July 15, 2025)

Available tools:
- search_flights: Search for comprehensive flight data

Parameters for search_flights:
- departure_airport: 3-letter airport code (e.g., "DEL", "JFK")
- arrival_airport: 3-letter airport code (e.g., "LHR", "LAX", "DXB")
- outbound_date: Date in YYYY-MM-DD format (IMPORTANT: Use correct year!)
- return_date: Optional, only for round-trip flights
- adults: Number of adult passengers (default: 1)
- children: Number of child passengers (default: 0)

Examples:
- "15 Jul 2025" → "2025-07-15"
- "July 15, 2025" → "2025-07-15"
- "15th July 2025" → "2025-07-15"
- "15 Jul" (no year specified) → "2025-07-15"

Process:
1. ALWAYS search for flights first using the tool
2. Analyze the results to find flights matching user preferences
3. Present organized results with clear recommendations

Airport code mapping:
- Delhi: DEL
- London Heathrow: LHR
- London Gatwick: LGW
- Dubai: DXB
- New York JFK: JFK
- New York LaGuardia: LGA
- New York Newark: EWR
- etc."""),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        # Hotel Agent
        hotel_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a hotel booking expert. ONLY respond to hotel and accommodation-related queries.

IMPORTANT RULES:
- If asked about non-hotel topics, politely decline and redirect to hotel booking
- Always use the search_hotels tool to find current hotel information
- Provide detailed hotel options with prices, ratings, amenities, and location details
- Include practical booking advice and tips
- You CAN search and analyze results for different criteria like star ratings, price ranges, amenities

Available tools:
- search_hotels: Search for hotels using Google Hotels engine

When searching hotels, if check-out date is not provided:
- Ask the user for check-out date, or
- Assume a 1-night stay (add 1 day to check-in date)

For hotel searches, you need:
- Location/destination
- Check-in date (YYYY-MM-DD format)
- Check-out date (YYYY-MM-DD format) 
- Number of guests (adults, children)
- Number of rooms
- Hotel preferences (star rating, amenities, etc.)

Present results with:
- Hotel name and star rating
- Price per night and total cost
- Key amenities and features
- Location and nearby attractions
- Booking recommendations"""),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        # Bind tools to agents
        itinerary_agent = itinerary_prompt | self.llm.bind_tools([self.tools["tavily"]])
        flight_agent = flight_prompt | self.llm.bind_tools([self.tools["search_flights"]])
        hotel_agent = hotel_prompt | self.llm.bind_tools([self.tools["search_hotels"]])
        
        return {
            "itinerary": itinerary_agent,
            "flight": flight_agent,
            "hotel": hotel_agent
        }
    
    def _create_router(self):
        """Create routing logic for agent selection"""
        router_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a routing expert for a travel planning system.

        Analyze the user's query and decide which specialist agent should handle it:

        - FLIGHT: Flight bookings, airlines, air travel, flight search, tickets, airports, departures, arrivals, airline prices
        - HOTEL: Hotels, accommodations, stays, rooms, hotel bookings, lodging, resorts, hotel search, hotel prices
        - ITINERARY: Travel itineraries, trip planning, destinations, activities, attractions, sightseeing, travel advice, weather, culture, food, general travel questions

        Respond with ONLY one word: FLIGHT, HOTEL, or ITINERARY

        Examples:
        "Book me a flight to Paris" → FLIGHT
        "Find hotels in Tokyo" → HOTEL
        "Plan my 5-day trip to Italy" → ITINERARY
        "Search flights from NYC to London" → FLIGHT
        "Where should I stay in Bali?" → HOTEL
        "What are the best attractions in Rome?" → ITINERARY
        "I need airline tickets" → FLIGHT
        "Show me hotel options" → HOTEL
        "Create an itinerary for Japan" → ITINERARY"""),
            ("user", "Query: {query}")
        ])
        
        router_chain = router_prompt | self.llm | StrOutputParser()
        
        def route_query(state):
            """Router function - decides which agent to call next"""
            user_message = state["messages"][-1].content
            
            try:
                decision = router_chain.invoke({"query": user_message}).strip().upper()
                agent_mapping = {
                    "FLIGHT": "flight_agent",
                    "HOTEL": "hotel_agent",
                    "ITINERARY": "itinerary_agent"
                }
                next_agent = agent_mapping.get(decision, "itinerary_agent")
                return next_agent
            except Exception:
                return "itinerary_agent"
        
        return route_query
    
    def _itinerary_agent_node(self, state: TravelPlannerState):
        """Itinerary planning agent node"""
        messages = state["messages"]
        response = self.agents["itinerary"].invoke({"messages": messages})
        
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_messages = []
            for tool_call in response.tool_calls:
                if tool_call['name'] == 'tavily_search_results_json':
                    try:
                        print(f"🔍 Tavily search query: {tool_call['args'].get('query', 'No query')}")
                        
                        # Use the direct search method instead of invoke
                        search_query = tool_call['args'].get('query', '')
                        if search_query:
                            tool_result = self.tools["tavily"].search(search_query, max_results=2)
                        else:
                            tool_result = {"error": "No search query provided"}
                        
                        print(f"📋 Tavily raw result: {type(tool_result)} - {str(tool_result)[:200]}...")
                        
                        # Handle different response types
                        if isinstance(tool_result, list):
                            if len(tool_result) == 0:
                                tool_result = "No search results found"
                            else:
                                tool_result = json.dumps(tool_result, indent=2)
                        elif isinstance(tool_result, dict):
                            tool_result = json.dumps(tool_result, indent=2)
                        elif not tool_result:
                            tool_result = "No search results found"
                        
                        # Ensure it's a string and not empty
                        tool_result = str(tool_result)
                        if not tool_result or tool_result.strip() == "":
                            tool_result = "Search completed but no results returned"
                        
                        print(f"✅ Processed tool result length: {len(tool_result)}")
                        
                    except Exception as e:
                        print(f"❌ Tavily search error: {e}")
                        tool_result = f"Search failed: {str(e)}"
                    
                    tool_messages.append(ToolMessage(
                        content=tool_result,
                        tool_call_id=tool_call['id']
                    ))
            
            if tool_messages:
                all_messages = messages + [response] + tool_messages
                try:
                    final_response = self.agents["itinerary"].invoke({"messages": all_messages})
                    return {"messages": [response] + tool_messages + [final_response]}
                except Exception as e:
                    print(f"❌ Error in final response: {e}")
                    # Return a fallback response
                    fallback_response = self.agents["itinerary"].invoke({"messages": messages})
                    return {"messages": [fallback_response]}
        
        return {"messages": [response]}
    
    def _flight_agent_node(self, state: TravelPlannerState):
        """Flight booking agent node"""
        messages = state["messages"]
        response = self.agents["flight"].invoke({"messages": messages})
        
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_messages = []
            for tool_call in response.tool_calls:
                if tool_call['name'] == 'search_flights':
                    try:
                        tool_result = self.tools["search_flights"].invoke(tool_call['args'])
                        if not tool_result or tool_result.strip() == "":
                            tool_result = "No flight results found for your search criteria."
                    except Exception as e:
                        tool_result = f"Flight search failed: {str(e)}"
                    
                    tool_messages.append(ToolMessage(
                        content=tool_result,
                        tool_call_id=tool_call['id']
                    ))
            
            if tool_messages:
                all_messages = messages + [response] + tool_messages
                final_response = self.agents["flight"].invoke({"messages": all_messages})
                return {"messages": [response] + tool_messages + [final_response]}
        
        return {"messages": [response]}
    
    def _hotel_agent_node(self, state: TravelPlannerState):
        """Hotel booking agent node"""
        messages = state["messages"]
        response = self.agents["hotel"].invoke({"messages": messages})
        
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_messages = []
            for tool_call in response.tool_calls:
                if tool_call['name'] == 'search_hotels':
                    try:
                        tool_result = self.tools["search_hotels"].invoke(tool_call['args'])
                        if not tool_result or tool_result.strip() == "":
                            tool_result = "No hotel results found for your search criteria."
                    except Exception as e:
                        tool_result = f"Hotel search failed: {str(e)}"
                    
                    tool_messages.append(ToolMessage(
                        content=tool_result,
                        tool_call_id=tool_call['id']
                    ))
            
            if tool_messages:
                all_messages = messages + [response] + tool_messages
                final_response = self.agents["hotel"].invoke({"messages": all_messages})
                return {"messages": [response] + tool_messages + [final_response]}
        
        return {"messages": [response]}
    
    def _router_node(self, state: TravelPlannerState):
        """Router node - determines which agent should handle the query"""
        user_message = state["messages"][-1].content
        next_agent = self.router(state)
        
        return {
            "next_agent": next_agent,
            "user_query": user_message
        }
    
    def _route_to_agent(self, state: TravelPlannerState):
        """Conditional edge function - routes to appropriate agent"""
        next_agent = state.get("next_agent")
        
        if next_agent == "flight_agent":
            return "flight_agent"
        elif next_agent == "hotel_agent":
            return "hotel_agent"
        elif next_agent == "itinerary_agent":
            return "itinerary_agent"
        else:
            return "itinerary_agent"
    
    def _build_workflow(self):
        """Build the complete LangGraph workflow"""
        workflow = StateGraph(TravelPlannerState)
        
        # Add nodes
        workflow.add_node("router", self._router_node)
        workflow.add_node("flight_agent", self._flight_agent_node)
        workflow.add_node("hotel_agent", self._hotel_agent_node)
        workflow.add_node("itinerary_agent", self._itinerary_agent_node)
        
        # Set entry point
        workflow.set_entry_point("router")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "router",
            self._route_to_agent,
            {
                "flight_agent": "flight_agent",
                "hotel_agent": "hotel_agent",
                "itinerary_agent": "itinerary_agent"
            }
        )
        
        # Add edges to END
        workflow.add_edge("flight_agent", END)
        workflow.add_edge("hotel_agent", END)
        workflow.add_edge("itinerary_agent", END)
        
        # Compile with memory
        checkpointer = InMemorySaver()
        return workflow.compile(checkpointer=checkpointer)
    
    def chat(self, message: str, thread_id: str = "default"):
        """Process a single message and return response"""
        config = {"configurable": {"thread_id": thread_id}}
        
        result = self.workflow.invoke(
            {"messages": [HumanMessage(content=message)]},
            config
        )
        
        return result["messages"][-1].content
    
    def chat_stream(self, message: str, thread_id: str = "default"):
        """Stream response for a message"""
        config = {"configurable": {"thread_id": thread_id}}
        
        for chunk in self.workflow.stream(
            {"messages": [HumanMessage(content=message)]},
            config
        ):
            yield chunk


# For LangGraph Cloud deployment
app = TravelPlannerApp()

# Gradio Interface Functions
def create_gradio_interface():
    """Create and configure the Gradio interface"""
    
    def chat_function(message, history, session_id):
        """Handle chat messages with session memory"""
        try:
            # Use session_id as thread_id for maintaining conversation context
            response = app.chat(message, thread_id=session_id)
            return response
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def reset_conversation():
        """Reset conversation by returning new session ID"""
        return str(uuid.uuid4())
    
    # Create the Gradio interface
    with gr.Blocks(
        title="🧳 Multi-Agent Travel Planner",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 900px !important;
        }
        .chat-message {
            font-size: 14px !important;
        }
        """
    ) as demo:
        
        gr.Markdown("""
        # 🧳 Multi-Agent Travel Planning System
        
        **Your AI-powered travel assistant with specialized agents for:**
        - ✈️ **Flight Search & Booking** - Find and compare flights
        - 🏨 **Hotel Search & Booking** - Discover accommodations
        - 🗺️ **Itinerary Planning** - Create detailed travel plans
        
        Just type your travel question and let our agents help you plan your perfect trip!
        """)
        
        # Session state for maintaining conversation context
        session_id = gr.State(value=str(uuid.uuid4()))
        
        # Chat interface
        chatbot = gr.Chatbot(
            label="Travel Assistant",
            height=500,
            show_label=True,
            container=True,
            bubble_full_width=False
        )
        
        with gr.Row():
            msg = gr.Textbox(
                placeholder="Ask me about flights, hotels, or travel planning...",
                label="Your Message",
                scale=4,
                container=False
            )
            send_btn = gr.Button("Send", scale=1, variant="primary")
        
        with gr.Row():
            clear_btn = gr.Button("Clear Chat", scale=1)
            gr.Markdown("**Examples:** *Find flights from NYC to London*, *Hotels in Tokyo for 3 nights*, *Plan a 5-day trip to Italy*")
        
        # Event handlers
        def respond(message, history, session_id):
            if not message.strip():
                return history, ""
            
            # Add user message to history
            history.append([message, None])
            
            # Get bot response
            bot_response = chat_function(message, history, session_id)
            
            # Add bot response to history
            history[-1][1] = bot_response
            
            return history, ""
        
        def clear_chat():
            return [], str(uuid.uuid4())
        
        # Wire up the events
        msg.submit(
            respond,
            inputs=[msg, chatbot, session_id],
            outputs=[chatbot, msg]
        )
        
        send_btn.click(
            respond,
            inputs=[msg, chatbot, session_id],
            outputs=[chatbot, msg]
        )
        
        clear_btn.click(
            clear_chat,
            outputs=[chatbot, session_id]
        )
        
        # Example buttons
        gr.Examples(
            examples=[
                "Find flights from Delhi to Dubai on July 25, 2025",
                "Search hotels in Paris for 2 nights starting August 1st",
                "Plan a 7-day itinerary for Japan",
                "I need round-trip flights from NYC to London",
                "Best hotels in Bali under $200 per night"
            ],
            inputs=msg,
            label="Example Queries"
        )
        
        gr.Markdown("""
        ---
        💡 **Tips:**
        - Be specific with dates, locations, and preferences
        - The system remembers your conversation context
        - Each agent specializes in their domain for better results
        """)
    
    return demo


def main():
    """Main function to launch the Gradio interface"""
    print("🚀 Starting Multi-Agent Travel Planning System...")
    
    try:
        # Create and launch the Gradio interface
        demo = create_gradio_interface()
        
        # Launch the interface
        demo.launch(
            share=False,  # Set to True if you want to create a public link
            server_name="127.0.0.1",  # Use localhost instead of 0.0.0.0
            server_port=7860,
            show_error=True,
            quiet=False,
            inbrowser=True  # Automatically open browser
        )
        
    except Exception as e:
        print(f"❌ Error launching interface: {str(e)}")
        print("Please check your environment variables and dependencies.")


if __name__ == "__main__":
    main()