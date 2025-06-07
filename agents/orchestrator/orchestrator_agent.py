from typing import Any, Dict, List, Optional
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import BaseTool, Tool
from langchain.memory import ConversationBufferMemory
from langchain_anthropic import ChatAnthropic
from ..base_agent import BaseAgent, AgentResponse
import os
import logging
from shared.database.base import SessionLocal
from shared.models.account import Transaction, Account
import httpx
from agents.financial.financial_agent import FinancialAgent
from agents.security.security_agent import SecurityAgent
import re


class OrchestratorAgent(BaseAgent):
    """Orchestrator Agent that manages workflow and task delegation"""
    
    def __init__(self):
        super().__init__(
            name="orchestrator",
            description="Manages user intent, task delegation, and conversation flow"
        )
        self.llm = ChatAnthropic(
            model="claude-3-7-sonnet-20250219",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            stream=False
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are TaskFin, an intelligent, friendly financial assistant. Your name is TaskFin. Never say you are Claude, Anthropic, or any other AI model. Always answer as TaskFin. If the user asks about your identity, say: 'I am TaskFin, your financial assistant.'
            You help users with financial tasks (like accounts, payments, payees, subscriptions), but you are also capable of general conversation, small talk, and answering questions about yourself or the world. Always be clear, helpful, and conversational. If the user asks something unrelated to finance, respond naturally and engagingly.
            
            You have access to the following tools:
            {tools}
            Tool names: {tool_names}
            
            When making a payment, always ask the user if they want to use an existing account or add a new one, and if they want to pay an existing payee or add a new payee. Guide the user to provide any missing details.
            
            When you need to use a tool, use the following format:
            Thought: [your reasoning here]
            Action: [the tool name, from the list above]
            Action Input: [the input to the tool]
            
            When you want to respond directly to the user, use:
            Thought: [your reasoning here]
            Final Answer: [your response here]
            
            IMPORTANT:
            - Only take ONE step at a time.
            - If you use Action, output ONLY one Action and Action Input, and nothing else.
            - If you use Final Answer, output ONLY the Final Answer and nothing else.
            - Do NOT output multiple Actions or both Action and Final Answer in the same response.
            - If the user asks a general, personal, or small talk question, respond conversationally and do not mention financial tools unless relevant.
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}\n{agent_scratchpad}")
        ])
        self.financial_agent = FinancialAgent()
        self.security_agent = SecurityAgent()
    
    def initialize(self) -> None:
        """Initialize the orchestrator agent"""
        agent = create_react_agent(
            llm=self.llm,
            tools=self.get_tools(),
            prompt=self.prompt
        )
        self.agent_executor = AgentExecutor(agent=agent, tools=self.get_tools(), verbose=True, handle_parsing_errors=True)
    
    async def process(self, input_data: dict) -> dict:
        """Process user input, classify intent, and delegate to the correct agent/tool."""
        user_input = input_data.get("input", "")
        context = input_data.get("context", {})
        lowered = user_input.lower().strip()
        # Respond to greetings without requiring authentication
        if lowered in ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"]:
            return {"success": True, "data": {"response": "Hello! I'm your financial assistant, ready to help you with tasks like making payments, managing payees, and other financial activities. How can I assist you today?"}}
        # Finance-related intents (require authentication)
        if any(word in lowered for word in ["account", "balance"]):
            intent = "list_accounts"
        elif any(word in lowered for word in ["payee"]):
            if any(word in lowered for word in ["add", "new", "create"]):
                intent = "add_payee"
            else:
                intent = "list_payees"
        elif any(word in lowered for word in ["transaction", "history"]):
            # Extract 'last N transactions' or 'N transactions' if present, supporting digit and word forms
            number_words = {
                'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
            }
            match = re.search(r"(?:last |past )?(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten) transactions", lowered)
            if match:
                n = match.group(1)
                if n.isdigit():
                    n_val = int(n)
                else:
                    n_val = number_words.get(n, 5)
                intent = ("list_transactions", n_val)
            else:
                intent = ("list_transactions", None)
        elif any(word in lowered for word in ["payment", "pay", "send money", "transfer"]):
            if any(word in lowered for word in ["make", "send", "pay"]):
                intent = "make_payment"
            else:
                intent = "list_payments"
        elif any(word in lowered for word in ["subscription", "recurring", "auto-pay"]):
            intent = "list_subscriptions"
        elif any(word in lowered for word in ["add account", "new account", "create account"]):
            intent = "add_account"
        else:
            # Route all other input (including small talk/general questions) through the agent chain
            try:
                result = await self.agent_executor.ainvoke({
                    "input": user_input,
                    "chat_history": self.memory.chat_memory.messages,
                    "context": context
                })
                return {"success": True, "data": {"response": result["output"] if isinstance(result, dict) and "output" in result else str(result)}}
            except Exception as e:
                return {"success": True, "data": {"response": "I'm here to help with your financial tasks and general questions! (Agent error: " + str(e) + ")"}}
        # For finance intents, require authentication
        user_id = context.get('user_id')
        access_token = context.get('access_token')
        if not user_id or not access_token:
            return {"success": True, "data": {"response": "You are not authenticated. Please log in to access your financial data."}}
        output = self._route_intent(intent, input_data, context)
        return {"success": True, "data": {"response": output}}
    
    def get_tools(self) -> List[Tool]:
        """Return the list of tools available to the orchestrator"""
        return [
            Tool(
                name="delegate_to_financial_agent",
                func=lambda task: self._delegate_to_financial_agent(task, context=getattr(self, '_latest_context', None)),
                description="Delegate financial tasks to the financial agent"
            ),
            Tool(
                name="add_payee",
                func=self._add_payee,
                description="Add a new payee (friend or organization) to the user's payee list"
            ),
            Tool(
                name="make_payment",
                func=self._make_payment,
                description="Make a payment to a payee from a specified account. Always ask the user if they want to use an existing account or add a new one, and if they want to pay an existing payee or add a new payee. Guide the user to provide any missing details."
            ),
            Tool(
                name="list_payees",
                func=self._list_payees,
                description="List all payees for the current user."
            ),
            Tool(
                name="get_conversation_context",
                func=self._get_conversation_context,
                description="Get the current conversation context"
            ),
            Tool(
                name="authenticate_user",
                func=lambda username_password: self.authenticate_user(*username_password.split(",", 1)),
                description="Authenticate a user with username and password. Input should be 'username,password'"
            ),
            Tool(
                name="validate_session",
                func=self.validate_session,
                description="Validate a user session using the access token."
            )
        ]
    
    def _delegate_to_financial_agent(self, task: str, context: Optional[dict] = None) -> str:
        """Delegate a task to the financial agent and return real recent transactions."""
        return self.financial_agent.list_transactions(context or self._latest_context)

    def _add_payee(self, name: str, type: str = "friend", email: str = "", phone: str = "", account_number: str = "", bank_name: str = "") -> str:
        return self.financial_agent.add_payee(name, type, email, phone, account_number, bank_name, self._latest_context)

    def _make_payment(self, account_id: str, payee_id: str, amount: float, description: str = "", is_recurring: bool = False, frequency: str = None, end_date: str = None) -> str:
        return self.financial_agent.make_payment(account_id, payee_id, amount, description, is_recurring, frequency, end_date, self._latest_context)

    def _list_payees(self, _: str = "") -> str:
        return self.financial_agent.list_payees(self._latest_context)
    
    def _get_conversation_context(self, _: str) -> str:
        """Get the current conversation context"""
        return str(self.get_memory())

    def authenticate_user(self, username: str, password: str) -> str:
        return self.security_agent.authenticate_user(username, password)

    def validate_session(self, access_token: str) -> str:
        return self.security_agent.validate_session(access_token)

    def _route_intent(self, intent: str, input_data: dict, context: dict) -> str:
        """Classify intent and delegate to the correct agent/tool, always passing context."""
        user_id = context.get('user_id')
        access_token = context.get('access_token')
        if not user_id or not access_token:
            return "You are not authenticated. Please log in to access your financial data."
        # Route intent
        if isinstance(intent, tuple) and intent[0] == "list_transactions":
            # Pass the original user message for LLM-driven filtering
            user_message = input_data.get("input", "")
            return self.financial_agent.list_transactions(context, limit=intent[1], user_message=user_message)
        elif intent in ["list_accounts", "show_accounts", "accounts"]:
            return self.financial_agent.list_accounts(context)
        elif intent in ["add_account"]:
            # Stub: implement add_account in FinancialAgent as needed
            return self.financial_agent.add_account(context)
        elif intent in ["list_payees", "show_payees", "payees"]:
            return self.financial_agent.list_payees(context)
        elif intent in ["add_payee"]:
            # Extract details from input_data or context as needed
            details = input_data.get("details", {})
            return self.financial_agent.add_payee(
                details.get("name", ""),
                details.get("type", "friend"),
                details.get("email", ""),
                details.get("phone", ""),
                details.get("account_number", ""),
                details.get("bank_name", ""),
                context
            )
        elif intent in ["list_transactions", "show_transactions", "transactions"]:
            user_message = input_data.get("input", "")
            return self.financial_agent.list_transactions(context, user_message=user_message)
        elif intent in ["make_payment"]:
            # Extract details from input_data or context as needed
            details = input_data.get("details", {})
            return self.financial_agent.make_payment(
                details.get("account_id", ""),
                details.get("payee_id", ""),
                details.get("amount", 0.0),
                details.get("description", ""),
                details.get("is_recurring", False),
                details.get("frequency", None),
                details.get("end_date", None),
                context
            )
        elif intent in ["list_payments"]:
            # Stub: implement list_payments in FinancialAgent as needed
            return self.financial_agent.list_payments(context)
        elif intent in ["list_subscriptions"]:
            # Stub: implement list_subscriptions in FinancialAgent as needed
            return self.financial_agent.list_subscriptions(context)
        # Add more intent routing as needed
        else:
            return "Sorry, I didn't understand your request. Please try again." 