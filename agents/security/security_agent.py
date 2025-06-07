from typing import Any, Dict, List, Optional
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import BaseTool, Tool
from langchain.memory import ConversationBufferMemory
from langchain_anthropic import ChatAnthropic
from ..base_agent import BaseAgent, AgentResponse
import httpx
import os

class SecurityAgent(BaseAgent):
    """Authentication & Security Agent that manages user validation and secure sessions"""
    
    def __init__(self):
        super().__init__(
            name="security",
            description="Manages user validation and secure sessions"
        )
        self.llm = ChatAnthropic(model="claude-3-7-sonnet-20250219", api_key=os.getenv("ANTHROPIC_API_KEY"), stream=False)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a security agent responsible for user authentication\nand session management. You must validate credentials, manage sessions,\nand enforce security policies.\n\nAvailable tools: {tools}\nTool names: {tool_names}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
    
    def initialize(self) -> None:
        """Initialize the security agent"""
        self.agent_executor = create_react_agent(
            llm=self.llm,
            tools=self.get_tools(),
            prompt=self.prompt
        )
    
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """Process authentication and security requests"""
        try:
            # Extract security request details
            request_type = input_data.get("type", "")
            credentials = input_data.get("credentials", {})
            context = input_data.get("context", {})
            
            # Process with agent executor
            result = await self.agent_executor.ainvoke({
                "input": f"Process {request_type} request with credentials: {credentials}",
                "chat_history": self.memory.chat_memory.messages,
                "context": context
            })
            
            # Update memory
            self.memory.save_context(
                {"input": f"Process {request_type}"},
                {"output": result["output"]}
            )
            
            return AgentResponse(
                success=True,
                message="Security request processed successfully",
                data={"response": result["output"]}
            )
            
        except Exception as e:
            return AgentResponse(
                success=False,
                message="Error processing security request",
                error=str(e)
            )
    
    def get_tools(self) -> List[Tool]:
        """Return the list of tools available to the security agent"""
        return [
            Tool(
                name="authenticate_user",
                func=self._authenticate_user,
                description="Authenticate a user with credentials"
            ),
            Tool(
                name="authorize_transaction",
                func=self._authorize_transaction,
                description="Authorize a transaction for a user"
            ),
            Tool(
                name="get_security_context",
                func=self._get_security_context,
                description="Get the current security context"
            )
        ]
    
    def _authenticate_user(self, credentials: str) -> str:
        """Authenticate a user with credentials"""
        # Implementation will be added when mock auth API is created
        return f"Authenticated user with credentials: {credentials}"
    
    def _validate_session(self, session_id: str) -> str:
        """Validate a user session"""
        # Implementation will be added when mock auth API is created
        return f"Validated session: {session_id}"
    
    def _generate_mfa_challenge(self, user_id: str) -> str:
        """Generate a multi-factor authentication challenge"""
        # Implementation will be added when mock auth API is created
        return f"Generated MFA challenge for user: {user_id}"
    
    def _verify_mfa_code(self, challenge_id: str, code: str) -> str:
        """Verify a multi-factor authentication code"""
        # Implementation will be added when mock auth API is created
        return f"Verified MFA code for challenge: {challenge_id}"
    
    def _check_permissions(self, user_id: str, operation: str) -> str:
        """Check user permissions for an operation"""
        # Implementation will be added when mock auth API is created
        return f"Checked permissions for user {user_id} and operation {operation}" 
    
    def _authorize_transaction(self, transaction_details: str) -> str:
        """Authorize a transaction for a user"""
        # Implementation will be added when mock auth API is created
        return f"Authorized transaction: {transaction_details}"
    
    def _get_security_context(self, _: str) -> str:
        """Get the current security context"""
        # Implementation will be added when mock auth API is created
        return "Current security context information"

    def authenticate_user(self, username: str, password: str) -> str:
        try:
            resp = httpx.post(
                "http://localhost:8000/token",
                data={"username": username, "password": password},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                return f"Authentication successful. Access token: {data.get('access_token')}"
            else:
                return f"Authentication failed: {resp.text}"
        except Exception as e:
            return f"Error during authentication: {str(e)}"

    def validate_session(self, access_token: str) -> str:
        try:
            headers = {"Authorization": f"Bearer {access_token}"}
            resp = httpx.get(
                "http://localhost:8000/users/me",
                headers=headers,
                timeout=10
            )
            if resp.status_code == 200:
                return "Session is valid."
            else:
                return f"Session validation failed: {resp.text}"
        except Exception as e:
            return f"Error during session validation: {str(e)}" 