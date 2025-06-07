from typing import Any, Dict, List, Optional
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import BaseTool, Tool
from langchain.memory import ConversationBufferMemory
from langchain_anthropic import ChatAnthropic
from ..base_agent import BaseAgent, AgentResponse
import httpx
import os

class FinancialAgent(BaseAgent):
    """Financial Transaction Agent that handles account operations and payment processing"""
    
    def __init__(self):
        super().__init__(
            name="financial",
            description="Handles account operations and payment processing"
        )
        self.llm = ChatAnthropic(model="claude-3-7-sonnet-20250219", api_key=os.getenv("ANTHROPIC_API_KEY"), stream=False)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are TaskFin, an intelligent, friendly financial assistant. Your name is TaskFin. Never say you are Claude, Anthropic, or any other AI model. Always answer as TaskFin. If the user asks about your identity, say: 'I am TaskFin, your financial assistant.'
            You are responsible for handling account operations and payment processing. You must verify account details, check balances, and process payments securely.
            Available tools: {tools}
            Tool names: {tool_names}
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
    
    def initialize(self) -> None:
        """Initialize the financial agent"""
        self.agent_executor = create_react_agent(
            llm=self.llm,
            tools=self.get_tools(),
            prompt=self.prompt
        )
    
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """Process financial transactions and account operations"""
        try:
            # Extract transaction details and context
            transaction_type = input_data.get("type", "")
            details = input_data.get("details", {})
            context = input_data.get("context", {})
            
            # Process with agent executor
            result = await self.agent_executor.ainvoke({
                "input": f"Process {transaction_type} transaction with details: {details}",
                "chat_history": self.memory.chat_memory.messages,
                "context": context
            })
            
            # Update memory
            self.memory.save_context(
                {"input": f"Process {transaction_type}"},
                {"output": result["output"]}
            )
            
            return AgentResponse(
                success=True,
                message="Transaction processed successfully",
                data={"response": result["output"]}
            )
            
        except Exception as e:
            return AgentResponse(
                success=False,
                message="Error processing transaction",
                error=str(e)
            )
    
    def get_tools(self) -> List[Tool]:
        """Return the list of tools available to the financial agent"""
        return [
            Tool(
                name="process_payment",
                func=self._process_payment,
                description="Process a payment between accounts"
            ),
            Tool(
                name="check_balance",
                func=self._check_account_balance,
                description="Check the balance of an account"
            ),
            Tool(
                name="get_transaction_history",
                func=self._get_transaction_history,
                description="Get the transaction history for an account"
            )
        ]
    
    def _check_account_balance(self, account_id: str) -> str:
        """Check the balance of a specific account"""
        # Implementation will be added when mock banking API is created
        return f"Balance for account {account_id}: $1000.00"
    
    def _process_payment(self, payment_details: str) -> str:
        """Process a payment transaction"""
        # Implementation will be added when mock banking API is created
        return f"Processed payment: {payment_details}"
    
    def _get_transaction_history(self, account_id: str) -> str:
        """Get transaction history for an account"""
        # Implementation will be added when mock banking API is created
        return f"Transaction history for account {account_id}"
    
    def _verify_funds(self, account_id: str, amount: float) -> str:
        """Verify if sufficient funds are available"""
        # Implementation will be added when mock banking API is created
        return f"Sufficient funds available in account {account_id} for amount {amount}"

    def list_payees(self, context):
        user_id = context.get('user_id')
        access_token = context.get('access_token')
        if not user_id:
            return "Could not determine user ID to list payees. Please make sure you are logged in."
        if not access_token:
            return "Could not determine access token. Please log in again."
        url = "http://localhost:8000/users/payees"
        params = {"user_id": user_id}
        headers = {"Authorization": f"Bearer {access_token}"}
        try:
            print(f"[DEBUG] Requesting payees: url={url} params={params} headers={headers}")
            resp = httpx.get(
                url,
                params=params,
                headers=headers,
                timeout=20
            )
            if resp.status_code == 200:
                payees = resp.json()
                if not payees:
                    return "You have no payees yet. Please add a payee to get started."
                lines = ["Here are your payees:"]
                for p in payees:
                    lines.append(f"- {p['name']} ({p['type']}) | {p.get('email', '')} | {p.get('account_number', '')} | {p.get('bank_name', '')}")
                return "\n".join(lines)
            elif resp.status_code in (204, 404):
                return "You have no payees yet. Please add a payee to get started."
            else:
                return f"Failed to fetch payees: {resp.text}"
        except httpx.ReadTimeout:
            return "Timeout while fetching payees from localhost. Please try again later."
        except Exception as e:
            import traceback
            return f"Error fetching payees: {str(e)}\n{traceback.format_exc()}"

    def add_payee(self, name, type, email, phone, account_number, bank_name, context):
        user_id = context.get('user_id')
        access_token = context.get('access_token')
        if not user_id:
            return "Could not determine user ID to add payee."
        if not access_token:
            return "Could not determine access token. Please log in again."
        try:
            headers = {"Authorization": f"Bearer {access_token}"}
            resp = httpx.post(
                "http://localhost:8000/users/payees",
                params={"user_id": user_id},
                json={
                    "name": name,
                    "type": type,
                    "email": email,
                    "phone": phone,
                    "account_number": account_number,
                    "bank_name": bank_name
                },
                headers=headers,
                timeout=10
            )
            if resp.status_code == 200:
                return f"Payee '{name}' added successfully."
            else:
                return f"Failed to add payee: {resp.text}"
        except Exception as e:
            return f"Error adding payee: {str(e)}"

    def add_account(self, context):
        user_id = context.get('user_id')
        access_token = context.get('access_token')
        if not user_id or not access_token:
            return "You must be logged in to add an account."
        # For demo, just return a message. Implement real logic as needed.
        return "To add an account, please provide the account type, bank name, and account number. (This feature can be expanded to call a backend API.)"

    def list_payments(self, context):
        user_id = context.get('user_id')
        access_token = context.get('access_token')
        if not user_id or not access_token:
            return "You must be logged in to view payments."
        try:
            headers = {"Authorization": f"Bearer {access_token}"}
            resp = httpx.get(
                "http://localhost:8001/payments",
                params={"user_id": user_id},
                headers=headers,
                timeout=10
            )
            if resp.status_code == 200:
                payments = resp.json()
                if not payments:
                    return "You have no payments yet. Make a payment to see it here."
                lines = ["Here are your recent payments:"]
                for p in payments:
                    lines.append(f"- {p.get('date', '')}: ${p.get('amount', 0.0):.2f} to {p.get('payee_name', '')} ({p.get('status', '')})")
                return "\n".join(lines)
            elif resp.status_code in (204, 404):
                return "You have no payments yet. Make a payment to see it here."
            else:
                return f"Failed to fetch payments: {resp.text}"
        except Exception as e:
            return f"Error fetching payments: {str(e)}"

    def list_subscriptions(self, context):
        user_id = context.get('user_id')
        access_token = context.get('access_token')
        if not user_id or not access_token:
            return "You must be logged in to view subscriptions."
        try:
            headers = {"Authorization": f"Bearer {access_token}"}
            resp = httpx.get(
                "http://localhost:8001/subscriptions",
                params={"user_id": user_id},
                headers=headers,
                timeout=10
            )
            if resp.status_code == 200:
                subs = resp.json()
                if not subs:
                    return "You have no subscriptions yet. Set up a recurring payment to see it here."
                lines = ["Here are your subscriptions:"]
                for s in subs:
                    lines.append(f"- {s.get('name', '')}: ${s.get('amount', 0.0):.2f} every {s.get('frequency', '')} (Next: {s.get('next_payment', '')})")
                return "\n".join(lines)
            elif resp.status_code in (204, 404):
                return "You have no subscriptions yet. Set up a recurring payment to see it here."
            else:
                return f"Failed to fetch subscriptions: {resp.text}"
        except Exception as e:
            return f"Error fetching subscriptions: {str(e)}"

    def analyze_transactions(self, context):
        user_id = context.get('user_id')
        access_token = context.get('access_token')
        if not user_id or not access_token:
            return "You must be logged in to analyze transactions."
        try:
            headers = {"Authorization": f"Bearer {access_token}"}
            resp = httpx.get(
                f"http://localhost:8001/transactions",
                params={"user_id": user_id},
                headers=headers,
                timeout=10
            )
            if resp.status_code == 200:
                transactions = resp.json()
                if not transactions:
                    return "You have no transactions to analyze."
                # Simple analysis: total spent, total received
                total_spent = sum(t['amount'] for t in transactions if t.get('type') == 'debit')
                total_received = sum(t['amount'] for t in transactions if t.get('type') == 'credit')
                return f"In your recent transactions, you spent ${total_spent:.2f} and received ${total_received:.2f}."
            else:
                return f"Failed to analyze transactions: {resp.text}"
        except Exception as e:
            return f"Error analyzing transactions: {str(e)}"

    def make_payment(self, account_id, payee_id, amount, description, is_recurring, frequency, end_date, context):
        access_token = context.get('access_token')
        user_id = context.get('user_id')
        if not user_id or not access_token:
            return "Could not determine user ID or access token. Please log in again."
        try:
            payment_data = {
                "account_id": account_id,
                "payee_id": payee_id,
                "amount": amount,
                "description": description,
                "user_id": user_id
            }
            if is_recurring:
                payment_data.update({
                    "is_recurring": True,
                    "frequency": frequency,
                    "end_date": end_date
                })
            headers = {"Authorization": f"Bearer {access_token}"}
            resp = httpx.post(
                "http://localhost:8001/transactions/payment",
                json=payment_data,
                headers=headers,
                timeout=10
            )
            if resp.status_code == 200:
                return "Payment successful!"
            else:
                return f"Payment failed: {resp.text}"
        except Exception as e:
            return f"Error making payment: {str(e)}"

    def list_transactions(self, context, limit=None, user_message=None):
        user_id = context.get('user_id')
        access_token = context.get('access_token')
        if not user_id:
            return "Could not determine user ID to list transactions. Please make sure you are logged in."
        if not access_token:
            return "Could not determine access token. Please log in again."
        try:
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json"
            }
            params = {"user_id": user_id}
            resp = httpx.get(
                f"http://localhost:8001/transactions",
                params=params,
                headers=headers,
                timeout=10
            )
            if resp.status_code == 200:
                try:
                    transactions = resp.json()
                    # Deduplicate by transaction ID
                    seen = set()
                    unique_transactions = []
                    for t in sorted(transactions, key=lambda t: t["timestamp"], reverse=True):
                        if t["id"] not in seen:
                            unique_transactions.append(t)
                            seen.add(t["id"])
                    # Compose LLM prompt
                    prompt = (
                        "You are TaskFin, an intelligent financial assistant. "
                        "The user has requested: '" + (user_message or "Show my transactions") + "'.\n"
                        "Here is the list of all their recent transactions as JSON:\n"
                        f"{unique_transactions}\n"
                        "Please select, filter, and format the transactions according to the user's request. "
                        "Return the result as a markdown table with columns Date, Description, Amount, Status. "
                        "If no transactions match, say so."
                    )
                    # Use the LLM to process and format the result
                    llm_response = self.llm.invoke(prompt)
                    return llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
                except Exception as e:
                    return f"Error processing transactions: {str(e)}"
            elif resp.status_code in (204, 404):
                return "You have no transactions yet. To get started, make a payment and your transactions will appear here."
            else:
                return f"Failed to fetch transactions: {resp.text}"
        except Exception as e:
            return f"Error fetching transactions: {str(e)}"

    def list_accounts(self, context):
        user_id = context.get('user_id')
        access_token = context.get('access_token')
        if not user_id:
            return "Could not determine user ID to list accounts. Please make sure you are logged in."
        if not access_token:
            return "Could not determine access token. Please log in again."
        try:
            headers = {"Authorization": f"Bearer {access_token}"}
            resp = httpx.get(
                "http://localhost:8001/accounts",
                params={"user_id": user_id},
                headers=headers,
                timeout=10
            )
            if resp.status_code == 200:
                accounts = resp.json()
                if not accounts:
                    return "You have no accounts yet. Please add an account to get started."
                lines = ["Here are your accounts:"]
                for a in accounts:
                    lines.append(f"- {a['name']} ({a['type']}) | {a['balance']:.2f}")
                return "\n".join(lines)
            elif resp.status_code in (204, 404):
                return "You have no accounts yet. Please add an account to get started."
            else:
                return f"Failed to fetch accounts: {resp.text}"
        except Exception as e:
            return f"Error fetching accounts: {str(e)}"

    def remove_account(self, account_id, context):
        user_id = context.get('user_id')
        access_token = context.get('access_token')
        if not user_id or not access_token:
            return "You must be logged in to remove an account."
        try:
            headers = {"Authorization": f"Bearer {access_token}"}
            resp = httpx.delete(
                f"http://localhost:8001/accounts/{account_id}",
                params={"user_id": user_id},
                headers=headers,
                timeout=10
            )
            if resp.status_code == 200:
                return f"Account {account_id} removed successfully."
            elif resp.status_code in (204, 404):
                return f"Account {account_id} not found."
            else:
                return f"Failed to remove account: {resp.text}"
        except Exception as e:
            return f"Error removing account: {str(e)}"

    def update_account(self, account_id, update_fields, context):
        user_id = context.get('user_id')
        access_token = context.get('access_token')
        if not user_id or not access_token:
            return "You must be logged in to update an account."
        try:
            headers = {"Authorization": f"Bearer {access_token}"}
            resp = httpx.put(
                f"http://localhost:8001/accounts/{account_id}",
                params={"user_id": user_id},
                json=update_fields,
                headers=headers,
                timeout=10
            )
            if resp.status_code == 200:
                return f"Account {account_id} updated successfully."
            elif resp.status_code in (204, 404):
                return f"Account {account_id} not found."
            else:
                return f"Failed to update account: {resp.text}"
        except Exception as e:
            return f"Error updating account: {str(e)}"

    def get_account_details(self, account_id, context):
        user_id = context.get('user_id')
        access_token = context.get('access_token')
        if not user_id or not access_token:
            return "You must be logged in to view account details."
        try:
            headers = {"Authorization": f"Bearer {access_token}"}
            resp = httpx.get(
                f"http://localhost:8001/accounts/{account_id}",
                params={"user_id": user_id},
                headers=headers,
                timeout=10
            )
            if resp.status_code == 200:
                acc = resp.json()
                return f"Account {account_id}: {acc.get('type', '')} at {acc.get('bank_name', '')}, Balance: ${acc.get('balance', 0.0):.2f}"
            elif resp.status_code in (204, 404):
                return f"Account {account_id} not found."
            else:
                return f"Failed to fetch account details: {resp.text}"
        except Exception as e:
            return f"Error fetching account details: {str(e)}"

    def summarize_spending_by_category(self, context):
        user_id = context.get('user_id')
        access_token = context.get('access_token')
        if not user_id or not access_token:
            return "You must be logged in to analyze spending."
        try:
            headers = {"Authorization": f"Bearer {access_token}"}
            resp = httpx.get(
                f"http://localhost:8001/transactions",
                params={"user_id": user_id},
                headers=headers,
                timeout=10
            )
            if resp.status_code == 200:
                transactions = resp.json()
                if not transactions:
                    return "You have no transactions to analyze."
                # Group by category
                category_totals = {}
                for t in transactions:
                    cat = t.get('category', 'Uncategorized')
                    amt = t.get('amount', 0.0)
                    if t.get('type') == 'debit':
                        category_totals[cat] = category_totals.get(cat, 0.0) + amt
                if not category_totals:
                    return "No spending data by category found."
                lines = ["Spending by category:"]
                for cat, amt in category_totals.items():
                    lines.append(f"- {cat}: ${amt:.2f}")
                return "\n".join(lines)
            else:
                return f"Failed to analyze spending: {resp.text}"
        except Exception as e:
            return f"Error analyzing spending: {str(e)}"

    def get_upcoming_bills(self, context):
        user_id = context.get('user_id')
        access_token = context.get('access_token')
        if not user_id or not access_token:
            return "You must be logged in to view upcoming bills."
        try:
            headers = {"Authorization": f"Bearer {access_token}"}
            resp = httpx.get(
                f"http://localhost:8001/bills/upcoming",
                params={"user_id": user_id},
                headers=headers,
                timeout=30
            )
            if resp.status_code == 200:
                bills = resp.json()
                if not bills:
                    return "You have no upcoming bills."
                lines = ["Your upcoming bills:"]
                for b in bills:
                    lines.append(f"- {b.get('name', '')}: ${b.get('amount', 0.0):.2f} due {b.get('due_date', '')}")
                return "\n".join(lines)
            elif resp.status_code in (204, 404):
                return "You have no upcoming bills."
            else:
                return f"Failed to fetch upcoming bills: {resp.text}"
        except Exception as e:
            return f"Error fetching upcoming bills: {str(e)}"

if __name__ == "__main__":
    import sys
    import os
    import httpx
    # Example usage: python financial_agent.py <user_id> <access_token>
    user_id = sys.argv[1] if len(sys.argv) > 1 else "1"
    access_token = sys.argv[2] if len(sys.argv) > 2 else ""
    url = "http://localhost:8000/users/payees"
    params = {"user_id": user_id}
    headers = {"Authorization": f"Bearer {access_token}", "User-Agent": "curl/7.64.1"}
    print(f"[TEST] Requesting payees: url={url} params={params} headers={headers}")
    try:
        resp = httpx.get(url, params=params, headers=headers, timeout=10)
        print(f"[TEST] Status: {resp.status_code}")
        print(f"[TEST] Body: {resp.text}")
    except Exception as e:
        print(f"[TEST] Exception: {e}") 