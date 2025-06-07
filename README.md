# TaskFin

TaskFin is an intelligent, modular financial assistant platform. It features multiple agents (Financial, Security, Orchestrator) and a Streamlit-based user interface.

## Requirements

- Python 3.8+
- [Anthropic API Key](https://console.anthropic.com/settings/keys)
- Streamlit

## Setup

1. Clone the repository and create a virtual environment:

    ```sh
    git clone https://github.com/NikhilDeekonda77/TaskFin.git
    cd TaskFin
    python3 -m venv venv
    source venv/bin/activate
    ```

2. Install dependencies:

    ```sh
    pip install -r requirements.txt
    ```

3. Create a `.env` file in the project root:

    ```
    ANTHROPIC_API_KEY=sk-ant-XXXXXXXXXXXXXXXXXXXXXXXXXXXX
    ```

---

## Running the Agents

**Replace `<entrypoint.py>` and `<port>` with your actual filenames and port numbers.**

- Financial Agent:
    ```sh
    uvicorn <financial_entrypoint.py>:app --port <financial_port> --reload
    ```
- Security Agent:
    ```sh
    uvicorn <security_entrypoint.py>:app --port <security_port> --reload
    ```
- Orchestrator Agent:
    ```sh
    uvicorn <orchestrator_entrypoint.py>:app --port <orchestrator_port> --reload
    ```

## Running the Streamlit Interface

Find your main Streamlit app file 

```sh
streamlit run app.py
```

---

## Project Structure

- `agents/` - Agent logic (financial, orchestrator, security)
- `orchestration/` - API, core logic, models, schemas, services, UI
- `frontend/` - (Optional) Frontend app
- `shared/` - Shared utilities, models, schemas
- `mock_services/` - Mock authentication and banking services
- `data/` - Data and synthetic data
- `tests/` - Test cases

---

## Environment Variables

- `ANTHROPIC_API_KEY`: Your Anthropic Claude API key (required)

---

## Security

- API keys and secrets must **never** be committed to git.
- The `.env` file is git-ignored by default.

---

