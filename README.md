# MedOryx - Adverse Drug Events Explorer

An adverse drug events explorer Streamlit app powered by the OnSIDES database, Haystack, and MCP-Alchemy.


## Overview

This application leverages large language models and the OnSIDES database to provide a natural language interface for exploring adverse drug events. MedOryx can help you:

- Query adverse drug events (ADEs) for specific medications
- Find drugs that cause particular side effects
- Compare adverse events across drug classes
- Explore serious ADEs from drug labels' Boxed Warnings section
- Access comprehensive adverse event data extracted from FDA drug labels

## Technologies

- **[OnSIDES](https://github.com/tatonetti-lab/onsides)**: Comprehensive database of adverse drug events extracted from drug labels
- **[MCP-Alchemy](https://github.com/runekaagaard/mcp-alchemy)**: MCP Server for Text-to-SQL
- **[Haystack](https://github.com/deepset-ai/haystack)**: AI orchestration framework
- **[Streamlit](https://streamlit.io/)**: Web app framework

## Setup

### Prerequisites

- Python 3.11+
- Anthropic API key
- [uv](https://github.com/astral-sh/uv) - Modern Python package manager

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/oryx1729/med-oryx-ade
   cd med-oryx-ade
   ```

2. Install dependencies with uv:
   ```
   uv pip install .
   ```

3. Set your Anthropic API key as an environment variable:
   ```
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

4. Set the SQLAlchemy database URL for MCP-Alchemy:
   ```
   export DB_URL=postgresql://user:password@localhost/dbname
   ```

## Usage

1. Start the application:
   ```
   uv run streamlit run app.py
   ```

2. Open your browser and navigate to the provided URL (typically http://localhost:8501)

3. Enter your questions about adverse drug events in the chat interface

### Example Questions

The agent can answer questions like:
- What are the most common side effects of metformin?
- Show me all cardiovascular adverse events associated with statins.
- Which antidepressants have the fewest reported gastrointestinal side effects?
- List all drugs that can cause severe liver toxicity.
- Compare the adverse events of amoxicillin and azithromycin.
- What pediatric-specific adverse events are reported for methylphenidate?
- Show me drugs with black box warnings for QT prolongation.
- Which antipsychotics have the highest risk of weight gain?
