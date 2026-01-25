# Market Analyst Agent

> ⚠️ **DISCLAIMER**: This is a **beginner demo project for educational purposes only**. Do NOT use this for actual trading or investment decisions. The analysis generated is for learning about agentic AI patterns and should not be considered financial advice.

An **Autonomous Investment Research Agent** demonstrating production-ready agentic patterns from the **"Engineering the Agentic Stack"** blog series.

This project showcases how to build a sophisticated AI agent that can research stocks, analyze market data, and generate investment reports—while maintaining state, respecting human oversight, and running reliably in production environments.

## Why This Project?

| Challenge | How We Solve It |
|-----------|----------------|
| **Complex Reasoning** | Researches multiple sources (ReAct) and synthesizes reports (Plan-and-Execute) |
| **Persistence** | Remembers your portfolio (long-term) and research state (short-term) |
| **Tool Use** | Interacts with external APIs (YFinance, Tavily) through well-designed interfaces |
| **Human Oversight** | Requires approval before publishing reports or recommendations |
| **Production Readiness** | Survives crashes, handles long-running tasks, containerized deployment |

## Features

- 🧠 **Plan-and-Execute Architecture**: Breaks down research into structured steps
- 🔄 **ReAct Execution**: Thought-Action-Observation loop for each step
- 💾 **PostgreSQL Checkpointing**: Pause and resume mid-analysis
- 🧑‍💼 **User Profiles**: Redis-backed long-term memory for preferences
- ✋ **Human-in-the-Loop**: Approval required before publishing reports
- 🐳 **Containerized**: Docker Compose for production deployment

---

## Getting Started

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager
- Docker & Docker Compose (for persistence features)

### Step 1: Get Your API Keys

You'll need two API keys to run this project:

#### Anthropic API Key (Claude)

1. Go to [console.anthropic.com](https://console.anthropic.com/)
2. Sign up or log in to your account
3. Navigate to **API Keys** in the left sidebar
4. Click **Create Key** and copy the generated key
5. Save this as `ANTHROPIC_API_KEY`

> **Note**: Anthropic requires a payment method. New accounts typically get $5 in free credits.

#### Tavily API Key (Web Search)

1. Go to [tavily.com](https://tavily.com/)
2. Sign up for a free account
3. Navigate to your [API Keys dashboard](https://app.tavily.com/home)
4. Copy your API key
5. Save this as `TAVILY_API_KEY`

> **Note**: Tavily's free tier includes 1,000 API calls/month—plenty for development.

### Step 2: Install Dependencies

```bash
# Clone and enter the directory
cd market-analyst-agent

# Install dependencies with uv
uv sync
```

### Step 3: Configure Environment Variables

```bash
# Copy the environment template
cp .env.example .env

# Open .env and add your API keys
```

Edit the `.env` file with your API keys:

```bash
# Required: Your API keys
ANTHROPIC_API_KEY=sk-ant-api03-xxxxxxxxxxxxx
TAVILY_API_KEY=tvly-xxxxxxxxxxxxx

# PostgreSQL connection (defaults work with docker-compose)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=market_analyst
POSTGRES_USER=analyst
POSTGRES_PASSWORD=analyst_pass

# Redis connection (defaults work with docker-compose)
REDIS_HOST=localhost
REDIS_PORT=6379
```

### Step 4: Set Up PostgreSQL and Redis

The agent uses **PostgreSQL** for checkpointing (pause/resume) and **Redis** for user profile memory. You have three options:

#### Option A: Using Docker Compose (Recommended)

The easiest way to run both services:

```bash
# Start PostgreSQL and Redis in the background
docker compose -f docker/docker-compose.yml --env-file .env up -d postgres redis

# Verify services are running
docker compose -f docker/docker-compose.yml --env-file .env ps
```

Both services will be available at their default ports (`localhost:5432` for PostgreSQL, `localhost:6379` for Redis).

#### Option B: Using Standalone Docker Containers

If you prefer running containers separately:

```bash
# Start PostgreSQL
docker run -d \
  --name market-analyst-postgres \
  -e POSTGRES_DB=market_analyst \
  -e POSTGRES_USER=analyst \
  -e POSTGRES_PASSWORD=analyst_pass \
  -p 5432:5432 \
  -v market_analyst_pgdata:/var/lib/postgresql/data \
  postgres:16-alpine

# Start Redis
docker run -d \
  --name market-analyst-redis \
  -p 6379:6379 \
  -v market_analyst_redis:/data \
  redis:7-alpine
```

#### Option C: Using Native Installations

**macOS (Homebrew):**

```bash
# Install and start PostgreSQL
brew install postgresql@16
brew services start postgresql@16

# Create the database and user
createdb market_analyst
psql market_analyst -c "CREATE USER analyst WITH PASSWORD 'analyst_pass';"
psql market_analyst -c "GRANT ALL PRIVILEGES ON DATABASE market_analyst TO analyst;"
psql market_analyst -c "GRANT ALL ON SCHEMA public TO analyst;"

# Install and start Redis
brew install redis
brew services start redis
```

**Ubuntu/Debian:**

```bash
# Install PostgreSQL
sudo apt update
sudo apt install postgresql postgresql-contrib

# Create database and user
sudo -u postgres psql -c "CREATE DATABASE market_analyst;"
sudo -u postgres psql -c "CREATE USER analyst WITH PASSWORD 'analyst_pass';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE market_analyst TO analyst;"

# Install Redis
sudo apt install redis-server
sudo systemctl start redis-server
```

---

## Running the Agent

### Quick Test (No Persistence)

Run without PostgreSQL/Redis to test basic functionality:

```bash
uv run python -m market_analyst.cli "Analyze NVDA stock" --no-persist
```

### Full Mode (With Persistence)

With PostgreSQL and Redis running:

```bash
uv run python -m market_analyst.cli "Analyze NVDA stock"
```

### Model Selection

Choose between models based on your needs:

```bash
# Use Haiku (faster, cheaper - good for testing)
uv run python -m market_analyst.cli "Analyze NVDA stock" --model haiku

# Use Sonnet (default, more powerful - better analysis)
uv run python -m market_analyst.cli "Analyze NVDA stock" --model sonnet
```

### Using Docker (All-in-One)

Run everything in containers:

```bash
# Start all services including the app
docker compose -f docker/docker-compose.yml up --build

# In another terminal, run an analysis
docker compose -f docker/docker-compose.yml exec app python -m market_analyst.cli "Analyze NVDA stock"
```

---

## Usage Examples

### Basic Analysis

```bash
# Analyze a stock
uv run market-analyst "Analyze NVDA for investment potential"

# The agent will:
# 1. Create a research plan
# 2. Execute each step using tools
# 3. Generate a draft report
# 4. Pause for your approval
```

### Set User Profile

```bash
# Set risk tolerance (persists in Redis)
uv run market-analyst --set-profile --risk-tolerance conservative --horizon long

# Future analyses will consider this profile
uv run market-analyst "Analyze AAPL stock"
```

### Pause and Resume

```bash
# Start an analysis
uv run market-analyst "Deep analysis of semiconductor sector"
# ... analysis running ...
# Press Ctrl+C to pause

# Resume later
uv run market-analyst --resume --thread-id <thread-id>
```

### Approve Reports

```bash
# When a report is ready, it pauses for approval
uv run market-analyst --approve --thread-id <thread-id>
```

---

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Planner   │────▶│  Executor   │────▶│  Reporter   │
│             │     │  (ReAct)    │     │             │
└─────────────┘     └──────┬──────┘     └──────┬──────┘
                           │                    │
                           ▼                    ▼
                     ┌───────────┐        ┌───────────┐
                     │   Tools   │        │   HITL    │
                     │ (YFinance │        │ (Approval)│
                     │  Tavily)  │        └───────────┘
                     └───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │     PostgreSQL         │
              │   (Checkpointing)      │
              └────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │        Redis           │
              │   (User Profiles)      │
              └────────────────────────┘
```

---

## Blog Series Mapping

This project demonstrates concepts from each part of the blog series:

| Blog Post | Demo Feature |
|-----------|--------------|
| Part 1: Cognitive Engine | Plan-and-Execute + ReAct in `nodes/` |
| Part 2: The Cortex | PostgreSQL checkpointing + Redis profiles |
| Part 3: Tool Ergonomics | Pydantic-validated tools in `tools/` |
| Part 4: Human-in-the-Loop | `interrupt_before` on report publishing |
| Part 5: Production | Docker Compose deployment |

---

## Configuration Reference

All environment variables (set in `.env`):

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | ✅ | Anthropic API key for Claude | - |
| `TAVILY_API_KEY` | ✅ | Tavily API key for web search | - |
| `POSTGRES_HOST` | ❌ | PostgreSQL host | `localhost` |
| `POSTGRES_PORT` | ❌ | PostgreSQL port | `5432` |
| `POSTGRES_DB` | ❌ | PostgreSQL database name | `market_analyst` |
| `POSTGRES_USER` | ❌ | PostgreSQL username | `analyst` |
| `POSTGRES_PASSWORD` | ❌ | PostgreSQL password | `analyst_pass` |
| `REDIS_HOST` | ❌ | Redis host | `localhost` |
| `REDIS_PORT` | ❌ | Redis port | `6379` |

---

## Troubleshooting

### PostgreSQL Connection Issues

```bash
# Check if PostgreSQL is running
docker compose -f docker/docker-compose.yml ps postgres

# Check logs
docker compose -f docker/docker-compose.yml logs postgres

# Test connection
psql -h localhost -U analyst -d market_analyst -c "SELECT 1;"
```

### Redis Connection Issues

```bash
# Check if Redis is running
docker compose -f docker/docker-compose.yml ps redis

# Test connection
redis-cli ping  # Should return "PONG"
```

### API Key Issues

- **Anthropic**: Ensure your key starts with `sk-ant-`
- **Tavily**: Ensure your key starts with `tvly-`

---

## License

MIT
