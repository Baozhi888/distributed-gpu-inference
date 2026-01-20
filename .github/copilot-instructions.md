# AI Coding Agent Instructions for Distributed GPU Inference Platform

## Architecture Overview
This is a distributed GPU inference system with three main components:
- **Server**: Central FastAPI coordinator managing jobs, workers, and scheduling
- **Workers**: GPU-enabled nodes executing inference tasks (LLM, image generation)
- **SDK**: Python client library for job submission

**Key Design Decisions:**
- Workers are "dumb terminals" - all decisions made by server
- Configuration loaded from server at runtime
- Async-first architecture using FastAPI and async SQLAlchemy
- Multi-region deployment with geo-aware scheduling

## Critical Workflows

### Server Setup
```bash
cd server
pip install -r requirements.txt
alembic upgrade head  # Initialize database
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Worker Deployment (Recommended)
```bash
npx gpu-worker  # Interactive setup with auto Python venv management
# OR
npm install -g gpu-worker
gpu-worker configure  # Interactive config wizard
gpu-worker start
```

### Full Stack Development
```bash
docker-compose up --build  # Spins up server + postgres + redis
```

### Worker CLI Operations
```bash
gpu-worker status     # Check worker health
gpu-worker set load_control.acceptance_rate 0.8  # Runtime config changes
```

## Project-Specific Patterns

### Configuration Hierarchy
Environment variables override YAML config override defaults:
```python
# From config.py - check WorkerConfig class
config = load_config()  # Merges .env > config.yaml > defaults
```

### Engine Implementation
All inference engines inherit from `BaseEngine`:
```python
class LLMEngine(BaseEngine):
    def load_model(self):  # GPU memory management
    def inference(self, params):  # Task execution
    def unload_model(self):  # Cleanup
```

### API Communication
Workers use signed requests with token auth:
```python
# From api_client.py
headers = self._headers(body, path)  # Auto-signs requests
response = self.client.post(f"{self.base_url}{path}", json=data, headers=headers)
```

### Database Models
Use SQLAlchemy async patterns:
```python
# From models/job.py
class Job(Base):
    status: Mapped[JobStatus]  # Enum-based status tracking
```

### Error Handling
Follow server-side validation with HTTPException:
```python
# From api/jobs.py
if not job:
    raise HTTPException(status_code=404, detail="Job not found")
```

## Integration Points

### External Dependencies
- **HuggingFace Transformers**: Model loading with `trust_remote_code=True`
- **PostgreSQL + Redis**: Async database connections
- **Docker Compose**: Multi-service orchestration

### Cross-Component Communication
- Server pushes config to workers via `/api/v1/workers/{id}/config`
- Workers heartbeat every 30s with GPU metrics
- Direct mode: P2P connections bypass server for low-latency tasks

### Security Patterns
- Token rotation with refresh tokens
- HMAC request signing for worker-server auth
- SSL verification configurable per environment

## Development Conventions

### Code Style
- Type hints mandatory for all functions
- Async/await for I/O operations
- Chinese comments, English code/variables
- Logging with `logger.info()` for key operations

### Testing Approach
- Integration tests via API endpoints
- Mock external services for unit tests
- GPU availability checks before inference

### Deployment
- Docker for containerization
- Environment-specific configs (.env files)
- Health checks for service dependencies

## Common Pitfalls
- Always check GPU memory before model loading
- Handle token expiration with automatic refresh
- Use absolute paths for file operations in workers
- Validate config merging order (env > yaml > defaults)