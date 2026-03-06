# lifebot

LifeBot is an offline rescue swarm simulation. It uses a local LLM (via Ollama) to generate concise decision rationales while controlling a simulated drone fleet through an MCP-style tool interface. The mission runs on a 2D grid with battery, scanning, and survivor detection logic.

## What it uses
- Python 3
- Local LLM via Ollama (default: mistral:7b-instruct)
- MCP-style tool server for drone control
- Grid-based simulation with drones and survivors

## Run locally
1. Start Ollama with the Mistral model:

   ```bash
   ollama run mistral
   ```

2. Run the offline mission:

   ```bash
   python main.py
   ```
