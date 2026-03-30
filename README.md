# Cell Architecture

This project explores a cell-based architecture for managing workers in a scalable system.  
The idea is inspired by biological cells, where independent units communicate and coordinate to perform complex tasks.

The system focuses on:
- organizing workers into structured cells
- handling communication between components
- monitoring performance metrics
- comparing fixed vs dynamically spawned workers

The goal is to understand how system design decisions affect performance and scalability.

---

## Structure

Main components:

- base_cell.py → defines the basic behavior of a cell
- cell_network.py → manages communication between cells
- message_router.py → routes messages to appropriate workers
- spawn_manager.py → handles creation of workers
- monitor.py → tracks metrics of workers
- trainer.py → runs the architecture

The project also generates performance graphs:

- dynamic_worker_metrics.png
- fixed_worker_metrics.png

---

## How to run

Clone the repo:

git clone https://github.com/your-username/cell-architecture.git

Move into folder:

cd cell-architecture

Create virtual environment:

python -m venv venv

Activate environment:

Windows:
venv\Scripts\activate

Mac/Linux:
source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Run:

python trainer.py

---

## Why I made this

I wanted to experiment with a modular system design where different components act independently but still work together efficiently.

This project helped me understand:
- worker coordination
- system scalability
- modular design patterns
- performance monitoring

---

## Future improvements

- better scaling logic
- async communication
- improved monitoring
- visualization dashboard
