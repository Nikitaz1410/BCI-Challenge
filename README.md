# BCI Project

This project allows for the acquisition and processing of Brain-Computer Interface data. It is constructed using the modern Python package manager **[uv](https://github.com/astral-sh/uv)** and adheres to a `src` layout structure.

This layout allows you to import your code as a package (e.g., `import bci.acquisition`) anywhere in the project without complex path hacks.

## 📂 Project Structure

* **`src/bci`**: The main source code package.
* **`pyproject.toml`**: The configuration file managing dependencies and build settings.
* **`uv.lock`**: The lockfile ensuring reproducible installations.

## 🚀 Getting Started

To get started with the project, follow these steps:
1. Install **uv** https://docs.astral.sh/uv/getting-started/
2. **Sync** the project dependencies:
   ```bash
   uv sync
   ```

## 🎯 Running the Project
To run the riemann example offline application, use the following command:
   ```bash
    uv run src/bci/example_riemann_offline.py
   ```
   This will execute the example script located in the `src/bci` directory, demonstrating the offline processing of BCI data using Riemannian geometry methods.
  
To stream data in a simulated online manner, first start the replay server:
   ```bash
    uv run src/bci/replay.py
   ```

In another terminal, run the online processing script:
   ```bash
    uv run src/bci/example_riemann_online.py
   ```
   This will execute the online example script, which processes data in real-time as it is streamed
