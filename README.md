# BCI Project

Project developed as part of **Praktikum: Developing Reliable Decoders for a Brain-Computer-Interface**

This project allows for the acquisition and processing of Brain-Computer Interface data. It is constructed using the modern Python package manager **[uv](https://github.com/astral-sh/uv)** and adheres to a `src` layout structure.

This layout allows you to import your code as a package (e.g., `import bci.acquisition`) anywhere in the project without complex path hacks.

## 📂 Project Structure

* **`src/bci`**: The main source code package.
* **`pyproject.toml`**: The configuration file managing dependencies and build settings.
* **`uv.lock`**: The lockfile ensuring reproducible installations.
* **`resources/configs`**: Directory containing configuration files.
* **`data`**: Directory containing data files.
* **`resources/game_assets/dino`**: Directory containing the assets for the dino game. !! Make sure to copy them there !!

## 🚀 Getting Started

To get started with the project, follow these steps:
1. Install **uv** https://docs.astral.sh/uv/getting-started/
2. **Sync** the project dependencies:
   ```bash
   uv sync
   ```

## 🎯 Running the Project

**Offline**

1. First, make sure to adapt the config file located at `src/bci/config/config.yaml` to your needs.
2. Run the offline acquisition script from the project root: `<run> src/bci/main_offline.py`
*Note*: It might take longer at the beginning until all Physionet data is downloaded.

**Online - No Dino Game**

1. First, make sure to adapt the config file located at `src/bci/config/config.yaml` to your needs.
2. You need a trained model and a AR threshold to run the online acquisition. You can train a model using the offline acquisition first.
3. First you need a stream of data. Run the replay script from the project root: `<run> src/bci/replay.py`. This will send data to a LSL stream from the test file selected in the config file.
4. Run the online acquisition script from the project root: `<run> src/bci/main_online.py`
*Note*: You can close the online acquisition at any time using Ctrl + C. (Intermediary results will be saved.)

**Online - Dino Game**
1. First, make sure to adapt the config file located at `src/bci/config/config.yaml` to have online_mode set to "dino".
2. Run the dino game script from the project root: `<run> src/bci/Game/DinoGamev2.py`
3. Same as the "Online - No Dino Game" instructions above (2-4).
