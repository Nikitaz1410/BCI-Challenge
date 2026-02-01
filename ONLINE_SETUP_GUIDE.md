# How to Run Online Adaptive LDA

## Prerequisites

1. **Trained Model**: You need a trained Adaptive LDA model
   - Location: `resources/models/adaptivelda_model.pkl`
   - Train it first using `main_offline_AdaptiveLDA.ipynb`

2. **Artifact Rejection Object**: Saved artifact rejection thresholds
   - Location: `resources/models/adaptivelda_artefact_removal.pkl`
   - Created during offline training

3. **Active LSL Streams**: You need two LSL streams running:
   - **EEG Stream**: Your EEG data stream
   - **Markers Stream**: Labels/markers stream
     - For dino game: `MyDinoGameMarkerStream`
     - For prerecorded: `Labels_Stream`

## Option 1: Using Jupyter Notebook (Recommended for Testing)

### Steps:

1. **Open the notebook**:
   ```bash
   jupyter notebook main_online_AdaptiveLDA.ipynb
   ```

2. **Run cells sequentially**:
   - **Cell 1-2**: Imports and setup (auto-installs `pylsl` if needed)
   - **Cell 3-4**: Load configuration and markers
   - **Cell 5**: Load trained model and artifact rejection
   - **Cell 6**: Initialize preprocessing and controller
   - **Cell 7**: Connect to LSL streams (waits 5 seconds to find streams)
   - **Cell 8**: Start online classification loop ⚠️ **Runs continuously**
   - **Cell 9**: Display statistics (after stopping)
   - **Cell 10**: Visualize accuracy over time

3. **To Stop**: Use the **Stop** button in Jupyter or interrupt the kernel (`Ctrl+C` or `I, I`)

4. **View Results**: After stopping, run cells 9-10 to see statistics and plots

## Option 2: Using Python Script

### Steps:

1. **Ensure LSL streams are active** (from replay or real EEG system)

2. **Run the script**:
   ```bash
   cd "/Users/amalbenslimen/BCI Challenge /BCI-Challenge"
   python -m src.bci.main_online_AdaptiveLDA
   ```
   Or directly:
   ```bash
   python src/bci/main_online_AdaptiveLDA.py
   ```

3. **To Stop**: Press `Ctrl+C` (results will be saved automatically)

## Configuration

Make sure your `bci_config.yaml` is set correctly:

```yaml
online: "dino"  # or "prerecorded" for testing without game
classification_threshold: 0.6  # Minimum confidence to accept prediction
ip: "127.0.0.1"  # IP for UDP communication to game
port: 5005  # Port for UDP communication
```

## Running with Replay Data (Testing)

If you want to test without real EEG hardware:

1. **Start replay script** (sends prerecorded data to LSL):
   ```bash
   python src/bci/replay.py
   ```
   Or check if there's a replay script for AdaptiveLDA specifically.

2. **Then run online script** (it will receive data from replay)

## Running with Dino Game

1. **Set config to dino mode**:
   ```yaml
   online: "dino"
   ```

2. **Start the Dino Game**:
   ```bash
   python src/bci/game/DinoGamev2.py
   ```

3. **Run the online script/notebook** (it will connect to game's marker stream)

## What Happens During Online Classification

1. **Data Acquisition**: Continuously receives EEG chunks from LSL stream
2. **Buffering**: Maintains sliding window of recent data
3. **Filtering**: Applies online bandpass filter
4. **Classification**: Extracts PSD features and predicts class
5. **Adaptation**: After each trial (when label changes), model updates its parameters
6. **Game Control**: Sends commands via UDP to game (if configured)
7. **Statistics**: Tracks accuracy, latency, and adaptation count

## Troubleshooting

### "Could not find EEG or Markers streams"
- Make sure LSL streams are running
- Check stream names match what's expected
- For dino game: stream should be named "MyDinoGameMarkerStream"
- Wait longer: script waits 5 seconds by default

### "Model not found"
- Train the model first using `main_offline_AdaptiveLDA.ipynb`
- Check that model is saved at: `resources/models/adaptivelda_model.pkl`

### "Artifact rejection file not found"
- This is optional - the script will continue without it
- But artifact rejection won't be applied online
- Train offline to generate this file

### Connection Issues
- Check IP and port settings in config match game/server
- Verify firewall isn't blocking UDP communication

## Expected Output

During classification, you'll see:
```
Label: 1 (rest) | Predicted: 0 (rest) | Conf: 85.23% | Adaptations: 3
🔄 Adapted model (Trial ended: rest → left_hand)
📊 Rolling accuracy (last 20): 72.50%
```

After stopping (`Ctrl+C`):
```
============================================================
FINAL STATISTICS
============================================================
Avg time per loop: 12.45 ms
Total Predictions: 150
  Rejected: 15
  Accepted Successes: 110
  Accepted Fails: 25
Total Adaptations: 12
Final Accuracy (accepted predictions): 81.48%
Overall Accuracy (all predictions): 73.33%
```

Plus a saved accuracy plot: `adaptive_lda_online_accuracy.png`
