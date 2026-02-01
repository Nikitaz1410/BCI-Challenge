# Testing Guide: Online Adaptive LDA

This guide shows you how to test the online Adaptive LDA classifier on your machine using prerecorded data.

## Prerequisites Checklist

✅ **Trained Model**: `resources/models/adaptivelda_model.pkl` must exist
- If not, train it first: Run `main_offline_AdaptiveLDA.ipynb`

✅ **Artifact Rejection**: `resources/models/adaptivelda_artefact_removal.pkl` (optional but recommended)

✅ **Dependencies**: `pylsl` must be installed
- The script will work if you've run the notebook, which auto-installs it

✅ **Test Data**: Available data for replay
- Either Physionet data (e.g., subject 42) OR your own recorded data

## Step-by-Step Testing Instructions

### Step 1: Configure for Testing

Edit `resources/configs/bci_config.yaml`:

```yaml
online: "prerecorded"  # Change from "dino" to "prerecorded" for testing
replay_subject_id: "Phy-42"  # Use "Phy-42" for Physionet or "sub-P999" for your data
```

**Note**: The `online` setting determines which marker stream name to look for:
- `"prerecorded"` → looks for `"Labels_Stream"` (from replay.py)
- `"dino"` → looks for `"MyDinoGameMarkerStream"` (from game)

### Step 2: Start Replay Stream (Terminal 1)

This simulates real-time EEG data by streaming prerecorded data:

```bash
cd "/Users/amalbenslimen/BCI Challenge /BCI-Challenge"
python -m src.bci.replay
```

Or:
```bash
python src/bci/replay.py
```

**What this does:**
- Loads data from Physionet or your test data
- Creates two LSL streams:
  - `EEG_Stream` (type: "EEG")
  - `Labels_Stream` (type: "Markers")
- Streams data chunk by chunk at real-time speed

**Keep this terminal running!** The replay will stream until all data is sent.

### Step 3: Run Online Classification (Terminal 2)

In a **new terminal**, run:

```bash
cd "/Users/amalbenslimen/BCI Challenge /BCI-Challenge"
python -m src.bci.main_online_AdaptiveLDA
```

Or:
```bash
python src/bci/main_online_AdaptiveLDA.py
```

**What happens:**
1. ✅ Loads configuration
2. ✅ Loads trained model
3. ✅ Looks for LSL streams (waits up to 5 seconds)
4. ✅ Connects to EEG and Labels streams
5. 🟢 Starts real-time classification loop
6. 📊 Adapts model after each trial
7. 📈 Tracks accuracy and statistics

**To stop:** Press `Ctrl+C` (results saved automatically)

### Step 4: View Results

After stopping (`Ctrl+C`), you'll see:

```
============================================================
STOPPING ONLINE PROCESSING
============================================================
Avg time per loop: 12.45 ms
Total Predictions: 150
  Rejected: 15
  Accepted Successes: 110
  Accepted Fails: 25
Total Adaptations: 12

Final Accuracy (accepted predictions): 81.48%
Overall Accuracy (all predictions): 73.33%
Total labeled trials: 135

✓ Accuracy plot saved: adaptive_lda_online_accuracy.png
============================================================
```

Plus an accuracy plot saved as `adaptive_lda_online_accuracy.png`

## Quick Test Script

Here's a simple test you can run to verify everything works:

```bash
#!/bin/bash
# Quick test script

# Terminal 1: Start replay (background)
python -m src.bci.replay &
REPLAY_PID=$!

# Wait for streams to initialize
sleep 3

# Terminal 2: Run classification
python -m src.bci.main_online_AdaptiveLDA

# Clean up
kill $REPLAY_PID 2>/dev/null
```

## Troubleshooting

### ❌ "Could not find EEG stream"

**Problem**: Replay script not running or streams not ready

**Solution**: 
- Make sure replay.py is running first
- Wait a few seconds for LSL streams to initialize
- Check that streams are visible: `python -m pylsl.streams_inlet` (if available)

### ❌ "Model not found"

**Problem**: Model hasn't been trained yet

**Solution**: 
- Train model first: Run `main_offline_AdaptiveLDA.ipynb`
- Check that `resources/models/adaptivelda_model.pkl` exists

### ❌ "Could not find Markers stream"

**Problem**: Wrong `online` setting or replay not sending labels

**Solution**:
- Check `config.online` is set to `"prerecorded"` (not `"dino"`)
- Verify replay.py is creating `"Labels_Stream"` (check output)
- Make sure replay has data loaded (check for errors)

### ⚠️ Labels mismatch

**Problem**: Model uses 0-indexed labels (0, 1, 2) but replay sends 1-indexed (1, 2, 3)

**Solution**: 
- The script handles this automatically
- If issues occur, check that replay.py is using correct label mapping

### 📊 Low accuracy

**Possible reasons**:
- Model not trained on similar data
- Data quality issues in replay
- Artifact removal too aggressive
- Check the accuracy plot for patterns

## Testing with Different Data Sources

### Option 1: Physionet Data
```yaml
replay_subject_id: "Phy-42"  # Any Physionet subject ID
```
- Automatically downloads if not present
- Uses standard Physionet event IDs (1, 2, 3)

### Option 2: Your Own Recorded Data
```yaml
replay_subject_id: "sub-P999"  # Your subject folder
```
- Data should be in `data/sub-P999/` or `data/datasets/sub-P999/`
- Must have processed .fif files in `raws/` folder
- Events should be in `events/` folder

## Expected Output During Testing

You should see output like:

```
Loading configuration from: .../bci_config.yaml
✓ Configuration loaded successfully!
Loading Adaptive LDA model from: .../adaptivelda_model.pkl
✓ Model loaded successfully!
  Classes: [0 1 2]
  Number of features: 176

Loading artifact rejection from: .../adaptivelda_artefact_removal.pkl
✓ Artifact rejection thresholds loaded!
✓ Filter initialized (online mode)
✓ BCI Controller initialized

Looking for EEG and Markers streams...
✓ Connected to EEG stream: EEG_Stream
✓ Connected to Labels stream: Labels_Stream
  EEG channels: 16
  Sampling rate: 160.0 Hz

============================================================
READY FOR ONLINE ADAPTIVE LDA CLASSIFICATION
============================================================
The model will adapt its parameters after each trial!
============================================================

Label: 1 (rest) | Predicted: 0 (rest) | Conf: 85.23% | Adaptations: 0
Label: 1 (rest) | Predicted: 0 (rest) | Conf: 87.12% | Adaptations: 0
🔄 Adapted model (Trial ended: rest → left_hand)
Label: 2 (left_hand) | Predicted: 1 (left_hand) | Conf: 72.45% | Adaptations: 1
...
```

## Next Steps After Testing

Once testing works:
1. ✅ Verify accuracy is reasonable (>60% for 3-class)
2. ✅ Check that adaptations are happening (watch adaptation count increase)
3. ✅ Review the accuracy plot to see trends
4. ✅ Try with different subjects/data
5. 🎮 Ready to test with real EEG hardware or Dino game!
