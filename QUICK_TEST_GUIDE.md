# Quick Test Guide: Online Adaptive LDA

Follow these steps to test the online Adaptive LDA classifier on your machine.

## Prerequisites Check

Before starting, verify you have:

✅ **Trained Model**: `resources/models/adaptivelda_model.pkl`
- If missing: Run `main_offline_AdaptiveLDA.ipynb` first

✅ **Config File**: `resources/configs/bci_config.yaml`
- Should have `online: "prerecorded"` for testing

✅ **Test Data**: Available for replay
- Check `replay_subject_id` in config (e.g., "Phy-42" or "sub-P999")

## Step-by-Step Testing

### Step 1: Configure for Testing

Edit `resources/configs/bci_config.yaml`:

```yaml
online: "prerecorded"  # Change from "dino" to "prerecorded" for testing
replay_subject_id: "Phy-42"  # Use Physionet subject or your data
```

**Why**: `"prerecorded"` mode looks for `"Labels_Stream"` (from replay.py), while `"dino"` looks for game markers.

### Step 2: Open Terminal 1 - Start Data Replay

This simulates real-time EEG by streaming prerecorded data:

```bash
cd "/Users/amalbenslimen/BCI Challenge /BCI-Challenge"
uv run python -m src.bci.replay
```

**What you'll see:**
```
Loading configuration from: ...
✓ Configuration loaded successfully!
Loaded subject 42 from Physionet for training.
Starting LSL streams (EEG and Labels)...
```

**Keep this terminal running!** The replay streams data until finished.

### Step 3: Open Terminal 2 - Run Online Classification

In a **new terminal window**, run:

```bash
cd "/Users/amalbenslimen/BCI Challenge /BCI-Challenge"
uv run python -m src.bci.main_online_AdaptiveLDA
```

**What you'll see:**
```
Loading configuration from: ...
✓ Configuration loaded successfully!
Loading Adaptive LDA model from: ...
✓ Model loaded successfully!
  Classes: [0 1 2]
  Number of features: 176

============================================================
SEARCHING FOR LSL STREAMS
============================================================
Looking for EEG and Markers streams...
📡 Found 2 LSL stream(s):
  1. Name: 'EEG_Stream' | Type: 'EEG' | Channels: 16
  2. Name: 'Labels_Stream' | Type: 'Markers' | Channels: 1

============================================================
STREAM CONNECTION SUCCESSFUL
============================================================
✓ Connected to EEG stream: EEG_Stream
✓ Connected to Labels stream: Labels_Stream
  EEG channels: 16
  Sampling rate: 160.0 Hz
  Window size: 250 samples (1.56 seconds)

============================================================
STARTING ONLINE ADAPTIVE LDA CLASSIFICATION
============================================================
The model will adapt its parameters after each trial!
Press Ctrl+C to stop and view results
============================================================

Label: 1 (rest) | Predicted: 0 (rest) | Conf: 85.23% | Adaptations: 0
Label: 1 (rest) | Predicted: 0 (rest) | Conf: 87.12% | Adaptations: 0
🔄 Adapted model (Trial ended: rest → left_hand)
Label: 2 (left_hand) | Predicted: 1 (left_hand) | Conf: 72.45% | Adaptations: 1
...
```

### Step 4: Stop and View Results

Press `Ctrl+C` in Terminal 2 to stop. You'll see:

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

## Troubleshooting

### ❌ "Could not find EEG stream"

**Problem**: Replay script not running or streams not ready

**Solution**:
1. Make sure Terminal 1 (replay.py) is running
2. Wait 2-3 seconds after starting replay before running main script
3. Check replay.py output for errors

### ❌ "Could not find Markers stream"

**Problem**: Wrong `online` setting in config

**Solution**:
- Set `online: "prerecorded"` in `bci_config.yaml`
- Restart both scripts

### ❌ "Model not found"

**Problem**: Model hasn't been trained

**Solution**:
- Run `main_offline_AdaptiveLDA.ipynb` to train the model
- Check that `resources/models/adaptivelda_model.pkl` exists

### ⚠️ Script runs but no predictions

**Problem**: No labeled data in stream

**Solution**:
- Check replay.py is sending labels (look for "Labels_Stream" in output)
- Verify test data has events/labels

## Visual Guide

```
Terminal 1 (Replay)          Terminal 2 (Online Classification)
─────────────────────        ─────────────────────────────────
$ uv run python              $ uv run python
  -m src.bci.replay            -m src.bci.main_online_AdaptiveLDA
                              │
                              ├─> Loads config ✓
                              ├─> Loads model ✓
                              ├─> Finds streams ✓
                              ├─> Connects to streams ✓
                              │
                              └─> Classification loop
                                  (Press Ctrl+C to stop)
```

## Expected Timeline

1. **0-5s**: Script loads config, model, connects to streams
2. **5s+**: Classification starts, predictions appear
3. **After each trial**: Model adapts (you'll see "🔄 Adapted model")
4. **Every 5s**: Status update (predictions count, adaptations)
5. **Ctrl+C**: Stop and see final statistics + plot

## Success Indicators

✅ **Streams connected**: See "✓ Connected to EEG stream" and "✓ Connected to Labels stream"

✅ **Predictions appearing**: See "Label: X | Predicted: Y | Conf: Z%" messages

✅ **Adaptations happening**: See "🔄 Adapted model" messages when trials end

✅ **Reasonable accuracy**: >60% for 3-class classification (better than 33% chance)

## Next Steps

Once testing works:
1. ✅ Try with different subjects/data
2. ✅ Adjust `classification_threshold` in config
3. ✅ Test with real EEG hardware
4. 🎮 Test with Dino game (set `online: "dino"`)
