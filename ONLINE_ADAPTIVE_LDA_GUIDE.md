# Guide: Running Online Adaptive LDA

This guide explains how to run the Hybrid Adaptive LDA model in real-time online mode.

## Prerequisites

1. **Trained Model**: You must have trained the model first using `main_offline_AdaptiveLDA.py`
   - Model will be saved to: `resources/models/hybrid_lda.pkl`
   - If missing, run the offline script first

2. **Python Environment**: Make sure you're in the `bci-challenge` conda environment
   ```bash
   conda activate bci-challenge
   ```

3. **Dependencies**: Ensure `pylsl` is installed (will auto-install if missing)

## Step-by-Step Instructions

### Option 1: Testing with Prerecorded Data (Recommended for First Run)

This mode replays existing EEG data through LSL streams for testing.

#### Step 1: Configure the Config File

Edit `resources/configs/bci_config.yaml`:

```yaml
online: "prerecorded"  # Use prerecorded data for testing
replay_subject_id: "sub-P999"  # Or "Phy-42" for Physionet data
```

#### Step 2: Start the Replay Stream (Terminal 1)

This simulates a real-time EEG stream by replaying recorded data:

```bash
cd "/Users/amalbenslimen/BCI Challenge /BCI-Challenge"
/opt/miniconda3/envs/bci-challenge/bin/python src/bci/replay.py
```

**Expected Output:**
```
Loading configuration from: ...
Configuration loaded successfully!
Loaded subject ... for training.
EEG data shape: (16, XXXXX), Labels shape: (XXXXX,)
Starting LSL streams (EEG and Labels)...
```

**Keep this terminal running!** The replay script will stream data continuously.

#### Step 3: Run the Online Classifier (Terminal 2)

In a **new terminal**, run the online classifier:

```bash
cd "/Users/amalbenslimen/BCI Challenge /BCI-Challenge"
/opt/miniconda3/envs/bci-challenge/bin/python src/bci/main_online_AdaptiveLDA.py
```

**Expected Output:**
```
Loading Hybrid LDA model from: ...
✓ Model loaded successfully!
  Stage info: {...}
  Number of features: 22

SEARCHING FOR LSL STREAMS
📡 Found 2 LSL stream(s):
  1. Name: 'EEG_Stream' | Type: 'EEG' | Channels: 16
  2. Name: 'Labels_Stream' | Type: 'Markers' | Channels: 1

STREAM CONNECTION SUCCESSFUL
✓ Connected to EEG stream: EEG_Stream
✓ Connected to Labels stream: Labels_Stream

STARTING ONLINE ADAPTIVE LDA CLASSIFICATION
The model will adapt its parameters after each trial!
Press Ctrl+C to stop and view results
```

#### Step 4: Monitor the Output

You'll see real-time classification results:

```
Label: 1 (left_hand) | Predicted: 1 (left_hand) | Conf: 0.65% | Adaptations: 0
🔄 Adapted model (Trial ended: left_hand → right_hand)
Label: 2 (right_hand) | Predicted: 2 (right_hand) | Conf: 0.72% | Adaptations: 1
📊 Rolling accuracy (last 20): 0.65
```

#### Step 5: Stop and View Results

Press `Ctrl+C` to stop. You'll see:

```
STOPPING ONLINE PROCESSING
Total Predictions: 150
  Accepted Successes: 95
  Accepted Fails: 25
Total Adaptations: 30

Final Accuracy (accepted predictions): 79.17%
Overall Accuracy (all predictions): 75.33%

✓ Accuracy plot saved: adaptive_lda_online_accuracy.png
```

---

### Option 2: Real-Time with Dino Game

For actual real-time BCI control (requires active EEG hardware):

#### Step 1: Configure for Dino Mode

Edit `resources/configs/bci_config.yaml`:

```yaml
online: "dino"  # Real-time mode with dino game
ip: "127.0.0.1"  # Game IP address
port: 5005  # Game UDP port
```

#### Step 2: Start Your EEG Stream

- Connect your EEG hardware
- Start your LSL stream with name "EEG_Stream"
- Start marker stream with name "MyDinoGameMarkerStream"

#### Step 3: Run the Online Classifier

```bash
cd "/Users/amalbenslimen/BCI Challenge /BCI-Challenge"
/opt/miniconda3/envs/bci-challenge/bin/python src/bci/main_online_AdaptiveLDA.py
```

The model will:
- Connect to your EEG stream
- Classify in real-time
- Send commands to the dino game via UDP
- Adapt after each trial based on game feedback

---

## What the Online Script Does

1. **Loads the trained model** from `hybrid_lda.pkl`
2. **Connects to LSL streams** (EEG data + labels/markers)
3. **Processes data in real-time**:
   - Filters incoming EEG chunks
   - Removes unwanted channels
   - Extracts log-bandpower features (mu + beta)
   - Classifies using HybridLDA
4. **Adapts the model** after each trial:
   - When a trial ends (label changes)
   - Updates class means using EMA
   - Improves accuracy over time
5. **Sends commands** to the game (if in dino mode)
6. **Tracks statistics** and saves accuracy plots

---

## Key Features

### Real-Time Adaptation

The model adapts after each trial:
```
🔄 Adapted model (Trial ended: left_hand → right_hand)
```

This updates the class means using:
```
new_mean = 0.99375 × old_mean + 0.00625 × new_sample
```

### Statistics Tracking

- **Rolling accuracy**: Last 20 trials
- **Overall accuracy**: All predictions
- **Adaptation count**: How many times the model adapted
- **Confidence threshold**: Only accepts predictions above threshold

### Output Files

- `adaptive_lda_online_accuracy.png`: Accuracy plot over time

---

## Troubleshooting

### Error: "Model not found"
**Solution**: Train the model first:
```bash
python src/bci/main_offline_AdaptiveLDA.py
```

### Error: "Could not find EEG stream"
**Solution**: 
- Make sure `replay.py` is running (for prerecorded mode)
- Check that your LSL stream is named "EEG_Stream"
- Verify LSL is installed: `pip install pylsl`

### Error: "Could not find Markers stream"
**Solution**:
- For prerecorded: Make sure `replay.py` is running
- For dino mode: Check marker stream name is "MyDinoGameMarkerStream"
- Verify config has `online: "prerecorded"` or `online: "dino"`

### Low Accuracy
**Possible causes**:
- Model needs more adaptation time
- Feature extraction mismatch
- Channel configuration mismatch
- Label mapping issues

### Adaptation Not Working
**Check**:
- Labels are being received (check console output)
- Trial boundaries are detected (label changes)
- Model update() is being called (check adaptation count)

---

## Configuration Options

In `bci_config.yaml`:

```yaml
online: "prerecorded"  # or "dino"
replay_subject_id: "sub-P999"  # Subject to replay
classification_threshold: 0.6  # Confidence threshold (0-1)
classification_buffer: 10  # Majority voting buffer size
ip: "127.0.0.1"  # Game IP (dino mode only)
port: 5005  # Game port (dino mode only)
```

---

## Expected Performance

Based on offline results:
- **Baseline accuracy**: ~40-45%
- **With adaptation**: Improves over time
- **Processing latency**: <1 ms per classification
- **Adaptation rate**: After each trial (when label changes)

---

## Next Steps

1. **Test with prerecorded data** first to verify everything works
2. **Monitor adaptation**: Watch how accuracy improves over time
3. **Tune parameters**: Adjust `classification_threshold` if needed
4. **Try real-time**: Once confident, test with actual EEG hardware

---

## Notes

- The model adapts **after** each trial (when label changes)
- Adaptation uses a small learning rate (0.00625) for stability
- Channel removal happens automatically based on `config.remove_channels`
- Features are extracted using the same method as offline training
