"""
Online BCI with Combined Adaptive LDA
Real-time EEG classification with online parameter adaptation

Uses CombinedAdaptiveLDA (winning model: HybridLDA + Core LDA with adaptive selection)
The key feature: Model adapts its parameters after each trial based on true labels
"""

import pickle
import socket
import sys
import time
from pathlib import Path

# Add src directory to Python path to allow imports
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import numpy as np
import matplotlib.pyplot as plt

# Try to import pylsl, install if missing
try:
    from pylsl import StreamInlet, resolve_streams
except ImportError:
    import subprocess
    # sys is already imported at line 10
    print("⚠️  pylsl not found. Installing...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pylsl"], 
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        from pylsl import StreamInlet, resolve_streams
        print("✓ pylsl installed successfully!")
    except Exception as e:
        print(f"❌ Failed to install pylsl: {e}")
        print("Please install manually: pip install pylsl")
        sys.exit(1)

from bci.preprocessing.filters import Filter
from bci.transfer.transfer import BCIController
from bci.utils.bci_config import load_config
from bci.models.adaptive_lda_modules.combined_adaptive_lda import CombinedAdaptiveLDA
from bci.models.adaptive_lda_modules.feature_extraction import extract_log_bandpower_features

# Marker definitions (CombinedAdaptiveLDA uses 0=rest, 1=left, 2=right)
markers = {
    0: "rest",
    1: "left_hand",
    2: "right_hand"
}

def extract_features(signals, sfreq):
    """Extract log-bandpower features for online use."""
    return extract_log_bandpower_features(signals, sfreq=sfreq, mu_band=(8, 12), beta_band=(13, 30))

if __name__ == "__main__":
    # Load the config file
    # Detect project root: handle both workspace root and BCI-Challenge subdirectory
    # Script is at: [workspace]/BCI-Challenge/src/bci/main_online_AdaptiveLDA.py
    script_dir = Path(__file__).parent.parent.parent  # Goes up 3 levels from script
    
    # Check if we're in a BCI-Challenge subdirectory (workspace structure)
    # The script is at: BCI-Challenge/src/bci/main_online_AdaptiveLDA.py
    # So script_dir should be BCI-Challenge directory
    # Check if script_dir contains "src" and "data" directories to confirm it's the project root
    if (script_dir / "src").exists() and (script_dir / "data").exists():
        # This is the BCI-Challenge project root
        current_wd = script_dir
    elif (script_dir / "BCI-Challenge" / "src").exists() and (script_dir / "BCI-Challenge" / "data").exists():
        # We're in workspace root, need to go into BCI-Challenge
        current_wd = script_dir / "BCI-Challenge"
    else:
        # Fallback: assume script_dir is correct
        current_wd = script_dir

    try:
        config_path = current_wd / "resources" / "configs" / "bci_config.yaml"
        print(f"Loading configuration from: {config_path}")
        config = load_config(config_path)
        print("Configuration loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        sys.exit(1)

    # Initialize variables
    np.random.seed(config.random_state)

    model_path = current_wd / "resources" / "models" / "combined_adaptive_lda.pkl"
    artefact_rejection_path = (
        current_wd / "resources" / "models" / "adaptivelda_artefact_removal.pkl"
    )

    # Load trained model
    print(f"Loading Combined Adaptive LDA model from: {model_path}")
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("Please train the model first using main_offline_AdaptiveLDA.py")
        sys.exit(1)
    
    # Load model (saved as dict with 'model' key)
    with open(model_path, "rb") as f:
        model_dict = pickle.load(f)
        clf = model_dict['model']
        model_sfreq = model_dict.get('sfreq', config.fs)
    
    print("✓ Model loaded successfully!")
    print(f"  Model stats: {clf.get_stats()}")
    print(f"  Stage info: {clf.get_stage_info()}")
    print(f"  Number of features: {clf.n_features_}")
    print(f"  Model sampling frequency: {model_sfreq} Hz")

    # Load artifact rejection thresholds
    print(f"\nLoading artifact rejection from: {artefact_rejection_path}")
    if not artefact_rejection_path.exists():
        print("⚠️  Warning: Artifact rejection file not found. Continuing without it.")
        ar = None
    else:
        ar = pickle.load(open(artefact_rejection_path, "rb"))
        print("✓ Artifact rejection thresholds loaded!")

    # Initialize filter for online processing
    filter = Filter(config, online=True)
    print("✓ Filter initialized (online mode)")

    # Initialize transfer function for sending commands to game
    controller = BCIController(config)
    print("✓ BCI Controller initialized")

    # Get channel indices to keep (for removing channels)
    if hasattr(config, 'remove_channels') and config.remove_channels:
        channel_indices_to_keep = [i for i, ch in enumerate(config.channels) if ch not in config.remove_channels]
        n_channels_after_removal = len(channel_indices_to_keep)
        print(f"  Will remove channels: {config.remove_channels}")
        print(f"  Keeping {n_channels_after_removal} channels out of {len(config.channels)}")
    else:
        channel_indices_to_keep = list(range(len(config.channels)))
        n_channels_after_removal = len(config.channels)

    # Buffers for storing incoming data
    # Buffer size matches number of channels after removal
    buffer = np.zeros((n_channels_after_removal, int(config.window_size)), dtype=np.float32)
    label_buffer = np.zeros((1, int(config.window_size)), dtype=np.int32)

    # Statistics tracking
    avg_time_per_classification = 0.0
    number_of_classifications = 0
    total_fails = 0
    total_successes = 0
    total_predictions = 0
    total_rejected = 0
    total_adaptations = 0  # Track how many times we adapted

    # Probability threshold for accepting predictions
    probability_threshold = config.classification_threshold if hasattr(config, 'classification_threshold') else 0.6
    print(f"\n✓ Statistics initialized")
    print(f"  Probability threshold: {probability_threshold}")

    # For visualization: track accuracy over time
    accuracy_history = []
    window_accuracies = []
    window_size_viz = 20  # Calculate rolling accuracy every 20 predictions

    print("\n✓ Preprocessing and model objects initialized!")

    # Find the EEG stream from LSL and establish connection
    print("\n" + "="*60)
    print("SEARCHING FOR LSL STREAMS")
    print("="*60)
    print("Looking for EEG and Markers streams...")
    print("(This may take up to 5 seconds)")
    
    streams = resolve_streams(wait_time=5.0)
    
    # Show all available streams for debugging
    print(f"\n📡 Found {len(streams)} LSL stream(s):")
    for i, stream in enumerate(streams):
        print(f"  {i+1}. Name: '{stream.name()}' | Type: '{stream.type()}' | Channels: {stream.channel_count()}")
    
    eeg_streams = [s for s in streams if s.type() == "EEG"]
    if config.online == "dino":
        label_streams = [
            s
            for s in streams
            if s.type() == "Markers" and s.name() == "MyDinoGameMarkerStream"
        ]
        expected_label_name = "MyDinoGameMarkerStream"
    else:
        label_streams = [
            s for s in streams if s.type() == "Markers" and s.name() == "Labels_Stream"
        ]
        expected_label_name = "Labels_Stream"

    if not eeg_streams:
        print("\n❌ ERROR: Could not find EEG stream!")
        print("   Available streams:")
        for s in streams:
            print(f"     - {s.name()} (type: {s.type()})")
        print("\n💡 TIP: Make sure your EEG stream is running (replay.py or real hardware)")
        sys.exit(1)
    if not label_streams:
        print(f"\n❌ ERROR: Could not find Markers stream named '{expected_label_name}'!")
        print(f"   Current online mode: {config.online}")
        print(f"   Expected stream name: '{expected_label_name}'")
        print("   Available marker streams:")
        marker_streams = [s for s in streams if s.type() == "Markers"]
        if marker_streams:
            for s in marker_streams:
                print(f"     - {s.name()} (type: {s.type()})")
        else:
            print("     (none found)")
        print(f"\n💡 TIP: For testing, set 'online: prerecorded' in config and run replay.py")
        sys.exit(1)

    inlet = StreamInlet(eeg_streams[0], max_chunklen=32)
    inlet_labels = StreamInlet(label_streams[0], max_chunklen=32)

    print("\n" + "="*60)
    print("STREAM CONNECTION SUCCESSFUL")
    print("="*60)
    print(f"✓ Connected to EEG stream: {eeg_streams[0].name()}")
    print(f"✓ Connected to Labels stream: {label_streams[0].name()}")
    print(f"  EEG channels: {inlet.info().channel_count()}")
    print(f"  Sampling rate: {inlet.info().nominal_srate()} Hz")
    print(f"  Window size: {config.window_size} samples ({config.window_size/config.fs:.2f} seconds)")
    print("\n" + "="*60)
    print("STARTING ONLINE COMBINED ADAPTIVE LDA CLASSIFICATION")
    print("="*60)
    print("Using CombinedAdaptiveLDA (HybridLDA + Core LDA with adaptive selection)")
    print("The model will adapt its parameters after each trial!")
    print("Press Ctrl+C to stop and view results")
    print("="*60 + "\n")

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        previous_label = 0  # Track previous label to detect trial boundaries
        trial_buffer = None  # Store data for one trial
        trial_true_label = None  # Store true label for adaptation

        iteration_count = 0
        last_feedback_time = time.time()
        feedback_interval = 5.0  # Print status every 5 seconds
        
        while True:
            try:
                start_classification_time = time.time() * 1000  # in milliseconds
                eeg_chunk, timestamp = inlet.pull_chunk()
                labels_chunk, label_timestamp = inlet_labels.pull_chunk()
                crt_label = None
                iteration_count += 1

                # Check if sample and labels are valid and non-empty
                if eeg_chunk:
                    # Convert to numpy arrays and transpose to (n_channels, n_samples)
                    eeg_chunk = np.array(eeg_chunk).T  # shape (n_channels, n_samples)
                    
                    # Remove unwanted channels (if specified in config)
                    if hasattr(config, 'remove_channels') and config.remove_channels:
                        eeg_chunk = eeg_chunk[channel_indices_to_keep, :]
                    
                    # Filter incoming data chunk first (stateful filter updates zi)
                    filtered_chunk = filter.apply_filter_online(eeg_chunk)
                    
                    n_new_samples = filtered_chunk.shape[1]

                    # Safety: If new data is larger than the buffer, just take the end of it
                    if n_new_samples >= config.window_size:
                        buffer = filtered_chunk[:, -config.window_size :]
                    else:
                        # Update the buffers with the filtered chunks of data
                        buffer[:, :-n_new_samples] = buffer[:, n_new_samples:]
                        buffer[:, -n_new_samples:] = filtered_chunk
                elif iteration_count == 1:
                    print("⚠️  Warning: No EEG data received in first iteration. Waiting for data...")

                if labels_chunk:
                    labels_chunk = np.array(labels_chunk).T  # shape (1, n_samples)
                    n_new_labels = labels_chunk.shape[1]

                    if n_new_labels >= config.window_size:
                        label_buffer = labels_chunk[:, -config.window_size :]
                    else:
                        label_buffer[:, :-n_new_labels] = label_buffer[:, n_new_labels:]
                        label_buffer[:, -n_new_labels:] = labels_chunk

                # Extract the current label (most present in the buffer)
                unique, counts = np.unique(label_buffer, return_counts=True)
                if len(unique) > 0:
                    label_counts = dict(zip(unique, counts))
                    crt_label = max(label_counts, key=lambda k: label_counts[k])
                else:
                    crt_label = 0  # fallback to unknown

                # Detect trial boundary (label changed from non-zero to different non-zero)
                if previous_label != 0 and crt_label != previous_label and crt_label != 0:
                    # Trial just ended! Adapt the model with previous trial data
                    if trial_buffer is not None and trial_true_label is not None and trial_true_label != 0:
                        try:
                            # Extract features from trial buffer for adaptation
                            # trial_buffer shape: (n_channels, n_samples)
                            trial_buffer_reshaped = trial_buffer[np.newaxis, :, :]  # (1, n_channels, n_samples)
                            trial_features = extract_features(trial_buffer_reshaped, config.fs)  # (1, n_features)
                            
                            # Adapt model parameters based on completed trial
                            # CombinedAdaptiveLDA.update expects: label (0,1,2) and x_feature (1D array)
                            clf.update(trial_true_label, trial_features[0])
                            total_adaptations += 1
                            print(f"🔄 Adapted model (Trial ended: {markers.get(trial_true_label, 'unknown')} → {markers.get(crt_label, 'unknown')})")
                        except Exception as e:
                            print(f"⚠️  Adaptation failed: {e}")
                            import traceback
                            traceback.print_exc()

                    # Reset trial buffer for new trial
                    trial_buffer = None
                    trial_true_label = None

                # Store current trial data
                if crt_label != 0:
                    trial_buffer = buffer.copy()
                    trial_true_label = crt_label

                previous_label = crt_label

                # Buffer already contains filtered data (filtered when added)
                # Reshape for feature extraction: (1, n_channels, n_samples)
                filtered_data_reshaped = buffer[np.newaxis, :, :]

                # Extract features (CombinedAdaptiveLDA expects features, not raw data)
                features = extract_features(filtered_data_reshaped, config.fs)  # Shape: (1, n_features)

                # Classify using extracted features
                probabilities = clf.predict_proba(features)

                if probabilities is None:
                    print("⚠️  Warning: Model returned None for probability.")
                    continue  # skip this iteration

                # Send command to game
                controller.send_command(probabilities, sock)

                # Get prediction
                prediction = np.argmax(probabilities, axis=1)[0]

                # Print classification result (only for non-zero labels or every N iterations)
                current_time = time.time()
                should_print = (crt_label != 0) or (current_time - last_feedback_time >= feedback_interval)
                
                if should_print:
                    if crt_label == 0 and current_time - last_feedback_time >= feedback_interval:
                        # Periodic status update
                        print(f"⏱️  Status: {total_predictions} predictions | {total_adaptations} adaptations | "
                              f"Avg time: {avg_time_per_classification / max(1, number_of_classifications):.1f} ms")
                        last_feedback_time = current_time
                    elif crt_label != 0:
                        # Classification result
                        print(
                            f"Label: {crt_label} ({markers.get(crt_label, 'unknown')}) | "
                            f"Predicted: {prediction} ({markers.get(prediction, 'unknown')}) | "
                            f"Conf: {probabilities[0][prediction]:.2%} | "
                            f"Adaptations: {total_adaptations}"
                        )

                total_predictions += 1

                # Track accuracy for non-unknown labels
                if crt_label != 0:
                    is_correct = (prediction) == crt_label  # Note: labels are now 0-indexed
                    accuracy_history.append(1 if is_correct else 0)

                    if probabilities[0][prediction] < probability_threshold:
                        total_rejected += 1
                    else:
                        total_successes += int(is_correct)
                        total_fails += int(not is_correct)

                    # Calculate rolling accuracy
                    if len(accuracy_history) >= window_size_viz:
                        rolling_acc = np.mean(accuracy_history[-window_size_viz:])
                        window_accuracies.append(rolling_acc)
                        if len(window_accuracies) % 5 == 0:  # Print every 5 windows
                            print(f"📊 Rolling accuracy (last {window_size_viz}): {rolling_acc:.2%}")

                number_of_classifications += 1

                end_classification_time = time.time() * 1000  # in milliseconds
                avg_time_per_classification += (
                    end_classification_time - start_classification_time
                )

            except KeyboardInterrupt:
                print("\n" + "="*60)
                print("STOPPING ONLINE PROCESSING")
                print("="*60)
                print(f"Avg time per loop: {avg_time_per_classification / max(1, number_of_classifications):.2f} ms")
                print(f"Filter latency: {filter.get_filter_latency():.2f} ms")
                print(f"Total Predictions: {total_predictions}")
                print(f"  Rejected: {total_rejected}")
                print(f"  Accepted Successes: {total_successes}")
                print(f"  Accepted Fails: {total_fails}")
                print(f"Total Adaptations: {total_adaptations}")

                if total_successes + total_fails > 0:
                    final_accuracy = total_successes / (total_successes + total_fails)
                    print(f"\nFinal Accuracy (accepted predictions): {final_accuracy:.2%}")

                if len(accuracy_history) > 0:
                    overall_accuracy = np.mean(accuracy_history)
                    print(f"Overall Accuracy (all predictions): {overall_accuracy:.2%}")
                    print(f"Total labeled trials: {len(accuracy_history)}")

                # Save accuracy plot
                if len(accuracy_history) > 0:
                    plt.figure(figsize=(12, 6))

                    # Plot 1: Raw accuracy over trials
                    plt.subplot(1, 2, 1)
                    plt.plot(accuracy_history, 'o-', alpha=0.6, markersize=3)
                    plt.axhline(y=np.mean(accuracy_history), color='r', linestyle='--',
                               label=f'Mean: {np.mean(accuracy_history):.2%}')
                    plt.xlabel('Trial Number', fontweight='bold')
                    plt.ylabel('Correct (1) / Incorrect (0)', fontweight='bold')
                    plt.title('Classification Accuracy Over Time', fontweight='bold')
                    plt.legend()
                    plt.grid(alpha=0.3)
                    plt.ylim([-0.1, 1.1])

                    # Plot 2: Rolling accuracy
                    if len(window_accuracies) > 0:
                        plt.subplot(1, 2, 2)
                        plt.plot(window_accuracies, 'o-', color='steelblue', linewidth=2)
                        plt.axhline(y=np.mean(window_accuracies), color='r', linestyle='--',
                                   label=f'Mean: {np.mean(window_accuracies):.2%}')
                        plt.xlabel(f'Window Number (size={window_size_viz})', fontweight='bold')
                        plt.ylabel('Rolling Accuracy', fontweight='bold')
                        plt.title(f'Rolling Accuracy (Window={window_size_viz} trials)', fontweight='bold')
                        plt.legend()
                        plt.grid(alpha=0.3)
                        plt.ylim([0, 1.0])

                    plt.tight_layout()
                    plot_path = current_wd / "combined_adaptive_lda_online_accuracy.png"
                    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                    print(f"\n✓ Accuracy plot saved: {plot_path}")
                    plt.close()

                print("="*60)
                break
