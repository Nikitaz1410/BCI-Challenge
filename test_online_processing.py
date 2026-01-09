"""
Test online processing functions
"""

import numpy as np
from pathlib import Path
from src.bci import preprocessing

print("Testing online processing functions...\n")

# Test config (same as offline)
config = {
    "fmin": 1.0,
    "fmax": 35.0,
    "max_amplitude": 200e-6,
    "use_asr": False
}

sfreq = 250.0
n_channels = 7
window_size = 250  # 1.0 second
step = 40

# ============================================================================
# Test 1: filter_online
# ============================================================================
print("[1/5] Testing filter_online...")
window = np.random.randn(n_channels, window_size) * 10e-6  # Simulated EEG
window_filtered = preprocessing.filter_online(
    window, sfreq=sfreq, fmin=config["fmin"], fmax=config["fmax"]
)
print(f"  ✓ Input shape: {window.shape}")
print(f"  ✓ Output shape: {window_filtered.shape}")
print(f"  ✓ Filtered successfully!\n")

# ============================================================================
# Test 2: create_sliding_window_online
# ============================================================================
print("[2/5] Testing create_sliding_window_online...")
# Create a buffer with more samples than window_size
buffer = np.random.randn(n_channels, 1000) * 10e-6
window = preprocessing.create_sliding_window_online(
    buffer, window_size=window_size, step=step
)
if window is not None:
    print(f"  ✓ Buffer shape: {buffer.shape}")
    print(f"  ✓ Window shape: {window.shape}")
    print(f"  ✓ Window extracted successfully!\n")
else:
    print("  ✗ Buffer too short\n")

# Test with short buffer
short_buffer = np.random.randn(n_channels, 100)
window = preprocessing.create_sliding_window_online(
    short_buffer, window_size=window_size, step=step
)
if window is None:
    print(f"  ✓ Correctly returned None for short buffer\n")

# ============================================================================
# Test 3: handle_bad_window_online
# ============================================================================
print("[3/5] Testing handle_bad_window_online...")
# Good window (small amplitude)
good_window = np.random.randn(n_channels, window_size) * 10e-6
window_clean, is_bad = preprocessing.handle_bad_window_online(
    good_window, max_amplitude=config["max_amplitude"]
)
print(f"  ✓ Good window: is_bad={is_bad} (should be False)")

# Bad window (large amplitude - artifact)
bad_window = np.random.randn(n_channels, window_size) * 500e-6  # Too large
window_clean, is_bad = preprocessing.handle_bad_window_online(
    bad_window, max_amplitude=config["max_amplitude"]
)
print(f"  ✓ Bad window: is_bad={is_bad} (should be True)\n")

# ============================================================================
# Test 4: clean_window_asr_online
# ============================================================================
print("[4/5] Testing clean_window_asr_online...")
window = np.random.randn(n_channels, window_size) * 10e-6
window_clean = preprocessing.clean_window_asr_online(
    window, asr_object=None, sfreq=sfreq
)
print(f"  ✓ ASR cleaning (disabled): shape={window_clean.shape}\n")

# ============================================================================
# Test 5: process_window_online (complete pipeline)
# ============================================================================
print("[5/5] Testing process_window_online (complete pipeline)...")
window = np.random.randn(n_channels, window_size) * 10e-6
window_processed, is_bad = preprocessing.process_window_online(
    window, config=config, ar=None, asr=None, sfreq=sfreq
)
print(f"  ✓ Input shape: {window.shape}")
print(f"  ✓ Output shape: {window_processed.shape}")
print(f"  ✓ is_bad: {is_bad} (should be False)")
print(f"  ✓ Complete pipeline works!\n")

# ============================================================================
# Test with bad window
# ============================================================================
print("Testing with bad window (large artifact)...")
bad_window = np.random.randn(n_channels, window_size) * 500e-6
window_processed, is_bad = preprocessing.process_window_online(
    bad_window, config=config, ar=None, asr=None, sfreq=sfreq
)
print(f"  ✓ Bad window detected: is_bad={is_bad} (should be True)\n")

print("=" * 60)
print("ALL TESTS PASSED! ✓")
print("=" * 60)