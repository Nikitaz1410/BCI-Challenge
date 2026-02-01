# Possible Reasons for Low BCI Classification Accuracy

## 1. **Class Imbalance** ⚠️ HIGH PRIORITY
- **Problem**: Severe imbalance between classes can cause the model to favor the majority class
- **Current Status**: You've just added diagnostics to check this
- **Indicators**:
  - One class (especially class 3/right_hand) has significantly more samples
  - Model predicts the majority class most of the time
  - Low per-class accuracy for minority classes
- **Solutions**:
  - Use class weights in LDA (if sklearn supports it) or resampling
  - Check if imbalance comes from:
    - Original data collection
    - Aggressive artifact removal removing more samples from certain classes
    - Windowing creating more windows from certain epochs

## 2. **Aggressive Artifact Removal** ⚠️ HIGH PRIORITY
- **Problem**: Too many epochs/windows rejected, leaving insufficient training data
- **Current Issue**: You've been tuning artifact removal thresholds
- **Indicators**:
  - >50% of data rejected
  - Imbalanced rejection across classes
  - Very few samples left after rejection
- **Solutions**:
  - Adjust `percentile` parameter (currently 97.0, try 95.0 or 98.5)
  - Adjust `threshold_multiplier` (currently 1.0, try 0.8 or 1.2)
  - Check if certain classes have more artifacts naturally
  - Consider using ASR instead of or in addition to threshold-based rejection

## 3. **Feature Extraction Issues** ⚠️ MEDIUM PRIORITY
- **Problem**: PSD features may not be optimal for motor imagery
- **Current Implementation**: Uses Welch method with 500ms window, 20ms overlap
- **Potential Issues**:
  - **Window size too small**: 250 samples @ 160 Hz = 1.56 seconds may be too short for MI
  - **Frequency range**: 8-30 Hz (mu + beta) is correct, but frequency resolution may be low
  - **PSD vs CSP**: CSP (Common Spatial Patterns) is often better for MI than raw PSD
- **Solutions**:
  - Increase window size to capture more MI activity (2-4 seconds)
  - Try CSP feature extraction instead of/alongside PSD
  - Use log-PSD instead of linear PSD (better for neural data)
  - Try different frequency bands (e.g., 8-13 Hz mu, 13-30 Hz beta separately)

## 4. **Epoch Timing Issues** ⚠️ MEDIUM PRIORITY
- **Problem**: Wrong time window may miss MI activity
- **Current Settings**: `tmin=0.5, tmax=4.0` (3.5 second window)
- **Potential Issues**:
  - `tmin=0.5` might skip early MI response
  - MI activity typically starts 0.3-0.5s after cue and lasts 1-3 seconds
  - No baseline correction (`baseline=None`) - this is actually fine for MI
- **Solutions**:
  - Try `tmin=0.3` to capture earlier response
  - Try shorter window `tmax=3.5` to avoid later noise
  - Consider baseline correction `baseline=(-0.5, 0.0)`

## 5. **LDA Limitations** ⚠️ MEDIUM PRIORITY
- **Problem**: Standard LDA assumes linear separability and equal covariance
- **Current Implementation**: Custom LDA without class weights
- **Potential Issues**:
  - **No class balancing**: LDA doesn't inherently handle imbalanced classes
  - **Linear separability**: MI signals may require non-linear features
  - **Covariance assumption**: Assumes same covariance for all classes
- **Solutions**:
  - Use sklearn's LDA with `class_weight='balanced'` (if compatible)
  - Try Quadratic Discriminant Analysis (QDA) for different covariances
  - Consider other classifiers: SVM with RBF kernel, CSP+LDA (spatial filtering first)

## 6. **Cross-Subject Generalization** ⚠️ MEDIUM PRIORITY
- **Problem**: Training on multiple subjects may create inter-subject variability
- **Current Setup**: Using GroupKFold with multiple Physionet subjects
- **Potential Issues**:
  - Different subjects have different MI patterns
  - Individual differences in mu/beta reactivity
  - Subject-specific preprocessing might be needed
- **Solutions**:
  - Check per-subject accuracy to identify problematic subjects
  - Consider subject-specific normalization
  - Try within-subject cross-validation separately

## 7. **Data Quality Issues** ⚠️ LOW PRIORITY
- **Problem**: Poor signal quality, wrong channels, or preprocessing errors
- **Potential Issues**:
  - **Missing motor channels**: Ensure C3, C4, Cz are present (they are in your config)
  - **Reference issues**: M1_M2 reference may not be optimal
  - **Filter artifacts**: Edge effects from filtering
- **Solutions**:
  - Verify channel locations are correct
  - Try average reference instead of M1_M2
  - Check for filter ringing/artifacts

## 8. **Windowing Strategy** ⚠️ LOW PRIORITY
- **Problem**: Overlapping windows may create correlation and class leakage
- **Current Settings**: `window_size=250, step_size=32` (high overlap)
- **Potential Issues**:
  - High overlap (87%) creates many similar windows from same epoch
  - This can cause overfitting and inflated CV scores
  - Labels are correctly propagated, but temporal correlation exists
- **Solutions**:
  - Reduce overlap (e.g., `step_size=125` for 50% overlap)
  - Or use non-overlapping windows for training
  - Consider trial-level (not window-level) cross-validation

## 9. **Label Encoding Issues** ✅ FIXED
- **Status**: You've just converted labels to 0-indexed [0, 1, 2]
- **Was a problem**: If labels were [1, 2, 3], some sklearn functions might expect 0-indexed

## 10. **Insufficient Training Data** ⚠️ CHECK
- **Problem**: After artifact removal, you may have too few samples
- **Indicators**: 
  - Less than ~50 samples per class
  - High variance in CV scores across folds
- **Solutions**:
  - Collect more data
  - Be less aggressive with artifact removal
  - Use data augmentation (add noise, time shifts)

## Diagnostic Steps to Identify the Issue:

1. **Check class distribution** (you've added this):
   ```python
   # Before and after artifact removal
   # Per subject
   # Per fold
   ```

2. **Check feature separability**:
   ```python
   # Plot PSD features in 2D (PCA/t-SNE)
   # Check if classes are separable
   ```

3. **Check per-class accuracy**:
   ```python
   # From confusion matrix
   # Identify which classes are confused
   ```

4. **Check subject-specific performance**:
   ```python
   # Accuracy per subject
   # Identify problematic subjects
   ```

5. **Check feature quality**:
   ```python
   # Mean PSD per class and channel
   # Verify mu/beta suppression/enhancement
   ```

6. **Check model predictions**:
   ```python
   # Prediction probabilities distribution
   # Are predictions confident or uncertain?
   ```

## Recommended Action Plan:

### Immediate (High Impact):
1. ✅ **DONE**: Check class distribution before/after artifact removal
2. **NEXT**: Analyze which classes are most confused (confusion matrix)
3. **NEXT**: Check per-subject accuracy to identify data quality issues
4. **NEXT**: Tune artifact removal thresholds based on class distribution

### Short-term (Medium Impact):
5. Try CSP feature extraction instead of PSD
6. Adjust epoch timing (tmin, tmax)
7. Add class balancing to LDA (if possible)
8. Reduce window overlap to prevent overfitting

### Long-term (If still low):
9. Try alternative classifiers (SVM, QDA, CSP+LDA)
10. Consider subject-specific models
11. Collect more training data

## Expected Accuracy Ranges:

- **Good BCI**: 70-85% (3-class MI)
- **Acceptable**: 60-70%
- **Poor**: <60% (around chance level 33% for 3-class)

If you're getting <50%, likely issues are:
- Class imbalance (model predicting majority class)
- Too aggressive artifact removal
- Wrong features/classifier for MI
- Data quality issues
