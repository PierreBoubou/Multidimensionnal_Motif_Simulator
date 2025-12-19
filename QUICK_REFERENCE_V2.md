# QUICK REFERENCE - New Features in Version 2.0

## Major Improvements at a Glance

| Feature | Before | After |
|---------|--------|-------|
| Motif Visibility | ❌ Not visible | ✅ Clear with annotations |
| Base Signal Control | ❌ None | ✅ Amplitude scaling |
| Parameter Ranges | ⚠️ Restrictive sliders | ✅ Flexible input ranges |
| Multiple Motif Control | ⚠️ Random mix | ✅ Single type + dimension choice |
| Motif Preview | ❌ None | ✅ Real-time plot |
| Collision Detection | ⚠️ O(n×m) check | ✅ O(n) mask-based |
| Multiple Motif Types | ❌ Limited | ✅ Full control |
| Serialization | ⚠️ Warnings | ✅ Clean save |
| Load Datasets | ❌ Not possible | ✅ Full preview & load |

---

## 1️⃣ Base Signal Amplitude Control

**Location**: Sidebar → Signal Setup → Adjust Base Signal (after generation)

**What it does**: Scale the entire base signal up or down without regenerating

**How to use**:
1. Generate base signal
2. Scroll down in sidebar
3. Adjust "Base Signal Amplitude Scale" slider (0.1x to 3.0x)
4. Click "Apply Amplitude Scale"
5. Signal is multiplied by the scale factor

```
Original signal × scale = Adjusted signal
Example: Original with scale 2.0 = Twice as large
```

---

## 2️⃣ Flexible Parameter Ranges

**Location**: Add Motifs tabs → Parameter configuration sections

**What changed**:
- **Single Motif**: Now uses range sliders
  ```
  Frequency Range: [0.05 ─────●─────── 0.30]  ← Pick min and max
  Randomized within range when added
  ```

- **Multiple Motifs**: Now uses number inputs
  ```
  Min Motif Length: [50]
  Max Motif Length: [150]
  ```

**Benefits**:
- No more restrictive preset ranges
- Set exactly what you want
- Much more flexible

---

## 3️⃣ Single Motif Type for Multiple Motifs

**Location**: Add Motifs → Multiple Motifs tab

**Old Workflow** (Random mix):
```
Types: [✓] morlet  [✓] sine  [✓] triangle  [✓] square
→ Completely random selection
```

**New Workflow** (Controlled):
```
1. Motif Type to Add: [morlet ▼]
2. Dimensions: [✓] Dim 0  [✓] Dim 1  [ ] Dim 2
3. Length: Min [50] Max [150]
4. Amplitude: Min [1.0] Max [2.5]
5. Parameters (Morlet specific):
   - Frequency Range: [0.1 ────── 0.2]
   - Sigma Range: [1.0 ────── 2.0]
6. Click "Add Multiple Motifs"
```

**To add different types**:
- After adding 10 Morlets, change to "sine"
- Adjust settings
- Add 10 Sines
- Result: 20 total, 10 of each type

---

## 4️⃣ Motif Preview

**Location**: Add Motifs → Single Motif tab → Motif Preview (at bottom)

**What you see**:
- Interactive Plotly plot of the motif
- Updates in real-time as you change parameters
- Shows exact shape before adding
- Hover for exact values

**Example preview changes**:
```
Sine Motif:
Frequency 0.1  → Lower pitch, fewer oscillations
Frequency 0.3  → Higher pitch, more oscillations

Morlet Wavelet:
Sigma 0.5      → Narrow, localized
Sigma 3.0      → Wide, spread out

Dampened Oscillator:
Damping 0.05   → Slow decay
Damping 0.2    → Fast decay
```

---

## 5️⃣ Full Dimension Control for Batch Motifs

**Location**: Add Motifs → Multiple Motifs tab → Dimensions section

**What changed**:
```
Before: Random dimension selection (hard to control)

After: Explicit dimension checkboxes:
  ☐ Include Dimension 0
  ☐ Include Dimension 1
  ☑ Include Dimension 2
  ☑ Include Dimension 3
```

**Result**: All 10 motifs will be placed on dimensions 2 and 3 only

**Use case**:
- "I want all Sine motifs on dimensions 0-1"
- "I want all Exponentials on dimension 2 only"
- Mix and match by running multiple times

---

## 6️⃣ Dataset Loading

**Location**: Main tabs → "Load Dataset" tab

**Workflow**:
1. Click "Load Dataset" tab
2. Upload a `.pkl` file
3. See preview:
   - Signal shape
   - Number of motifs
   - Signal plot (first 3 dims)
   - Table of all motifs
4. Click "Load This Dataset" to restore full simulator

**What happens after loading**:
- Full simulator restored
- Can edit motifs
- Can add more motifs
- Can save as new file

---

## 7️⃣ Improved Collision Detection

**What changed** (internal):
```
Before: Check each new motif against all existing
  Time: O(n × m) where n=existing, m=new

After: Use boolean mask array
  Time: O(n) mask creation + O(1) lookups
```

**For you**: Faster placement of many motifs, same results

---

## 🎯 Typical Workflow (Now Improved)

### Creating a Simple Dataset

```
1. GENERATE BASE SIGNAL
   Signal Length: 1000
   Dimensions: 2
   Type: AR(2)
   
2. ADJUST AMPLITUDE (NEW!)
   Scale: 1.5 (make base signal stronger)
   Apply
   
3. ADD FIRST BATCH OF MOTIFS
   Type: sine
   Dimensions: [0, 1]
   Length: 50-100
   Amplitude: 1.0-2.0
   Frequency: 0.1-0.2
   Add 10 Motifs
   
4. PREVIEW (NEW!)
   Look at Motif Preview to see what they look like
   
5. ADD SECOND BATCH (NEW!)
   Type: morlet (changed!)
   Dimensions: [0] (only dimension 0)
   Adjust new parameters for Morlet
   Add 8 More Motifs
   
6. VIEW & EDIT
   See all 18 motifs in table
   Edit amplitude of motif #5
   
7. SAVE & LOAD
   Save as "my_dataset.pkl"
   Later: Load it back to continue
```

---

## 📊 Real-World Example

### Create a Validation Dataset

```
Goal: Create dataset with specific motif distribution

Step 1: Generate Base Signal
- Length: 2000 samples
- Dimensions: 3
- Type: Sinusoidal (varying frequency)

Step 2: Adjust Base Signal
- Scale: 0.8 (make quieter)
- Reason: Motifs should stand out

Step 3: Add Morlet Wavelets
- Dimensions: [0, 1] (spread on 2 dims)
- Lengths: 60-120
- Amplitudes: 1.5-2.5
- Frequencies: 0.08-0.15
- Sigmas: 1.0-2.0
- Add: 15 motifs

Step 4: Add Exponential Motifs
- Dimensions: [1, 2] (different dims)
- Lengths: 50-100
- Amplitudes: 2.0-3.0
- Decay Rates: 0.08-0.15
- Add: 10 motifs

Step 5: Add Dampened Oscillators
- Dimensions: [2] (only last dimension)
- Lengths: 80-150
- Amplitudes: 1.0-2.0
- Frequencies: 0.1-0.2
- Damping: 0.05-0.15
- Add: 8 motifs

Step 6: Save & Document
- Save as "validation_mixed_3types.pkl"
- Total: 33 motifs, 3 types, 3 dimensions
- Ready for algorithm testing

Step 7: Share & Reload
- Send file to colleague
- They load it to see what you created
- They can edit and save variants
```

---

## 🔧 Tips & Tricks

### Tip 1: Create Many Versions
```
1. Generate base signal
2. Add 10 Sine motifs
3. Save as "version_sine_v1.pkl"
4. Add 10 Morlet motifs (same base)
5. Save as "version_sine_morlet.pkl"
6. Can now compare algorithm performance
```

### Tip 2: Preview Before Adding
```
1. Set up parameters in Single Motif tab
2. Look at preview plot
3. If it looks good, note down the exact ranges
4. Switch to Multiple Motifs tab
5. Use exact same ranges to add batch
```

### Tip 3: Build Incrementally
```
1. Add 5 motifs of type A
2. Visualize (check they look good)
3. Add 5 motifs of type B
4. Visualize (check they don't overlap badly)
5. Add 5 motifs of type C
6. Visualize (final check)
7. Save when happy
```

### Tip 4: Analyze Loaded Datasets
```
1. Load a dataset
2. Study the motif table:
   - Which dimensions are used?
   - What types are present?
   - What parameter ranges?
3. Use this info to create similar datasets
```

---

## ⚙️ Technical Details

### Serialization Fix
```
Before: pickle.dump(dataclass_with_numpy)
Result: Warnings about non-serializable objects

After: Convert to plain dict with native Python types
Result: Clean saves, no warnings
```

### Collision Detection
```
Before: For each new motif, check against all existing
for existing in motif_list:
    if overlaps(new, existing):
        return False

After: Use mask array
mask[start:end, dim] = False  # Mark occupied
# Later: Check if all([mask[start:end, dim]])
```

---

## 🆘 Troubleshooting

**Q: Motifs not showing?**  
A: Try hitting refresh (F5) or scroll up to see visualization

**Q: Can't find parameter I need?**  
A: Make sure you're in the right tab (Single vs Multiple)

**Q: Want to add different motif types?**  
A: Add one type, then change type selector and add another batch

**Q: Motifs keep overlapping?**  
A: Either increase signal length or decrease motif lengths

---

## 🎉 Summary

**Version 2.0 makes the app**:
- ✅ More visible (motifs clearly shown)
- ✅ More controllable (full parameter ranges)
- ✅ More predictable (single type selection)
- ✅ More informative (preview plots)
- ✅ More efficient (better collision detection)
- ✅ More useful (dataset loading)
- ✅ Better quality (clean serialization)

**Everything still works the same way, just better!**

---

## Launch Command (Unchanged)

```bash
streamlit run streamlit_motif_simulator.py
```

---

*Quick Reference - Version 2.0*  
*For detailed info see: IMPROVEMENTS_AND_FIXES.md*
