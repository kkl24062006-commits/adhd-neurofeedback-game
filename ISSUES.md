# Focus Rocket — Flaws & Improvement List

Review of `adhd_neurofeedback_game.py`, ranked by severity. Line refs point at the
current file.

## CRITICAL

1. **"Distractors" don't do anything — higher difficulty is fake.**
   `Distractor` objects are created (`distractors = [Distractor() for _ in range(cfg["distractors"])]`, L583) and drawn every frame (L706-709), but there is no collision detection or any other hook between them and the rocket, score, or focus value. Levels 2-4 advertise "Distractors: 3/5/8" (L486) as part of what makes them harder, but mechanically Hard and Expert are identical to Easy except for the threshold/sustain numbers. The core difficulty knob for a distraction-resistance game doesn't affect gameplay.

2. **Live-mode baseline can silently fail, making the session unplayable.**
   L625-630: if fewer than 3 valid TBR samples are collected during the 20 s live calibration (dropped packets, late stream start, noisy connection), `source.set_baseline()` is never called, so `_baseline_mu` stays `None`. `_tbr_to_focus()` (L153-157) then returns a constant `0.5` for the *entire* session. `target` becomes `0.5 + threshold_pct`, which the frozen `0.5` focus can never reach — the rocket can never climb. Nothing on screen tells the operator/child this happened; it just looks like a broken or unresponsive game.

3. **No validation that hardware channel order matches `CHANNEL_NAMES`.**
   L52 hardcodes `["FP1","FP2","O1","O2","C3","CZ","FZ","C4"]` and L114 indexes into incoming LSL samples using that assumed order, with no check against the actual stream's channel metadata (`StreamInfo.desc()` channel labels are never read, L104-110). If the QNeuro device or a different montage reports channels in another order, theta/beta power is computed from the wrong electrodes — silently, with no error — undermining the entire feedback signal.

4. **Silent EEG dropout during live play.**
   L159-177: if the LSL stream disconnects mid-session, `pull_sample(timeout=1.0)` just keeps timing out and the loop `continue`s forever; `_focus`/`_tbr` freeze at their last values and the game keeps running as if nothing happened. For a tool whose entire premise is real-time brain-state feedback, a stalled signal should be visibly flagged, not presented as continued live data.

## HIGH

5. **`session_log.csv` baseline is a meaningless constant for live sessions.**
   L625-630 vs L734-743: for live sessions `baseline` is *always* written as `0.5` (the real calibrated TBR baseline is only `print()`-ed to the console, never saved). `results_figures.py` and `results_mat.m` both read this log file directly — any downstream comparison of "baseline" across sessions is silently broken for every live run mixed with CSV/sim runs.

6. **`CSVBandPowerSource` fails with a raw `KeyError` on malformed input.**
   L224-228 assumes `FZ_Theta`/`CZ_Theta` (or `_Relative_Z` variants) columns exist for every row with no validation or a clear error message pointing at what's expected — a wrong or third-party CSV just crashes with a confusing traceback.

7. **Divide-by-zero crash on an empty band-power CSV.**
   L241-244 / L253-254: if the CSV has 0 data rows, `self._n = 0`, and the first `update()` call does `self._idx % self._n` → `ZeroDivisionError`. No guard for empty input.

## MEDIUM

8. **Gameplay tuning is all unnamed magic numbers.**
   Climb speed `70 * dt` (L667), fall speed `30 * dt` (L680), sustain-decay factor `dt * 0.4` (L679), star-speed multiplier `rocket_speed * 40` (L295), flame scaling, etc. are inline literals with no named constants — makes rebalancing difficulty error-prone and undocumented.

9. **Fixed, non-resizable 520×700 window** (L56) with no DPI scaling — can render very small on high-resolution displays; no windowed/fullscreen toggle.

10. **No recalibration path.** If the 20 s live baseline capture is contaminated (child fidgets, talks, electrode shifts), the only recovery is quitting and relaunching the whole program — there's no in-game "recalibrate" action.

11. **Overly broad exception handling when starting an EEG source.**
    L779-793 catches a bare `Exception` around live/CSV/sim source construction and prints one generic message — LSL-not-found, missing CSV, and bad hardware errors all look identical to the user, making field troubleshooting harder.

## LOW / POLISH

12. **Distractor palette may not be colorblind-safe.** All three `DISTRACT_CLR` colors (L86) are hue-only distinguishable (purple/pink/blue) with no shape or pattern cue — relevant since ADHD often co-occurs with other visual-processing differences in the target population.

13. **No audio or haptic feedback**, only visual score popups (L327-344) — a missed engagement/reward channel for a game specifically designed to sustain a child's attention.

14. **No automated tests** for the signal-processing math (`_bandpower`, `_compute_tbr`, `_tbr_to_focus`) even though correctness of that math is the actual scientific claim behind the game.
