# Closing a silence run that reaches the end of the file

**Where:** `bin/narrate.py`, `snap_to_silence`

The reference clip is cut at a silence rather than mid-word. A hard cut
through speech gives the model a chopped-off word as its only example of
what audio looks like just before end-of-speech, and the ASR transcript
then describes audio that stops mid-syllable. Measured: a hard 20 s slice
cut at RMS 0.149 — full speech level — and the voice audibly drifted.

## What breaks if you change it back

The loop that finds silences only closes a run when speech **resumes after
it**. A silence that runs to the end of the file is therefore never
recorded, and the snap falls back to the last *interior* gap.

On this project's clip — speech to 55.8 s, then 0.2 s of silence to the end
— asking for 56 s snapped to **51.7 s**, discarding 4.3 s (8 %) of the
reference for no reason and with no warning.

`clamp_internal_silence` had always handled the trailing case. This function
did not, and that asymmetry between two functions in the same file was the
whole bug.

## Two thresholds, deliberately different

An **interior** gap needs 0.3 s to be confidently a pause rather than a stop
consonant, because the cut lands in the *middle* of it and must not clip
speech on either side.

The **end of the file** only has to show the speaker finished, which 0.1 s
does — and the cut goes at the file end, not the run's midpoint, because
there is no speech after it to protect and the midpoint would throw away
half the trailing audio.

⚠️ Holding the tail to the interior rule is what cost the 4.3 s above. If
you unify the thresholds, you reintroduce it.
