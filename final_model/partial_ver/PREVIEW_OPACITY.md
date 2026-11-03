# Peripheral preview opacity

The canonical Partial observation remains the v7 step-separated H1--H4
representation. Channels 21--24 retain temporal identity, so overlapping H2
and H4 previews on one tile remain positive together. Renderer overlap does not
collapse or otherwise redefine those observation channels.

Direct PLUS acquisition is exact and consumes no opacity random draw. After a
peripheral tile passes the unchanged spatial-detection stage, opacity mode
`off` also acquires all positive preview steps exactly. In opacity mode
`actual`, a preview-bearing tile creates one shared visibility event. Its
probability is the maximum renderer source alpha among the tile's positive
steps divided by 255. A success preserves every original positive H bit; a miss
clears all four preview bits while state and tile freshness are still refreshed.
Spatial failure leaves memory untouched and consumes no opacity draw.

The shared event is a visibility model for a rendered overlapping peripheral
cue. The `alpha_max / 255` mapping is an explicit modeling assumption and is not
a claim that the model reproduces human psychophysics.
