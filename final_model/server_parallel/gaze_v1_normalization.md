# GazeNet v1 spatial normalization contract

This contract applies only inside the isolated GazeNet v1 observation builder.
It does not change Partial memory storage or the ActionNet observation.

## Audited raw representation

`OvercookedBackend.canonical_pair` converts the 20 channels returned by
`lossless_state_encoding` to contiguous `float32` after multiplying every
value by 255. The Partial memory snapshot preserves those 20 values. Its
freshness channel is constructed as `freshness / MAX_FRESHNESS`, where
`MAX_FRESHNESS = 5`. The four remembered preview planes use marker values
0 and 255.

The canonical array axes are `(x, y, channel)` and the required raw shape is
exactly `(8, 5, 25)`.

| Channel | Semantic meaning | Canonical raw representation | Valid raw values | GazeNet v1 normalization | Normalized range |
|---:|---|---|---|---|---|
| 0 | self position memory | binary location mask multiplied by 255 | `{0, 255}` | `raw / 255` | `{0, 1}` |
| 1 | partner position memory | binary location mask multiplied by 255 | `{0, 255}` | `raw / 255` | `{0, 1}` |
| 2-5 | self orientation memory (four directions) | binary direction masks multiplied by 255 | `{0, 255}` | `raw / 255` | `{0, 1}` |
| 6-9 | partner orientation memory (four directions) | binary direction masks multiplied by 255 | `{0, 255}` | `raw / 255` | `{0, 1}` |
| 10 | pot location | static binary location mask multiplied by 255 | `{0, 255}` | `raw / 255` | `{0, 1}` |
| 11 | counter location | static binary location mask multiplied by 255 | `{0, 255}` | `raw / 255` | `{0, 1}` |
| 12 | onion-dispenser location | static binary location mask multiplied by 255 | `{0, 255}` | `raw / 255` | `{0, 1}` |
| 13 | dish-dispenser location | static binary location mask multiplied by 255 | `{0, 255}` | `raw / 255` | `{0, 1}` |
| 14 | serving location | static binary location mask multiplied by 255 | `{0, 255}` | `raw / 255` | `{0, 1}` |
| 15 | pot onion-count memory | literal semantic count 0-3 multiplied by 255 | `{0, 255, 510, 765}` | `raw / (3 * 255)` | `{0, 1/3, 2/3, 1}` |
| 16 | pot cook-time memory | literal semantic cook time 0-20 multiplied by 255 | `{0, 255, ..., 5100}` in steps of 255 | `raw / (20 * 255)` | `{0, .05, ..., 1}` |
| 17 | soup location memory | binary location mask multiplied by 255 | `{0, 255}` | `raw / 255` | `{0, 1}` |
| 18 | dish location memory | binary location mask multiplied by 255 | `{0, 255}` | `raw / 255` | `{0, 1}` |
| 19 | onion location memory | binary location mask multiplied by 255 | `{0, 255}` | `raw / 255` | `{0, 1}` |
| 20 | freshness | already normalized Action-memory freshness | float32 `{0, .2, .4, .6, .8, 1}` | unchanged | same six values |
| 21 | remembered H1 preview | binary marker plane | `{0, 255}` | `raw / 255` | `{0, 1}` |
| 22 | remembered H2 preview | binary marker plane | `{0, 255}` | `raw / 255` | `{0, 1}` |
| 23 | remembered H3 preview | binary marker plane | `{0, 255}` | `raw / 255` | `{0, 1}` |
| 24 | remembered H4 preview | binary marker plane | `{0, 255}` | `raw / 255` | `{0, 1}` |

Cook-time validity follows the encoder's representable 0-20 contract. Under
the canonical instant-ready transition, decision-boundary states commonly
contain endpoint values, but the observation contract still accepts every
integer semantic cook time the encoder can represent.

## Construction and validation

The builder copies the source into a new contiguous `float32` array, requires
finite values and exact membership in the per-channel sets above, and raises
on a mismatch. It never clips spatial memory. After semantic normalization it
appends one fixation plane containing exactly one value 1 at the current tile
and 0 elsewhere, producing `(8, 5, 26)`.

The actor and separate critic consume this final array directly. Neither
network applies `/ 255` or any other blanket spatial rescaling.

The seven-value context remains exactly:

`[clamp(remaining_ms / 600, 0, 1), T200, T400, T600, H0, H2, H4]`.
