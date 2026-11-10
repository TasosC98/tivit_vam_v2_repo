"""PianoVAM visual (video-only) piano transcription.

Recognise pressed piano keys (onset + key-release offset + pitch) directly
from the keyboard video and export a MIDI file. Audio is never used.
"""

__version__ = "0.1.0"

N_KEYS = 88          # MIDI pitches 21 (A0) .. 108 (C8)
PITCH_MIN = 21
PITCH_MAX = 108
