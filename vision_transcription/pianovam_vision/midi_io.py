"""Write decoded notes to a Standard MIDI File (and read GT MIDI if needed)."""
from __future__ import annotations

from pathlib import Path
from typing import List

from .labels import Note


def write_midi(notes: List[Note], path: str | Path, program: int = 0) -> None:
    """Write notes to a .mid file. program 0 = Acoustic Grand Piano."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=program, name="visual-transcription")
    for n in notes:
        if n.offset <= n.onset:
            continue
        inst.notes.append(
            pretty_midi.Note(
                velocity=int(max(1, min(127, n.velocity))),
                pitch=int(n.pitch),
                start=float(n.onset),
                end=float(n.offset),
            )
        )
    pm.instruments.append(inst)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(path))
