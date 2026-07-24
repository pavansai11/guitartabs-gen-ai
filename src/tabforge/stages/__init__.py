"""Pipeline stages. Heavy audio dependencies are imported lazily inside the
functions that need them so the pure-logic stages (segment, articulation,
fretboard) import and run with numpy/scipy alone."""
