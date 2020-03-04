# Python script to update molecules from bond topology (see Fortunato et al., ARL-TR-8213 (2017), Table A-5)
import combat as pdlmps
s = pdlmps.System.from_data('data.expanded_composite', atom_style='molecular')
s.molecules_from_bonds()
s.write_data('data.expanded_composite.molecules', atom_style='molecular')
