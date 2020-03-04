# Python script to update molecule information in a LAMMPS dump and data file (see Fortunato et al., ARL-TR-8213 (2017), Table A-3)
import combat
s = combat.System.from_data('data.polycrystal', atom_style='atomic')
s.particles['mol'] = s.particles['type']
s.particles['type'] = 1
s.write_dump('dump.polycrystal')
s.write_data('data.polycrystal.molecules', atom_style='molecular')
