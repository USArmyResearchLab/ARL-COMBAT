# Python script to identify grain boundary particles (see Fortunato et al., ARL-TR-8213 (2017), Table A-7)
import combat

# uncomment for parallel version
# from mpi4py import MPI
# comm = MPI.COMM_WORLD

# uncomment for parallel version
# s = combat.System.from_dump('dump.polycrystal', comm=comm, cutoff=10.0)
s = combat.System.from_dump('dump.polycrystal', cutoff=5.0)
s.num_grains = len(set(s.particles['mol'].values))
s.make_neigh_tree()
s.define_grain_boundaries()
s.shrink_boundary_identifiers()
s.particles = s.particles[s.particles['num_grains'] > 1]
del s.particles['neighbors']
# uncomment for parallel version
#...if comm.rank==0:
#print('gathering global number of particles.)
#nparticles=comm.gather(len(s.particles[s.particles['ghost']==0]), root=0)
#if comm.rank == 0:
#    s.global_particles = np.sum(nparticles)
#    s.dump_header('dump.header')
#s.write_dump('dump.{}p'.format(comm.rank), header=False, ghost=False)
#comm.Barrier()
#if comm.rank == 0:
#    print('concatenating')
#    Popen('cat dump.header dump.*p > dump.boundaries, shell=True).communicate()
#    Popen('rm dump.header dump.*p', shell=True).communicate()
s.write_dump('dump.boundaries')
