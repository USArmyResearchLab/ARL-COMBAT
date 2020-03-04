# Python script to compute weight-averaged local temperature (see Fortunato et al., ARL-TR-8213 (2017), Table A-8)
import combat
import numpy as np

# uncomment for parallel version
# from mpi4py import MPI
# comm = MPI.COMM_WORLD

# uncomment for parallel version
# s = combat.System.from_dump('dump.singlecrystal', comm=comm, cutoff=12)
s = combat.System.from_dump('dump.singlecrystal', cutoff=12)

s.make_neigh_tree()

s.temperature(mass=100)
s.weighted_avg('temperature', 12)
def quad(dist):
    ratio = dist/12
    return 15/2/np.pi*np.power(1-ratio, 2)
s.weighted_avg('temperature', 12, weighting=quad)

def avg(dist):
    return [1 for _ in dist]

s.weighted_avg('temperature', 12, weighting=avg)

del s.particles['iloc_neighbors']

# uncomment for parallel version
#if comm.rank==0:
#    print('gathering global number of particles.)
#nparticles=comm.gather(len(s.particles[s.particles['ghost']==0]), root=0)
#if comm.rank == 0:
#    s.global_particles = np.sum(nparticles)
#    s.dump_header('dump.header')
#s.write_dump('dump.{}p'.format(comm.rank), header=False, ghost=False)
#comm.Barrier()
#if comm.rank == 0:
#    print('concatenating')
#    Popen('cat dump.header dump.*p > dump.weighting', shell=True).communicate()
#    Popen('rm dump.header dump.*p', shell=True).communicate()
s.write_dump('dump.weighting')
