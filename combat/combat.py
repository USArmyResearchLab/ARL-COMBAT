# -------------------------------------------------------------------------
#    The ARL Computational Model Builder and Analysis Toolkit (ARL COMBAT) 
#    https://github.com/USArmyResearchLab/ARL-COMBAT
#    US Army CCDC Army Research Laboratory
#
#    This software is distributed under the Creative Commons Zero 1.0 
#    Universal (CC0 1.0) Public Domain Dedication.
#
#    See the README file in the top-level ARL COMBAT directory.
# ------------------------------------------------------------------------- 

# ------------------------------------------------------------------------- 
#    Contributing Authors:  Michael Fortunato and Joseph Mattson
# ------------------------------------------------------------------------- 

import numpy as np
import pandas as pd
import multiprocessing
from scipy import spatial
import networkx as nx
import itertools

lammps_atom_style = {
    'atomic' : ['id', 'type', 'x', 'y', 'z'],
    'molecular': ['id', 'mol', 'type', 'x', 'y', 'z'],
    'full': ['id', 'mol', 'type', 'charge', 'x', 'y', 'z'],
    'dpd': ['id', 'type', 'dpd_theta', 'x', 'y', 'z'],
    'dpd molecular': ['id', 'type', 'x', 'y', 'z', 'dpd_theta', 'mol']
}

class System(object):
    def __init__(self, xlo=None, xhi=None, ylo=None, yhi=None, zlo=None, zhi=None, particles=None, bonds=None, angles=None):
        self.xlo = xlo
        self.xhi = xhi
        self.ylo = ylo
        self.yhi = yhi
        self.zlo = zlo
        self.zhi = zhi
        if particles is not None:
            self.particles = particles
        if bonds is not None:
            self.bonds = bonds
        if angles is not None:
            self.angles = angles
        self.unwrapped = False
        self.neighbor_tree = None
        
    def write_data(self, fname=None, atom_style='molecular', images=False):
        if atom_style not in lammps_atom_style:
            print('only the following atom styles are supported thus far:')
            for sty in lammps_atom_style:
                print('  {}'.format(sty))
            return
        if not fname:
            print('no filename given')
            return
        #particles_str = np.savetxt(strio, self.particles.values, fmt='%d %d %d %f %f %f')
        with open(fname, 'w') as f:
            f.write('comment line\n')
            f.write('\n')
            f.write('{} atoms\n'.format(len(self.particles)))
            f.write('{} atom types\n'.format(self.particles['type'].max()))
            if hasattr(self, 'bonds'):
                f.write('{} bonds\n'.format(len(self.bonds)))
                f.write('{} bond types\n'.format(self.bonds['type'].max()))
            if hasattr(self, 'angles'):
                f.write('{} angles\n'.format(len(self.angles)))
                f.write('{} angle types\n'.format(self.angles['type'].max()))
            if hasattr(self, 'dihedrals'):
                f.write('%s dihedrals\n' % len(self.dihedrals))
                f.write('{} dihedral types\n'.format(self.dihedrals['type'].max()))
            f.write('\n')
            f.write('{} {} xlo xhi\n'.format(self.xlo, self.xhi))
            f.write('{} {} ylo yhi\n'.format(self.ylo, self.yhi))
            f.write('{} {} zlo zhi\n'.format(self.zlo, self.zhi))
            f.write('\n')
            f.write('Atoms\n')
            f.write('\n')
            data_args = lammps_atom_style[atom_style][1:]
            self.particles[data_args].to_string(f, header=False, index_names=False)
            f.write('\n')
            if hasattr(self, 'bonds'):
                f.write('\n')
                f.write('Bonds\n')
                f.write('\n')
                self.bonds[['type', 'a', 'b']].to_string(f, header=False, index_names=False)
                f.write('\n')
            if hasattr(self, 'angles'):
                f.write('\n')
                f.write('Angles\n')
                f.write('\n')
                self.angles[['type', 'a', 'b', 'c']].to_string(f, header=False, index_names=False)
                f.write('\n')
            
    def dump_header(self, fname=None):
        if not fname:
            print('no filename given')
            return
        with open(fname, 'w') as f:
            f.write('ITEM: TIMESTEP\n')
            f.write('0\n')
            f.write('ITEM: NUMBER OF ATOMS\n')
            f.write('{}\n'.format(self.global_particles))
            f.write('ITEM: BOX BOUNDS pp pp pp\n')
            f.write('{} {}\n'.format(self.global_xlo, self.global_xhi))
            f.write('{} {}\n'.format(self.global_ylo, self.global_yhi))
            f.write('{} {}\n'.format(self.global_zlo, self.global_zhi))
            f.write('ITEM: ATOMS {}\n'.format('id ' + ' '.join(list(self.particles))))
            
    def dump_owned(self, fname=None, pid=0, header=False, ghost=False):
        if not fname:
            print('no filename given')
            return
        with open(fname, 'w') as f:
            if len(self.particles) > 2000000:
                n_chunks = int(len(self.particles)/1000000)
                chunks = np.array_split(self.particles, n_chunks)
                for chunk in chunks:
                    if ghost:
                        chunk.to_string(f, header=False, index_names=False)
                    else:
                        chunk[chunk['ghost']==0].to_string(f, header=False, index_names=False)
            else:
                if ghost:
                    self.particles.to_string(f, header=False, index_names=False)
                else:
                    self.particles[self.particles['ghost']==0].to_string(f, header=False, index_names=False)
            f.write('\n')
            
    def write_dump(self, fname=None, header=True, ghost=False):
        if not fname:
            print('no filename given')
            return
        with open(fname, 'w') as f:
            if header:
                f.write('ITEM: TIMESTEP\n')
                f.write('0\n')
                f.write('ITEM: NUMBER OF ATOMS\n')
                if ghost or 'ghost' not in self.particles:
                    f.write('{}\n'.format(len(self.particles)))
                else:
                    f.write('{}\n'.format(len(self.particles[self.particles['ghost']==0])))
                f.write('ITEM: BOX BOUNDS pp pp pp\n')
                f.write('{} {}\n'.format(self.xlo, self.xhi))
                f.write('{} {}\n'.format(self.ylo, self.yhi))
                f.write('{} {}\n'.format(self.zlo, self.zhi))
                f.write('ITEM: ATOMS {}\n'.format('id ' + ' '.join(list(self.particles))))
            if ghost or 'ghost' not in self.particles:
                self.particles.to_csv(f, sep=' ', chunksize=100000, header=False, index_label=False, na_rep=-1)
            else:
                self.particles[self.particles['ghost']==0].to_csv(f, sep=' ', chunksize=100000, header=False, index_label=False, na_rep=-1)

    @classmethod
    def from_dump(cls, fname=None, comm=None, cutoff=0.0, chunksize=1000000, periodic_dims=['x', 'y', 'z'], unscale=False, grain_identifier='mol'):
        # temporary solution for getting periodic ghost neighbors when running in serial
        class FakeComm(object):
            def __init__(self):
                self.size = 1
                self.rank = 0
        if comm is None:
            comm = FakeComm()
        if not fname:
            print('no filename given')
            return
        with open(fname) as f:
            next(f)
            t_step = int(next(f))
            next(f)
            n_particles = int(next(f))
            next(f)
            xlo, xhi = list(map(float, next(f).split()))
            ylo, yhi = list(map(float, next(f).split()))
            zlo, zhi = list(map(float, next(f).split()))
            data_args = next(f).split()[2:]
        particles = pd.DataFrame(columns=data_args[1:])
        s = cls(xlo, xhi, ylo, yhi, zlo, zhi, particles=particles)
        s.num_grains = 0
        s.proc_dist(comm.size)
        s.set_proc_bounds(comm.rank)
        reader = pd.read_csv(fname, sep='\s+', skiprows=9, names=data_args, index_col='id', chunksize=min(chunksize, n_particles))
        for chunk in reader:
            s_tmp = cls(xlo, xhi, ylo, yhi, zlo, zhi, particles=chunk)
            if grain_identifier in s.particles:
                s.num_grains = max(s.num_grains, s_tmp.particles[grain_identifier].max())
            if unscale:
                s_tmp.unscale()
            s_tmp.wrap()
            s_tmp.proc_dist(comm.size)
            s_tmp.set_proc_bounds(comm.rank)
            s_tmp.own()
            s_tmp.ghost(comm.rank, cutoff)
            if periodic_dims:
                s_tmp.periodic_neighbors(comm.rank, cutoff, periodic_dims)
                s_tmp.ghost(comm.rank, cutoff)
            s.particles = s.particles.append(s_tmp.particles[(s_tmp.particles['own']==comm.rank) | (s_tmp.particles['ghost']==1)])
        s.particles['orig_image'] = abs(s.particles.index)
        pbc_images = s.particles.loc[s.particles['own'] < 0]
        s.particles = s.particles.loc[s.particles['own'] >= 0]
        max_id = max(s.particles.index.values)+1
        pbc_images['new_index'] = range(max_id, max_id+len(pbc_images))
        pbc_images.set_index('new_index', inplace=True)
        s.particles = s.particles.append(pbc_images)
        return s
        
    def read_bond_dump(self, fname, types=False):
        names = ['a', 'b']
        if types:
            names = ['type'] + names
        self.bonds = pd.read_csv(fname, skiprows=9, names=names, sep='\s+')

    @classmethod
    def from_data(cls, fname=None, atom_style='molecular', images=True, index_id=True):
        if not fname:
            print('no filename given')
            return
        if atom_style not in lammps_atom_style:
            print('only molecular atom style supported thus far')
            return
        data_start = bonds_start = angles_start = 0
        particles = bonds = angles = dihedrals = None
        n_particles = 0
        images_present = None
        with open(fname) as f:
            lineno = 0
            for line in f:
                lineno += 1
                if 'atoms' in line:
                    n_particles = int(line.split()[0])
                if 'bonds' in line:
                    n_bonds = int(line.split()[0])
                if 'angles' in line:
                    n_angles = int(line.split()[0])
                if 'xlo' in line:
                    xlo, xhi = list(map(float, line.split()[:2]))
                if 'ylo' in line:
                    ylo, yhi = list(map(float, line.split()[:2]))
                if 'zlo' in line:
                    zlo, zhi = list(map(float, line.split()[:2]))
                if 'Atoms' in line:
                    data_start = lineno + 1
                    line = next(f)
                    lineno += 1
                    line = next(f)
                    lineno += 1
                    if len(line.split()) == len(lammps_atom_style[atom_style]) + 3:
                        images_present = True
                if 'Bonds' in line:
                    bonds_start = lineno + 1
                if 'Angles' in line:
                    angles_start = lineno + 1
        data_args = lammps_atom_style[atom_style]
        if images and images_present:
            data_args.extend(['ix', 'iy', 'iz'])
        if data_start and n_particles:
            particles = pd.read_csv(fname, sep='\s+', skiprows=data_start, nrows=n_particles, names=data_args)
        if bonds_start and n_bonds:
            bonds = pd.read_csv(fname, sep='\s+', skiprows=bonds_start, nrows=n_bonds, names=['id', 'type', 'a', 'b'], index_col='id')
        if angles_start and n_angles:
            angles = pd.read_csv(fname, sep='\s+', skiprows=angles_start, nrows=n_angles, names=['id', 'type', 'a', 'b', 'c'], index_col='id')
        if index_id and 'id' in particles:
            particles.set_index('id', inplace=True)
        return cls(xlo, xhi, ylo, yhi, zlo, zhi, particles=particles, bonds=bonds, angles=angles)

    def proc_id(self, px, py, pz):
        return pz*self.npy*self.npx + py*self.npx + px

    def proc_ijk(self, pid):
        if pid < self.npx:
            return pid, 0, 0
        pz = int(pid//(self.npx*self.npy))
        py = int((pid-pz*self.npy*self.npx)//self.npy)
        px = int((pid-pz*self.npy*self.npx-py*self.npy))
        return px, py, pz

    def proc_dist(self, numproc):
        self.numproc = numproc
        self.npx, self.npy, self.npz = proc_array(numproc)
        self.proc_dx = (self.xhi-self.xlo)/self.npx
        self.proc_dy = (self.yhi-self.ylo)/self.npy
        self.proc_dz = (self.zhi-self.zlo)/self.npz

    def own(self):
        if len(self.particles) == 0:
            return
        px = ((self.particles['x']-self.global_xlo)//self.proc_dx).astype(int)
        py = ((self.particles['y']-self.global_ylo)//self.proc_dy).astype(int)
        pz = ((self.particles['z']-self.global_zlo)//self.proc_dz).astype(int)
        self.particles['own'] = self.proc_id(px, py, pz)

    def ghost(self, pid, cutoff):
        if len(self.particles) == 0:
            return
        if cutoff >= self.proc_dx/2 or cutoff >= self.proc_dy/2 or cutoff >= self.proc_dz/2:
            print('warning: cutoff > domain/2')
        px, py, pz = self.proc_ijk(pid)
        self.particles['ghost'] = (
            (self.particles['own'] != pid) &
            (self.particles['x'] < self.xhi+cutoff) & 
            (self.particles['x'] > self.xlo-cutoff) &
            (self.particles['y'] < self.yhi+cutoff) & 
            (self.particles['y'] > self.ylo-cutoff) & 
            (self.particles['z'] < self.zhi+cutoff) & 
            (self.particles['z'] > self.zlo-cutoff)
        ).astype(int)
        self.particles = self.particles.drop_duplicates(subset=['x', 'y', 'z'])

    def periodic_neighbors(self, pid, cutoff, periodic_dims=['x', 'y', 'z']):
        if len(self.particles) == 0:
            return
        self.particles = self.particles.loc[(
            (self.particles['own'] == pid) |
            (self.particles['ghost'] == 1) |
            (self.particles['x'] >= self.global_xhi-cutoff) |
            (self.particles['y'] >= self.global_yhi-cutoff) |
            (self.particles['z'] >= self.global_zhi-cutoff) |
            (self.particles['x'] <= self.global_xlo+cutoff) |
            (self.particles['y'] <= self.global_ylo+cutoff) |
            (self.particles['z'] <= self.global_zlo+cutoff)
        )]
        for dim in periodic_dims:
            global_dim_hi = getattr(self, 'global_{}hi'.format(dim))
            global_dim_lo = getattr(self, 'global_{}lo'.format(dim))

            periodic_tmp = self.particles.loc[(
                (self.particles[dim] >= global_dim_hi-cutoff) &
                (self.particles[dim] <= global_dim_hi)
            )]
            periodic_tmp.is_copy = False
            periodic_tmp['own'] = -1
            periodic_tmp['id'] = -abs(periodic_tmp.index)
            periodic_tmp.set_index('id', inplace=True)
            periodic_tmp[dim] = periodic_tmp[dim] - (global_dim_hi - global_dim_lo)
            self.particles = self.particles.append(periodic_tmp)
            
            periodic_tmp = self.particles.loc[(
                (self.particles[dim] >= global_dim_lo) &
                (self.particles[dim] <= global_dim_lo+cutoff)
            )]
            periodic_tmp.is_copy = False
            periodic_tmp['own'] = -1
            periodic_tmp['id'] = -abs(periodic_tmp.index)
            periodic_tmp.set_index('id', inplace=True)
            periodic_tmp[dim] = periodic_tmp[dim] + (global_dim_hi - global_dim_lo)
            self.particles = self.particles.append(periodic_tmp)
            
        for dim1, dim2 in itertools.combinations(periodic_dims, 2):
            global_dim1_hi = getattr(self, 'global_{}hi'.format(dim1))
            global_dim1_lo = getattr(self, 'global_{}lo'.format(dim1))
            
            global_dim2_hi = getattr(self, 'global_{}hi'.format(dim2))
            global_dim2_lo = getattr(self, 'global_{}lo'.format(dim2))
            
            periodic_tmp = self.particles.loc[(
                (self.particles[dim1] >= global_dim1_hi-cutoff) &
                (self.particles[dim1] <= global_dim1_hi) &
                (self.particles[dim2] >= global_dim2_hi-cutoff) &
                (self.particles[dim2] <= global_dim2_hi)
            )]
            periodic_tmp.is_copy = False
            periodic_tmp['own'] = -1
            periodic_tmp['id'] = -abs(periodic_tmp.index)
            periodic_tmp.set_index('id', inplace=True)
            periodic_tmp[dim1] = periodic_tmp[dim1] - (global_dim1_hi - global_dim1_lo)
            periodic_tmp[dim2] = periodic_tmp[dim2] - (global_dim2_hi - global_dim2_lo)
            self.particles = self.particles.append(periodic_tmp)
            
            periodic_tmp = self.particles.loc[(
                (self.particles[dim1] >= global_dim1_hi-cutoff) &
                (self.particles[dim1] <= global_dim1_hi) &
                (self.particles[dim2] >= global_dim2_lo) &
                (self.particles[dim2] <= global_dim2_lo+cutoff)
            )]
            periodic_tmp.is_copy = False
            periodic_tmp['own'] = -1
            periodic_tmp['id'] = -abs(periodic_tmp.index)
            periodic_tmp.set_index('id', inplace=True)
            periodic_tmp[dim1] = periodic_tmp[dim1] - (global_dim1_hi - global_dim1_lo)
            periodic_tmp[dim2] = periodic_tmp[dim2] + (global_dim2_hi - global_dim2_lo)
            self.particles = self.particles.append(periodic_tmp)
            
            periodic_tmp = self.particles.loc[(
                (self.particles[dim1] >= global_dim1_lo) &
                (self.particles[dim1] <= global_dim1_lo+cutoff) &
                (self.particles[dim2] >= global_dim2_hi-cutoff) &
                (self.particles[dim2] <= global_dim2_hi)
            )]
            periodic_tmp.is_copy = False
            periodic_tmp['own'] = -1
            periodic_tmp['id'] = -abs(periodic_tmp.index)
            periodic_tmp.set_index('id', inplace=True)
            periodic_tmp[dim1] = periodic_tmp[dim1] + (global_dim1_hi - global_dim1_lo)
            periodic_tmp[dim2] = periodic_tmp[dim2] - (global_dim2_hi - global_dim2_lo)
            self.particles = self.particles.append(periodic_tmp)
            
            periodic_tmp = self.particles.loc[(
                (self.particles[dim1] >= global_dim1_lo) &
                (self.particles[dim1] <= global_dim1_lo+cutoff) &
                (self.particles[dim2] >= global_dim2_lo) &
                (self.particles[dim2] <= global_dim2_lo+cutoff)
            )]
            periodic_tmp.is_copy = False
            periodic_tmp['own'] = -1
            periodic_tmp['id'] = -abs(periodic_tmp.index)
            periodic_tmp.set_index('id', inplace=True)
            periodic_tmp[dim1] = periodic_tmp[dim1] + (global_dim1_hi - global_dim1_lo)
            periodic_tmp[dim2] = periodic_tmp[dim2] + (global_dim2_hi - global_dim2_lo)
            self.particles = self.particles.append(periodic_tmp)

    def set_proc_bounds(self, pid):
        px, py, pz = self.proc_ijk(pid)
        self.global_xlo = self.xlo
        self.global_xhi = self.xhi
        self.global_ylo = self.ylo
        self.global_yhi = self.yhi
        self.global_zlo = self.zlo
        self.global_zhi = self.zhi
        self.xlo, self.xhi = px*self.proc_dx-self.xlo, (px+1)*self.proc_dx-self.xlo
        self.ylo, self.yhi = py*self.proc_dy-self.ylo, (py+1)*self.proc_dy-self.ylo
        self.zlo, self.zhi = pz*self.proc_dz-self.zlo, (pz+1)*self.proc_dz-self.zlo
        
    def comm_molecules(self, comm):
        if 'mol' not in self.particles:
            print('molecules not present')
            return
        gather_mols = comm.allgather(list(set(self.particles['mol'])))
        global_mols = list(set(np.concatenate(gather_mols)))
        rank_mols = np.array_split(global_mols, comm.size)
        self.mols = list(np.array_split(global_mols, comm.size)[comm.rank])
        for rank in range(comm.size):
            if comm.rank == 0:
                print('gather {}'.format(rank))
            mol_particles = comm.gather(self.particles.loc[self.particles['mol'].isin(list(rank_mols[rank]))], root=rank)
            if rank == comm.rank:
                self.mol_particles = pd.concat(mol_particles, sort=False)

    def set_box_edges(self, buffer=0):
        if len(self.particles) == 0:
            return
        self.xlo=np.min(self.particles['x'])-buffer
        self.xhi=np.max(self.particles['x'])+buffer
        self.ylo=np.min(self.particles['y'])-buffer
        self.yhi=np.max(self.particles['y'])+buffer
        self.zlo=np.min(self.particles['z'])-buffer
        self.zhi=np.max(self.particles['z'])+buffer

    def unscale(self):
        if 'xs' not in self.particles or 'ys' not in self.particles or 'zs' not in self.particles:
            print('scaled coordinates not found, cannot unscale')
            return
        self.particles['x'] = self.xlo + self.particles['xs']*(self.xhi-self.xlo)
        self.particles['y'] = self.ylo + self.particles['ys']*(self.yhi-self.ylo)
        self.particles['z'] = self.zlo + self.particles['zs']*(self.zhi-self.zlo)
        del self.particles['xs']
        del self.particles['ys']
        del self.particles['zs']
        
    def set_images(self):
        self.particles['ix'] = int((self.particles['x']-self.xlo)//(self.xhi-self.xlo))
        self.particles['iy'] = int((self.particles['y']-self.ylo)//(self.yhi-self.ylo))
        self.particles['iz'] = int((self.particles['z']-self.zlo)//(self.zhi-self.zlo))

    def image(self):
        if not 'ix' in self.particles or not 'iy' in self.particles or not 'iz' in self.particles:
            print('error: images not present')
            return
        if self.unwrapped == True:
            print('system already unwrapped')
            return
        self.particles['x'] = self.particles['x']+self.particles['ix']*(self.xhi-self.xlo)
        self.particles['y'] = self.particles['y']+self.particles['iy']*(self.yhi-self.ylo)
        self.particles['z'] = self.particles['z']+self.particles['iz']*(self.zhi-self.zlo)
        self.unwrapped = True
        
    def wrap(self):
        if len(self.particles) == 0:
            return
        while np.any(self.particles['x'] > self.xhi):
            self.particles.loc[self.particles['x'] > self.xhi, ['x']] -= (self.xhi-self.xlo)
        while np.any(self.particles['y'] > self.yhi):
            self.particles.loc[self.particles['y'] > self.yhi, ['y']] -= (self.yhi-self.ylo)
        while np.any(self.particles['z'] > self.zhi):
            self.particles.loc[self.particles['z'] > self.zhi, ['z']] -= (self.zhi-self.zlo)
        while np.any(self.particles['x'] < self.xlo):
            self.particles.loc[self.particles['x'] < self.xlo, ['x']] += (self.xhi-self.xlo)
        while np.any(self.particles['y'] < self.ylo):
            self.particles.loc[self.particles['y'] < self.ylo, ['y']] += (self.yhi-self.ylo)
        while np.any(self.particles['z'] < self.zlo):
            self.particles.loc[self.particles['z'] < self.zlo, ['z']] += (self.zhi-self.zlo)
            self.unwrapped = False
            
    def bond_lengths(self):
        if not hasattr(self, 'bonds'):
            print('no bonds present')
            return
        self.bonds['length'] = np.linalg.norm(self.particles.loc[self.bonds['a'], ['x', 'y', 'z']].values-self.particles.loc[self.bonds['b'], ['x', 'y', 'z']].values, axis=1)

    def molecules_from_bonds(self):
        if not hasattr(self, 'bonds'):
            print('no bonds present')
            return
        nmols = len(set(self.particles['mol']))
        allgraph = nx.Graph()
        allgraph.add_edges_from(self.bonds[['a', 'b']].values)
        graphs = nx.connected_components(allgraph)
        for m, g in enumerate(graphs, 1):
            self.particles.loc[g, 'mol'] = m + nmols

    def convex_hull(self, data=None, molecule=None):
        if molecule:
            data = self.particles.loc[self.particles['mol'] == molecule]
        elif not data:
            data = self.particles
            
        hull = spatial.ConvexHull(data[['x','y','z']].values)
        return hull.area, hull.volume

    def rg(self, data=None, molecule=None, vector=False):
        if molecule:
            data = self.particles.loc[self.particles['mol'] == molecule]
        elif not data:
            data = self.particles

        center = np.array([data['x'].mean(), data['y'].mean(), data['z'].mean()])

        rg_vec = np.sqrt(np.sum(np.power(data[['x', 'y', 'z']] - center, 2))/len(data))
        if vector:
            return rg_vec
        else:
            return np.linalg.norm(rg_vec)

    def rg_all(self):
        self.particles['mol_rg'] = self.particles['mol'].map({n: self.rg(molecule=n) for n in set(self.particles['mol'].values)})
        
    def sep_distances(self, cutoff=7.0):
        if self.neighbor_tree is None:
            self.make_neigh_tree()
        res = self.neighbor_tree.sparse_distance_matrix(spatial.cKDTree(self.particles.loc[self.particles['ghost']==0, ['x', 'y', 'z']].values), cutoff, output_type='coo_matrix')
        res.eliminate_zeros()
        return res.data

    def make_neigh_tree(self):
        if len(self.particles) == 0:
            self.neighbor_tree = spatial.cKDTree([[]])
        else:
            self.neighbor_tree = spatial.cKDTree(self.particles[['x', 'y', 'z']].values)

    def neighbors(self, p_id, cutoff):
        print('DEPRECATION WARNING: please use System.neighbors_of() in place of System.neighbors()')
        return self.neighbors_of(p_id, cutoff)
        
    def neighbors_of(self, p_id, cutoff):
        if len(self.particles) == 0:
            return []
        if not self.neighbor_tree:
            self.make_neigh_tree()
        neighbors = list(self.particles.iloc[self.neighbor_tree.query_ball_point(self.particles.loc[p_id, ['x', 'y', 'z']].values, cutoff)].index.values)
        neighbors.remove(p_id)
        return neighbors
        
    def neighbors_at(self, coords, cutoff):
        if len(self.particles) == 0:
            return []
        if not self.neighbor_tree:
            self.make_neigh_tree()
        neighbors = list(self.particles.iloc[self.neighbor_tree.query_ball_point(coords, cutoff)].index.values)
        return neighbors

    def neighbors_all(self, cutoff, remove_self=True):
        def remove_id(row):
            n = list(row['neighbors'])
            n.pop(n.index(row.name))
            return n
        if not self.neighbor_tree:
            self.make_neigh_tree()
        self.particles['neighbors'] = [np.take(self.particles.index.values, n) for n in self.neighbor_tree.query_ball_point(self.particles.loc[:, ['x', 'y', 'z']].values, cutoff)]
        if remove_self:
            self.particles['neighbors'] = self.particles.apply(remove_id, axis=1)
        
    def iloc_neighbors(self, cutoff, remove_self=True):
        def remove_id(row):
            n = list(row['iloc_neighbors'])
            n.pop(n.index(self.particles.index.get_loc(row.name)))
            return n
        self.particles['iloc_neighbors'] = self.neighbor_tree.query_ball_point(self.particles.loc[:, ['x', 'y', 'z']].values, cutoff)
        if remove_self:
            self.particles['iloc_neighbors'] = self.particles.apply(remove_id, axis=1)
        
    def closest_neighbor(self):
        if not self.neighbor_tree:
            self.make_neigh_tree()
        res = self.neighbor_tree.query(self.particles[['x', 'y', 'z']].values, 2)
        self.particles['closest_neigh'] = res[0][:, 1]

    def define_grain_boundaries(self, cutoff=7.0, grain_identifier='mol'):
        if 'neighbors' not in self.particles:
            self.neighbors_all(cutoff=cutoff)
        if grain_identifier not in self.particles:
            print('cannot use {} to identify grains'.format(grain_identifier))
        def neigh_mols(row):
            return sorted(list(set(self.particles.loc[row['neighbors'], grain_identifier])))
        self.particles['grains'] = self.particles.apply(neigh_mols, axis=1)
        self.particles['num_grains'] = self.particles['grains'].apply(len)
        def identifier_num(grains):
            identifierNum = int(0)
            for i in range(len(grains)): #creates identifier number for grain boundary based on the grains around it
                identifierNum += grains[i]*(self.num_grains**i)
            return identifierNum
        self.particles['interface'] = self.particles['grains'].apply(identifier_num)
        del self.particles['grains']
    
    def shrink_boundary_identifiers(self):
        smallVals = {}
        smallVal = 1
        for i in self.particles['interface']:
            if i not in smallVals:
                smallVals[i] = smallVal
                smallVal += 1
        def assign_new_identifiers(row):
            return smallVals[row]
        self.particles['small_interface'] = self.particles['interface'].apply(assign_new_identifiers)
    
    def shrink_boundary_identifiers_mpi(self, comm):
        if comm.rank == 0:
            interfaceVals = comm.allgather(list(set(self.particles['interface'])))
            smallVals = {}
            smallVal = 1
            for i in interfaceVals:
                for j in i:
                    if j not in smallVals:
                        smallVals[j] = smallVal
                        smallVal += 1
        else:
            smallVals = None
        comm.Barrier()
        smallVals = comm.bcast(smallVals, root=0)
        def assign_new_identifiers(row):
            return smallVals[row]
        self.particles['small_interface'] = self.particles['interface'].apply(assign_new_identifiers)
    
    def expunge_nonboundary_particles(self):
        self.particles = self.particles[self.particles['num_grains'] > 1]

    ## https://stackoverflow.com/questions/45624653/calculate-average-of-groups-of-dataframe-rows-given-by-2-d-lists-of-indices-with
    def weighted_avg(self, data, cutoff, weighting=None, harmonic=False, remake_neigh=True):
        if data not in self.particles:
            print('{} not present, cannot calculate local average'.format(data))
            return
        if remake_neigh:
            self.iloc_neighbors(cutoff=cutoff, remove_self=False)
        def lucy(dist):
            ratio = dist/cutoff
            return (1 + 3*ratio)*np.power(1-ratio, 3)
        if weighting is None:
            weighting = lucy
        neighs = self.particles['iloc_neighbors'].values.tolist()
        lengths = np.array([len(x) for x in neighs])
        pos = self.particles[['x', 'y', 'z']].values
        pos_neigh = pos[np.concatenate(neighs)]
        pos_self = pos.repeat(lengths, axis=0)
        dists = np.linalg.norm(pos_self - pos_neigh, axis=1)
        w = weighting(dists)
        values = self.particles[data].values[np.concatenate(neighs)]
        if harmonic:
            values = 1/values
        values = w*values
        positions = np.arange(len(neighs))
        
        self.particles['weighted_{}_'.format(weighting.__name__)+data] = np.bincount(
            positions.repeat(lengths),
            values
        ) / [np.sum(x) for x in np.split(w, np.cumsum(lengths)[:-1])]
        
        if harmonic:
            self.particles['weighted_{}_'.format(weighting.__name__)+data] = 1/self.particles['weighted_{}_'.format(weighting.__name__)+data].values
            
    def local_density(self, cutoff, weighting=None, remake_neigh=True):
        if remake_neigh:
            self.iloc_neighbors(cutoff=cutoff, remove_self=True)
        def lucy_dr(dist):
            ratio = dist/cutoff
            return (84./(5*np.pi*np.power(cutoff, 3)))*(1+3./2.*ratio)*np.power(1-ratio, 4)
        if weighting is None:
            weighting = lucy_dr
        neighs = self.particles['iloc_neighbors'].values.tolist()
        lengths = np.array([len(x) for x in neighs])
        pos = self.particles[['x', 'y', 'z']].values
        pos_neigh = pos[np.concatenate(neighs)]
        pos_self = pos.repeat(lengths, axis=0)
        dists = np.linalg.norm(pos_self - pos_neigh, axis=1)
        w = weighting(dists)
        positions = np.arange(len(neighs))
        
        self.particles['local_density'] = np.bincount(
            positions.repeat(lengths),
            w
        )

    ## https://stackoverflow.com/questions/45624653/calculate-average-of-groups-of-dataframe-rows-given-by-2-d-lists-of-indices-with
    def local_avg(self, data, cutoff):
        if data not in self.particles:
            print('{} not present, cannot calculate local average'.format(data))
            return
        if 'iloc_neighbors' not in self.particles:
            self.iloc_neighbors(cutoff=cutoff)
        neighs = self.particles['iloc_neighbors'].values.tolist()
        lengths = np.array([len(x) for x in neighs])
        positions = np.arange(len(neighs))
        values = self.particles[data].values
        self.particles['local_'+data] = np.bincount(
            positions.repeat(lengths),
            values[np.concatenate(neighs)]
        ) / lengths

    def temperature(self, mass=None):
        if 'vx' not in self.particles or 'vy' not in self.particles or 'vz' not in self.particles:
            print('velocities not present, cannot calculate temperature')
            return False
        if 'mass' not in self.particles and mass is None:
            print('masses not present, cannot calculate temperature')
            return False
        if mass:
            self.particles['mass'] = mass
        # conversion prefactor taken from lammps for units=metal
        mvv2e = 1.0364269e-4
        boltz = 8.617343e-5
        prefactor = mvv2e/(3*boltz)
        self.particles['temperature'] = prefactor * self.particles['mass'] * np.sum(self.particles[['vx', 'vy', 'vz']].values**2, axis=1)

    def distance(self, id1, id2):
        return np.linalg.norm(self.particles.loc[id1, ['x', 'y', 'z']].values - self.particles.loc[id2, ['x', 'y', 'z']].values, axis=0)
        
    def distances(self, id1, id2):
        return np.linalg.norm(self.particles.loc[id1, ['x', 'y', 'z']].values - self.particles.loc[id2, ['x', 'y', 'z']].values, axis=1)


class TrajectoryStats(object):
    def __init__(self, *traj_files):
        self.traj_files = traj_files
        self.frames = 0
        self.stats = pd.DataFrame(columns=['timestep', 'xlo', 'xhi', 'ylo', 'yhi', 'zlo', 'zhi'])
        for fname in self.traj_files:
            with open(fname) as f:
                header = [next(f) for _ in range(9)]
                timestep = int(header[1].strip())
                n_particles = int(header[3].strip())
                xlo, xhi = list(map(float, header[5].split()))
                ylo, yhi = list(map(float, header[6].split()))
                zlo, zhi = list(map(float, header[7].split()))
                data_args = header[8].split()[2:]
            particles = pd.read_csv(fname, sep='\s+', skiprows=9, nrows=n_particles, names=data_args, index_col='id')
            self.frames += 1
            avg_stats = {
                'frame': self.frames, 'timestep': timestep,
                'xlo': xlo, 'xhi': xhi,
                'ylo': ylo, 'yhi': yhi,
                'zlo': zlo, 'zhi': zhi,
                'n_particles': n_particles
            }
            for data in data_args[1:]:
                if 'avg_'+data not in self.stats:
                    self.stats['avg_'+data] = None
                    self.stats['std_'+data] = None
                avg_stats['avg_'+data] = particles[data].mean()
                avg_stats['std_'+data] = particles[data].std()
            self.last_frame = pd.Series(avg_stats)
            self.stats.loc[self.frames] = pd.Series(avg_stats)

# logic from lammps ProcMap::factor
def proc_array(nproc):
    facts = []
    for i in range(1, nproc+1):
        if nproc%i: continue
        nyz = nproc//i
        for j in range(1, nyz+1):
            if nyz%j: continue
            facts.append([nyz//j, j, i])
    max_val = list(map(lambda x: max(x), facts))
    min_id=max_val.index(min(max_val))
    return facts[min_id]
    
def parse_dump_header(fname):
    with open(fname) as f:
        next(f)
        t_step = int(next(f))
        next(f)
        n_particles = int(next(f))
        next(f)
        xlo, xhi = list(map(float, next(f).split()))
        ylo, yhi = list(map(float, next(f).split()))
        zlo, zhi = list(map(float, next(f).split()))
        data_args = next(f).split()[2:]
    return {'timestep': t_step, 'n_particles': n_particles, 'xlo': xlo, 'xhi': xhi, 'ylo': ylo, 'yhi': yhi, 'zlo': zlo, 'zhi': zhi, 'data_args': data_args}

def process_dump(fname, out_fname=None, unscale=False, scale=False, shrink_dims=[], dup_cols=[], chunksize=5000000):
    header_data = parse_dump_header(fname)
    d_dim = {'x': header_data['xhi']-header_data['xlo'], 'y': header_data['yhi']-header_data['ylo'], 'z': header_data['zhi']-header_data['zlo']}
    mins = maxs = None
    if shrink_dims:
        mins = {dim: 999999999 for dim in shrink_dims}
        maxs = {dim: -999999999 for dim in shrink_dims}
        reader = pd.read_csv(fname, sep='\s+', skiprows=9, names=header_data['data_args'], index_col='id', chunksize=chunksize)
        for chunk in reader:
            for dim in shrink_dims:
                if unscale:
                    mins[dim] = min(mins[dim], min(chunk[dim+'s']))
                    maxs[dim] = max(maxs[dim], max(chunk[dim+'s']))
                else:
                    mins[dim] = min(mins[dim], min(chunk[dim]))
                    maxs[dim] = max(maxs[dim], max(chunk[dim]))
        if unscale:
            for dim in shrink_dims:
                mins[dim] = header_data[dim+'lo'] + d_dim[dim]*mins[dim]
                maxs[dim] = header_data[dim+'lo'] + d_dim[dim]*maxs[dim]
    if out_fname is None:
        out_fname = fname+'.processed'
    data_args = [d for d in header_data['data_args']]
    for dup in dup_cols:
        new = dup[0]
        old = dup[1]
        data_args.append(new)
    if unscale:
        data_args = [d[0] if (d=='xs' or d=='ys' or d=='zs') else d for d in data_args]
    if scale:
        data_args = [d+'s' if (d=='x' or d=='y' or d=='z') else d for d in data_args]
    with open(fname) as fr:
        with open(out_fname, 'w') as fw:
            fw.write(next(fr))
            fw.write(next(fr))
            fw.write(next(fr))
            fw.write(next(fr))
            fw.write(next(fr))
            if 'x' in shrink_dims:
                fw.write('{} {}\n'.format(mins['x'], maxs['x']))
                next(fr)
            else:
                fw.write(next(fr))
            if 'y' in shrink_dims:
                fw.write('{} {}\n'.format(mins['y'], maxs['y']))
                next(fr)
            else:
                fw.write(next(fr))
            if 'z' in shrink_dims:
                fw.write('{} {}\n'.format(mins['z'], maxs['z']))
                next(fr)
            else:
                fw.write(next(fr))
            next(fr)
            fw.write('ITEM: ATOMS {}\n'.format(' '.join(data_args)))
            for line in fr:
                data = {}
                line_data = line.split()
                for arg in header_data['data_args']:
                    data[arg] = line_data.pop(0)
                if unscale:
                    data['x'] = header_data['xlo'] + float(data['xs'])*d_dim['x']
                    data['y'] = header_data['ylo'] + float(data['ys'])*d_dim['y']
                    data['z'] = header_data['zlo'] + float(data['zs'])*d_dim['z']
                if scale:
                    data['xs'] = (float(data['x']) - header_data['xlo']) / d_dim['x']
                    data['ys'] = (float(data['y']) - header_data['ylo']) / d_dim['y']
                    data['zs'] = (float(data['z']) - header_data['zlo']) / d_dim['z']
                for dup in dup_cols:
                    new, old = dup
                    data[new] = data[old]
                new_line = ''
                for arg in data_args:
                    new_line += '{} '.format(data[arg])
                new_line += '\n'
                fw.write(new_line)
                
            
