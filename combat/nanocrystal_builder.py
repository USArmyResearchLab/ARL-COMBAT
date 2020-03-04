# -------------------------------------------------------------------------
#    The ARL Computational Model Builder and Analysis Toolkit (ARL COMBAT)
#    https://github.com/USArmyResearchLab/AMMPMSS
#    US Army CCDC Army Research Laboratory
#
#    This software is distributed under the Creative Commons Zero 1.0 
#    Universal (CC0 1.0) Public Domain Dedication.
#
#    See the README file in the top-level ARL COMBAT directory.
# ------------------------------------------------------------------------- 

# ------------------------------------------------------------------------- 
#    Contributing Authors:  Daniel Foley and Shawn Coleman
# ------------------------------------------------------------------------- 

import os 
import sys 
import numpy as np 
from scipy import spatial as sp 
import random
import linecache
import math as mt

try:
    input = raw_input
except NameError:
    pass

reference_mode = input('Do you want to generate from a single reference or a configuration file ' \
'(respond single or config):\n')
reference_mode = 'single'
while reference_mode != 'single' and reference_mode != 'config':
    print("I could not interpret your choice, please check your spelling and try again.")
    reference_mode = input('single or config? ')
if reference_mode=='single':
    crref = input('Input the reference file path:\n')
    avg_grain_size = float(input('Input desired grain size in angstroms:\n'))
    box_bounds = input('Input the box dimensions in angstroms\nUse the following convention lowerx, upperx, lowery, uppery, lowerz, upperz:\n')
    direction = input('Input crystalline texturing direction\nReply "none" (quotations excluded) if no texturing desired\nUse following convention index_x, index_y, index_z:\n')
    if direction != 'none':
        direction = np.fromstring(direction,sep=',') 
        direcnorm = np.linalg.norm(direction) 
        direction = direction/direcnorm
elif reference_mode=='config': 
    ref_list=input('Input path of configuration file:\n')
    rescale_factor = float(input('Input a scaling factor (1 if no change desired, must be greater than 0):\n'))
    contwin = float(input('Percent of grains to be reassigned (decimal from 0-1):\n'))
    if contwin != 0.:
        twinr = input('Input path of new reference file:\n')
    else:
        twinr = 'foo'
    direction = input('Input crystalline texturing direction\nReply "none" (quotations ' \
        'excluded) if no texturing desired\nUse following convention ' \
        'index_x,index_y,index_z:\n')
    if direction != 'none':
        direction = np.fromstring(direction,sep=',')
        direcnorm = np.linalg.norm(direction)
        direction = direction/direcnorm

output_name = input('Input desired output basename:\n') 

#----------Define Functions---------- 
def efficient_packing(center,grain_size): 
    #defines a point space in which the next grain center may be placed 
    rot = np.linspace(0.,360*np.pi/180.,45).reshape(45,1) 
    r = np.linspace(.75*grain_size,1.25*grain_size,20).reshape(20,1) 
    zero = np.zeros((20,2)) 
    r = np.concatenate((r,zero),axis = 1) 
    point_space = np.empty([20*45,3])
    n = 0
    for j in range(0,20): #this can probably be vectorized... work in progress
        for i in range(0,45):
            R_y = np.array([[np.cos(rot[i,0]),0,np.sin(rot[i,0])],[0,1,0], \
                [-np.sin(rot[i,0]),0,np.cos(rot[i,0])]])
            point_space[n,0:3] = np.dot(r[j,0:3],R_y)
            n = n + 1
    point_space_2 = np.empty([20*45*45,3])
    n = 0
    for j in range(0,len(point_space)): #this can probably be vectorized... work in progress
        for i in range(0,45):
            R_z = np.array([[np.cos(rot[i,0]),-np.sin(rot[i,0]),0],[np.sin(rot[i,0]), \
                np.cos(rot[i,0]),0],[0,0,1]])
            point_space_2[n,0:3] = np.dot(point_space[j,0:3],R_z)
            n = n+1
    point_space_2 = point_space_2 + center
    return point_space_2
    
def ghost_centers(centroids,box_limits): 
    #defines grain centers as seen across periodic bounds as "ghosts" 
    ghost_x = np.array([-(box_limits[0,1]-box_limits[0,0]),0,(box_limits[0,1]-box_limits[0,0])]) 
    ghost_y = np.array([-(box_limits[1,1]-box_limits[1,0]),0,(box_limits[1,1]-box_limits[1,0])]) 
    ghost_z = np.array([-(box_limits[2,1]-box_limits[2,0]),0,(box_limits[2,1]-box_limits[2,0])]) 
    ghost = np.empty([len(ghost_x)*len(ghost_y)*len(ghost_z),3]) 
    roll = 0 
    for i in range(0,len(ghost_x)): #this can probably be vectorized... work in progress
        for j in range(0,len(ghost_y)):
            for k in range(0,len(ghost_z)):
                ghost[roll,0] = ghost_x[i]
                ghost[roll,1] = ghost_y[j]
                ghost[roll,2] = ghost_z[k]
                roll = roll + 1
    if len(np.shape(centroids)) == 1:
        grain_centers_1 = centroids.reshape(1,3)
        grain_centers_2 = centroids.reshape(1,3)
    else:
        grain_centers_1 = centroids
        grain_centers_2 = centroids
    for i in range(0,len(ghost)):
        if ghost[i,0] == 0 and ghost[i,1] == 0 and ghost[i,2] == 0:
            continue 
        else:
            hold_center = np.add(grain_centers_1[:,0:3], ghost[i,:]) 
            grain_centers_2 = np.concatenate((grain_centers_2,hold_center),axis=0) 
    return grain_centers_2
    
def grain_centroid(amorphous_edge,grain_size): 
    #assigns grain centers within an allowed space defined by the positions of existing grain centers 
    box_vol = (amorphous_edge[0,1]-amorphous_edge[0,0])*(amorphous_edge[1,1]-amorphous_edge[1,0])*(amorphous_edge[2,1]-amorphous_edge[2,0]) 
    grain_volume = ((4./3.)*(grain_size / 2)**3)*np.pi 
    N = int(box_vol / grain_volume)
    print('approximate number of grains = %i' % (N))
    centroids = np.array([[(grain_size / 2.),(grain_size / 2.),(grain_size / 2.)]])
    allowed = efficient_packing(centroids[0,0:3],grain_size)
    allowed = transpose_periodic_bounds(allowed,amorphous_edge)
    ghost = ghost_centers(centroids[0,:],amorphous_edge)
    for i in range(1,N):
        print('asigning grain %i' % (i))
        if len(allowed) == 0:
            print('out of space')
            break
        choose = random.randint(0,len(allowed)-1)
        centroids = np.concatenate((centroids,allowed[choose,0:3].reshape(1,3)))
        allowed = np.concatenate((allowed,efficient_packing(centroids[i,0:3],grain_size)))
        centtree = sp.cKDTree(ghost_centers(centroids[0:i+1,:],amorphous_edge))
        dist, index = centtree.query(allowed) 
        deny = np.where(dist < .75*grain_size,False,True) 
        if np.any(deny) == 'False': 
            print('out of space, %i grains assigned' % (i))
            break 
        allowed = allowed[deny,:] 
        allowed = transpose_periodic_bounds(allowed,amorphous_edge) 
    return centroids
    
def seed_rotations(refdat,grain_center,grabradius,grabedges,axis,angle): 
    #function extracts and rotates crystalline seeds (updated in version 2)
    holdx1 = int(grabedges[0,0]+grabradius)
    holdx2 = int(grabedges[0,1]-grabradius)
    if holdx1 == holdx2:
        holdx1 = holdx1 - 2
        holdx2 = holdx2 + 2
    if holdx1 > holdx2:
        holdx1 = int(grabedges[0,1]-grabradius)
        holdx2 = int(grabedges[0,0]+grabradius)
    holdy1 = int(grabedges[1,0]+grabradius)
    holdy2 = int(grabedges[1,1]-grabradius)
    if holdy1 == holdy2:
        holdy1 = holdy1 - 2
        holdy2 = holdy2 + 2
    if holdy1 > holdy2:
        holdy1 = int(grabedges[1,1]-grabradius) 
        holdy2 = int(grabedges[1,0]+grabradius)
    holdz1 = int(grabedges[2,0]+grabradius) 
    holdz2 = int(grabedges[2,1]-grabradius) 
    if holdz1 == holdz2: 
        holdz1 = holdz1 - 2 
        holdz2 = holdz2 + 2 
    if holdz1 > holdz2: 
        holdz1 = int(grabedges[2,1]-grabradius) 
        holdz2 = int(grabedges[2,0]+grabradius) 
    x_c = float(random.randint(holdx1,holdx2)) 
    y_c = float(random.randint(holdy1,holdy2))
    z_c = float(random.randint(holdz1,holdz2))
    dtree = sp.cKDTree(refdat[:,2:5])
    neighpoint = dtree.query_ball_point([x_c,y_c,z_c], grabradius)
    datsphere = refdat[np.asarray(neighpoint),2:5]
    datsphere1 = datsphere[:,0:3]-[x_c,y_c,z_c]
    R = np.array([[np.cos(angle)+(axis[0]**2)*(1-np.cos(angle)),\
        axis[0]*axis[1]*(1-np.cos(angle))-axis[2]*np.sin(angle),\
        axis[0]*axis[2]*(1-np.cos(angle))+axis[1]*np.sin(angle)],\
        [axis[1]*axis[0]*(1-np.cos(angle))+axis[2]*np.sin(angle),\
        np.cos(angle)+(axis[1]**2)*(1-np.cos(angle)), \
        axis[1]*axis[2]*(1-np.cos(angle))-axis[0]*np.sin(angle)],\
        [axis[2]*axis[0]*(1-np.cos(angle))-axis[1]*np.sin(angle),\
        axis[2]*axis[1]*(1-np.cos(angle))+axis[0]*np.sin(angle),\
        np.cos(angle)+(axis[2]**2)*(1-np.cos(angle))]])
    datsphererot = np.dot(datsphere1, R) 
    datspherefinal = datsphererot + grain_center 
    return datsphere, datspherefinal
    
def check_ownership(grain_centers,atoms,box_limits): 
    #checks to make sure an atom's nearest center is its parent center, deletes atom if not 
    grain_centers = ghost_centers(grain_centers,box_limits) 
    centertree = sp.cKDTree(grain_centers) 
    dist, label = centertree.query(atoms[:,0:3]) 
    check = np.equal(atoms[:,3],label) 
    output_atoms = atoms[check,0:4] 
    return output_atoms, grain_centers
    
def transpose_periodic_bounds(atoms,box_lims):
    #reflects atoms which exist outside the box bounds through the periodic conditions
    lower = np.empty([len(atoms),3])
    lower[:,0:3] = [box_lims[0,0],box_lims[1,0],box_lims[2,0]]
    upper = np.empty([len(atoms),3])
    upper[:,0:3] = [box_lims[0,1],box_lims[1,1],box_lims[2,1]]
    test_l = np.less_equal(atoms[:,0:3],lower[:,0:3])
    test_u = np.greater_equal(atoms[:,0:3],upper[:,0:3])
    shift_l = np.where(test_l,1,0)
    shift_u = np.where(test_u,-1,0)
    shift_L = np.multiply(shift_l[:,0:3],np.array([box_lims[0,1]-box_lims[0,0],\
        box_lims[1,1]-box_lims[1,0],\
        box_lims[2,1]-box_lims[2,0]]))
    shift_U = np.multiply(shift_u[:,0:3],np.array([box_lims[0,1]-box_lims[0,0],box_lims[1,1]-box_lims[1,0],box_lims[2,1]-box_lims[2,0]])) 
    out_atoms = atoms 
    out_atoms[:,0:3] = np.add(atoms[:,0:3],shift_L) 
    out_atoms[:,0:3] = np.add(atoms[:,0:3],shift_U) 
    return out_atoms
    
def reference_rescale(ref_atoms,box_lims,grain_size): 
    #function resizes the reference structure to accomodate requested grain size 
    rescale_length = np.array([(grain_size - (box_lims[0,1]-box_lims[0,0])),(grain_size - (box_lims[1,1]-box_lims[1,0])),(grain_size - (box_lims[2,1]-box_lims[2,0]))])
    if rescale_length[0] > 0.0 \
            or rescale_length[1] > 0.0\
            or rescale_length[2] > 0.0:
        n_x = int(rescale_length[0]/(box_lims[0,1]-box_lims[0,0]))+2
        n_y = int(rescale_length[1]/(box_lims[1,1]-box_lims[1,0]))+2
        n_z = int(rescale_length[2]/(box_lims[2,1]-box_lims[2,0]))+2
        new_ref = ref_atoms
        for i in range(0,n_x+1):
            for j in range(0,n_y+1):
                for k in range(0,n_z+1):
                    if i == 0 and k == 0 and j == 0:
                        continue
                    else:
                        new_ats = ref_atoms + [0,0,float(i)*(box_lims[0,1]-box_lims[0,0]),float(j)*(box_lims[1,1]-box_lims[1,0]),float(k)*(box_lims[2,1]-box_lims[2,0])] 
                        new_ref = np.concatenate((new_ref,new_ats)) 
    else:
        replicate_range = np.array([[box_lims[0,0],box_lims[0,0]+2*rescale_length[0]],[box_lims[1,0],box_lims[1,0]+2*rescale_length[1]],[box_lims[2,0],box_lims[2,0]+2*rescale_length[2]]]) 
        check_x=np.where(ref_atoms[:,2] <= replicate_range[0,1],True,False) 
        rep_atoms_x = ref_atoms[check_x[:],:] 
        rep_atoms_x[:,2]=rep_atoms_x[:,2] + (box_lims[0,1]-box_lims[0,0]) 
        rep_atoms = np.concatenate((ref_atoms,rep_atoms_x),axis=0)
        check_y=np.where(rep_atoms[:,3] <= replicate_range[1,1],True,False)
        rep_atoms_y = rep_atoms[check_y[:],:]
        rep_atoms_y[:,3]=rep_atoms_y[:,3] + (box_lims[1,1]-box_lims[1,0])
        rep_atoms = np.concatenate((rep_atoms,rep_atoms_y),axis=0)
        check_z=np.where(rep_atoms[:,4] <= replicate_range[2,1],True,False)
        rep_atoms_z = rep_atoms[check_z[:],:]
        rep_atoms_z[:,4]=rep_atoms_z[:,4] + (box_lims[2,1]-box_lims[2,0])
        rep_atoms = np.concatenate((rep_atoms,rep_atoms_z),axis=0)
        new_ref = rep_atoms
    return new_ref
def multireference(ref_list,grain_size,alimits,rescale,percent_twin,twin_reference,axis,angle):
    #function will extract centroid and box data from config file and build a new structure accordingly
    c_refs = np.loadtxt('%s' % (ref_list),skiprows=2,dtype={'names':('id','c_x','c_y','c_z','axisr_x','axisr_y','axisr_z','angler','file'),'formats':('i4','f4','f4','f4','f4','f4','f4','f4','S32')}) 
    n_grains = float(len(c_refs)) 
    n_twinned = n_grains * percent_twin 
    check = np.empty((int(n_twinned),1),dtype=int) 
    print('reassigning %i reference files' % (n_twinned))
    for i in range(0,int(n_twinned)): 
        itl = random.randint(0,len(c_refs)-1) 
        while np.any(check[:]==itl)==True: 
            itl = random.randint(0,len(c_refs)-1) 
        c_refs['file'][itl] = '%s' % (twin_reference) 
        check[i,0] = itl
    center = np.empty((len(c_refs),3))
    f=open('%s_centroids.config' % (output_name),'w')
    f.write('#data for centroids\n')
    f.write('%f %f %f %f %f %f %f\n' % (grain_size*rescale, alimits[0,0], alimits[0,1]*rescale,\
        alimits[1,0], alimits[1,1]*rescale, alimits[2,0], alimits[2,1]*rescale))
    finaldat = np.empty([0,4])
    print('populating atoms')
    for i in range(0,len(c_refs)):
        print('populating grain %i' % (i))
        limsx = np.fromstring(linecache.getline('%s' % (c_refs['file'][i]),6),sep=' ')
        limsy = np.fromstring(linecache.getline('%s' % (c_refs['file'][i]),7),sep=' ')
        limsz = np.fromstring(linecache.getline('%s' % (c_refs['file'][i]),8),sep=' ')
        limits = np.concatenate((limsx,limsy,limsz),axis=0).reshape(3,2)
        cdat=np.loadtxt('%s' % c_refs['file'][i],skiprows=9,usecols=(0,1,2,3,4))
        if 1.5*grain_size*rescale >= (limits[0,1]-limits[0,0]) or 1.5*grain_size*rescale >= (limits[1,1]-limits[1,0]) or 1.5*grain_size*rescale >= (limits[2,1]-limits[2,0]): 
            cdat = reference_rescale(cdat,limits,1.5*grain_size*rescale) 
            limits = np.array([[np.amin(cdat[:,2]),np.amax(cdat[:,2])],[np.amin(cdat[:,3]),np.amax(cdat[:,3])],[np.amin(cdat[:,4]),np.amax(cdat[:,4])]]) 
        if axis != 'none': 
            c_refs['axisr_x'] = axis[0] 
            c_refs['axisr_y'] = axis[1] 
            c_refs['axisr_z'] = axis[2] 
            haxis = axis 
            f.write('%i %f %f %f %f %f %f %f %s\n' % (i, c_refs['c_x'][i]*rescale, \
                c_refs['c_y'][i]*rescale, c_refs['c_z'][i]*rescale, c_refs['axisr_x'][i],\
                c_refs['axisr_y'][i], c_refs['axisr_z'][i], c_refs['angler'][i], \
                c_refs['file'][i]))
            center[i,:]=[c_refs['c_x'][i]*rescale,c_refs['c_y'][i]*rescale,c_refs['c_z'][i]*rescale]
            [trydat, seeddat]=seed_rotations(cdat,[c_refs['c_x'][i]*rescale,\
                c_refs['c_y'][i]*rescale,\
                c_refs['c_z'][i]*rescale],\
                1.5*rescale*grain_size/2,limits,haxis,\
                c_refs['angler'][i])
        else:
            f.write('%i %f %f %f %f %f %f %f %s\n' % (i, c_refs['c_x'][i]*rescale, \
                c_refs['c_y'][i]*rescale, \
                c_refs['c_z'][i]*rescale, \
                c_refs['axisr_x'][i], \
                c_refs['axisr_y'][i], \
                c_refs['axisr_z'][i], \
                c_refs['angler'][i], \
                c_refs['file'][i])) 
            center[i,:]=[c_refs['c_x'][i]*rescale,c_refs['c_y'][i]*rescale,c_refs['c_z'][i]*rescale] 
            haxis = np.array([c_refs['axisr_x'][i], c_refs['axisr_y'][i], c_refs['axisr_z'][i]]) 
            [trydat, seeddat]=seed_rotations(cdat,[c_refs['c_x'][i]*rescale,\
                c_refs['c_y'][i]*rescale,\
                c_refs['c_z'][i]*rescale],\
                1.5*rescale*grain_size/2,limits,haxis,\
                c_refs['angler'][i])
        seeddat = np.insert(seeddat,3,i,axis=1)
        finaldat = np.concatenate((finaldat,seeddat))
        linecache.clearcache()
    f.close()
    return finaldat, center
    
#----------Main Body---------- 
if reference_mode == 'single':
    cdat = np.loadtxt('%s' % (crref),skiprows=9,usecols=(0,1,2,3,4))
    limsx = np.fromstring(linecache.getline('%s' % (crref),6),sep=' ')
    limsy = np.fromstring(linecache.getline('%s' % (crref),7),sep=' ')
    limsz = np.fromstring(linecache.getline('%s' % (crref),8),sep=' ')
    lims = np.concatenate((limsx,limsy,limsz),axis=0).reshape(3,2)
    # print lims 
    # Check if the average grain size is larger than the box bounds 
    if 1.5*avg_grain_size >= (lims[0,1]-lims[0,0]) or 1.5*avg_grain_size >= (lims[1,1]-lims[1,0]) \
            or 1.5*avg_grain_size >= (lims[2,1]-lims[2,0]): 
        cdat = reference_rescale(cdat,lims,1.5*avg_grain_size) 
        lims = np.array([[np.amin(cdat[:,2]),np.amax(cdat[:,2])],\
            [np.amin(cdat[:,3]),np.amax(cdat[:,3])],\
            [np.amin(cdat[:,4]),np.amax(cdat[:,4])]]) 
    lims2 = np.fromstring(box_bounds,sep=',').reshape(3,2) 
    if avg_grain_size >= (lims2[0,1]-lims2[0,0]) or avg_grain_size >= (lims2[1,1]-lims2[1,0]) \
            or avg_grain_size >= (lims2[2,1]-lims2[2,0]): 
        sys.exit("grain size larger than defined simulation box, change box sizes")
    print('assigning grain centers')
    center = grain_centroid(lims2,avg_grain_size)
    finaldat = np.empty([0,4])
    a = np.ones((len(cdat),1)).reshape(len(cdat),1)
    b = np.arange(1,len(cdat)+1).reshape(len(cdat),1)
    cdat = np.concatenate((b,a,cdat[:,2:5]),axis=1)
    centout=np.empty((len(center),8))
    f=open('%s_centroids.config' % (output_name),'w')
    f.write('#data for centroids\n')
    f.write('%f %f %f %f %f %f %f\n' % (avg_grain_size,lims2[0,0],lims2[0,1],lims2[1,0],\
        lims2[1,1],lims2[2,0],lims2[2,1]))
    print('populating atoms')
    for i in range(0,len(center)):
        print('populating grain %i' % (i))
        if direction == 'none': 
            axis = np.array([random.uniform(0,10),random.uniform(0,10),random.uniform(0,10)]) 
            axnorm = np.linalg.norm(axis) 
            axis = axis/axnorm 
            angle = mt.radians(random.uniform(0,180)) 
        else:
            axis = direction 
            angle = random.uniform(0,180) 
        f.write('%i %f %f %f %f %f %f %f %s\n' % (i,center[i,0],center[i,1],center[i,2],axis[0],\
            axis[1],axis[2],angle,crref)) 
        [trydat, seeddat] = seed_rotations(cdat,center[i,0:3],1.5*avg_grain_size/2,lims,axis,angle) 
        seeddat = np.insert(seeddat,3,i,axis = 1)
        finaldat = np.concatenate((finaldat,seeddat))
    f.close()
    print('finaldat=', len(finaldat))
    print('trimming excess atoms')
    cut_atoms,check_centers = check_ownership(center,finaldat[:,0:4],lims2)
    print('cut_atoms=', len(cut_atoms))
    print('wrapping periodic bounds')
    #fin_atoms = transpose_periodic_bounds(cut_atoms,lims2)
    fin_atoms = cut_atoms
    idcol = np.ones([len(cut_atoms),1])
    fin_atoms = np.concatenate((idcol,cut_atoms), axis = 1)
    numcol = np.arange(1,len(fin_atoms)+1).reshape(len(fin_atoms),1)
    fin_atoms = np.concatenate((numcol,fin_atoms), axis = 1)
    output_atoms = np.column_stack((fin_atoms[:,0],fin_atoms[:,5]+1,fin_atoms[:,2],\
        fin_atoms[:,3],fin_atoms[:,4]))
    print('outputting to data file' )
    head = ("#data file for lammps\n%i atoms\n%s atom types\n%f %f xlo xhi\n%f %f ylo "\
        "yhi\n%f %f zlo zhi\n\nAtoms\n") % (len(output_atoms),len(center)+1,lims2[0,0],\
        lims2[0,1],lims2[1,0],lims2[1,1],lims2[2,0],lims2[2,1]) 
    np.savetxt('data.%s' % (output_name),output_atoms, fmt='%i %i %f %f %f') 
    f=open('data.%s' % (output_name),'r') 
    data1 = f.read() 
    f.close() 
    
    g=open('data.%s' % (output_name),'w')
    g.write('%s\n' % (head))
    g.write('%s' % (data1))
    g.close()
else:
    print('reading configuration data')
    conf = np.fromstring(linecache.getline('%s' % (ref_list),2),sep=' ')
    avg_grain_size = conf[0]
    lims = conf[1:].reshape(3,2)
    lims2 = lims * rescale_factor
    axis = direction
    angle = mt.radians(random.uniform(0,180))
    [finaldat, center] = multireference(ref_list,avg_grain_size,lims,rescale_factor,contwin,\
        twinr,axis,angle)
    if avg_grain_size*rescale_factor >= (lims2[0,1]-lims2[0,0]) \
            or avg_grain_size*rescale_factor >= (lims2[1,1]-lims2[1,0]) \
            or avg_grain_size*rescale_factor >= (lims2[2,1]-lims2[2,0]): 
        sys.exit("grain size larger than defined simulation box, change box sizes") 
    a = np.ones((len(finaldat),1)).reshape(len(finaldat),1) 
    b = np.arange(1,len(finaldat)+1).reshape(len(finaldat),1) 
    finaldat = np.concatenate((b,a,finaldat),axis=1) 
    
    print('trimming excess atoms' )
    cut_atoms,check_centers = check_ownership(center,finaldat[:,2:6],lims2) 
    a = np.ones((len(cut_atoms),1)).reshape(len(cut_atoms),1) 
    b = np.arange(1,len(cut_atoms)+1).reshape(len(cut_atoms),1)
    cut_atoms = np.concatenate((b,a,cut_atoms),axis=1)
    print('wrapping periodic bounds')
    fin_atoms = transpose_periodic_bounds(cut_atoms[:,2:5],lims2)
    idcol = np.ones([len(cut_atoms),1])
    fin_atoms = np.concatenate((idcol,cut_atoms), axis = 1)
    numcol = np.arange(1,len(fin_atoms)+1).reshape(len(fin_atoms),1)
    fin_atoms = np.concatenate((numcol,fin_atoms), axis = 1)
    output_atoms = np.column_stack((fin_atoms[:,0],fin_atoms[:,7]+1,fin_atoms[:,4],\
        fin_atoms[:,5],fin_atoms[:,6]))
    print('outputting to data file')
    head = ("#data file for lammps\n%i atoms\n%s atom types\n%f %f xlo xhi\n%f %f ylo "\
        "yhi\n%f %f zlo zhi\n\nAtoms\n") % (len(fin_atoms),len(center)+1,lims2[0,0],\
        lims2[0,1],lims2[1,0],lims2[1,1],lims2[2,0],lims2[2,1])
    np.savetxt('data.%s' % (output_name),output_atoms, fmt='%i %i %f %f %f')
    f=open('data.%s' % (output_name),'r')
    data1 = f.read()
    f.close()
    g=open('data.%s' % (output_name),'w')
    g.write('%s\n' % (head))
    g.write('%s' % (data1))
    g.close()
