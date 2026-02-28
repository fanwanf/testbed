import torch
import numpy as np


def shot_item_isaacgym(mesh,ray_origins_ini,ray_directions_ini,xRange,yRange,device,start=[0,0,0]):

    ray_origins = ray_origins_ini[start[0]:start[0]+xRange,start[1]:start[1]+yRange].copy().reshape((-1,3))
    ray_directions = ray_directions_ini[start[0]:start[0]+xRange,start[1]:start[1]+yRange].copy().reshape((-1,3))

    heightMapB = torch.zeros(xRange * yRange,device=device,dtype=torch.float32)
    heightMapH = torch.zeros(xRange * yRange,device=device,dtype=torch.float32)
    maskB = torch.zeros(xRange * yRange,device=device,dtype=torch.int32)
    maskH = torch.zeros(xRange * yRange,device=device,dtype=torch.int32)

    # Store the bottom height map and mask for each object at each rotation angle
    # Key fix: use mesh.bounds[0] instead of bounding_box.vertices[0]
    # bounds[0] is guaranteed to be [min_x, min_y, min_z]
    mesh = mesh.copy()
    mesh.apply_translation(-mesh.bounds[0])
    index_triB, index_rayB, locationsB = mesh.ray.intersects_id(ray_origins=ray_origins, ray_directions=ray_directions,
                                                            return_locations=True,   multiple_hits=False)
    if len(index_rayB) != 0:
        heightMapB[index_rayB] = torch.tensor(locationsB[:, 2], device=device, dtype=torch.float32)
        maskB[index_rayB] = 1
    else:
        heightMapB[:] = 0
        maskB[:] = 1
    
    heightMapB = heightMapB.view(xRange, yRange)
    maskB = maskB.view(xRange, yRange)

    ray_origins[:, 2] *= -1
    ray_directions[:, 2] *= -1
    index_triH, index_rayH, locationsH = mesh.ray.intersects_id( ray_origins=ray_origins, ray_directions=ray_directions,
                                                                 return_locations=True,   multiple_hits=False)
    if len(index_rayH) != 0:
        heightMapH[index_rayH] = torch.tensor(locationsH[:, 2], device=device, dtype=torch.float32)
        maskH[index_rayH] = 1
    else:
        heightMapH[:] = mesh.extents[2]
        maskH[:] = 1

    heightMapH = heightMapH.view(xRange, yRange)
    maskH = maskH.view(xRange, yRange)

    return heightMapH, heightMapB, maskH, maskB

