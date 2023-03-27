import os

import torch
import torch.nn.functional as F

def collision_plane_force(q, ratio_collide, plane_height):
    '''collision force when the shape is collide with a plane'''
    collide_force = torch.zeros_like(q)
    collide_indices = (q[:, -1] < plane_height)
    q_collide = q[collide_indices, :]
    if q_collide.shape[0] > 0:
        collide_dist = plane_height - q_collide[:, -1]
        collide_force[collide_indices, :] = ratio_collide * torch.column_stack((torch.zeros(collide_dist.shape[0], q.shape[1]-1).cuda(), collide_dist))
    return collide_force


def collision_circle_force(q, ratio_collide, circle_center, circle_radius):
	'''collision force when the shape is collide with a sphere'''
	collide_force = torch.zeros_like(q)
	collide_vec = q - circle_center
	collide_dist = torch.sqrt(torch.sum(collide_vec ** 2, dim = 1))
	collide_dir = collide_vec / collide_dist[:, None]
	collide_indices = (collide_dist < circle_radius)
	q_collide = q[collide_indices, :]
	if q_collide.shape[0] > 0:
		collide_dist = collide_dist[collide_indices]
		collide_dir = collide_dir[collide_indices, :]
		if q.shape[1] == 2:
			collide_force[collide_indices, :] = ratio_collide * collide_dist[:, None] * collide_dir
		else:
			collide_force[collide_indices, :] = ratio_collide * collide_dist[:, None, None] * collide_dir
	return collide_force