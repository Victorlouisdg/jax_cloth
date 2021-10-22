import bpy
import numpy as np
import time
import datetime

import matplotlib.pyplot as plt

t0 = time.time()


action = bpy.data.actions['ArmatureAction']
ycurve = action.fcurves[1]
y1_key = ycurve.keyframe_points[1]

def keypoint_set(keypoint, v):
    keypoint.co[1] = v
    keypoint.handle_left[1] = v
    keypoint.handle_right[1] = v


def simulate(n_frames):
    bpy.context.object.modifiers["Cloth"].show_viewport = False
    bpy.context.object.modifiers["Cloth"].show_viewport = True
    
    scene = bpy.context.scene
    scene.frame_set(1)
    for i in range(n_frames):
        scene.frame_set(scene.frame_current + 1)


def get_positions(object_name):
    ob = bpy.data.objects[object_name]
    depsgraph = bpy.context.evaluated_depsgraph_get()
    ob_eval = ob.evaluated_get(depsgraph)
    mesh = ob_eval.data

    positions = [(ob.matrix_world @ v.co) for v in mesh.vertices]
    return np.array(positions)


n_frames = 96
positions_goal = get_positions('Goal')

def loss(height):
    keypoint_set(y1_key, height)
    simulate(n_frames)
    positions = get_positions('Cloth')
    return np.sum(np.square(positions_goal - positions))


x_step = 0.001
xs = np.arange(0.0, 0.5 + x_step, x_step)
losses = []

for x in xs:
    loss_value = loss(x)
    print(f'x={x}, loss(x)={loss_value}')
    losses.append(loss_value)
    
timestamp = datetime.datetime.now()

fig = plt.figure()
plt.title("Loss Landscape for Toy Task: 1D Sleeve Folding")
plt.xlabel("Trajectory Peak Height")
plt.ylabel("Loss MSE (deviation from goal)")
plt.plot(xs, losses)
plt.savefig(f"loss_landscape_{timestamp}.png")

t1 = time.time()
print(f"Execution took {t1 - t0} seconds.")