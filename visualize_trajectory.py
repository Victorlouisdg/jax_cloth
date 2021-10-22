import bpy

def show_trajectory(armature_name):
    bpy.context.scene.frame_set(1)

    n_frames = 96

    vertices = []
    edges = [(i, i+1) for i in range(n_frames - 1)]
    faces = []

    armature = bpy.data.objects[armature_name]

    for i in range(n_frames):
        bpy.context.scene.frame_set(bpy.context.scene.frame_current + 1)
        head = armature.pose.bones["Bone"].head
        vertices.append(armature.location + head.copy())
        
    mesh = bpy.data.meshes.new('Trajectory')
    mesh.from_pydata(vertices, edges, faces)
    mesh.update()
    object = bpy.data.objects.new('Trajectory', mesh)
    collection = bpy.data.collections['Collection']
    collection.objects.link(object)


armatures = ['Armature', 'Armature.001', 'Armature.002', 'Armature.003']

for a in armatures:
    show_trajectory(a)

bpy.context.scene.frame_set(1)