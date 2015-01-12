import bpy;
import math
from mathutils import *
import sys;

#

################################################################################
##                                Settings                                    ##
################################################################################
#path to the file of bounding boxes
f = open("/home/chris/Projects/repos/siggraph/boxes", "r")
    
# number of segments for each sphere (corners) and number of vertices 
# for the edges
#for debugging use
#dSegs=1
#regCount=1

dSegs=24
# number of rings for the spheres
rCount=8

s=0.005 #radius of the sphere
r=0.0025 #radius of the edges
smooth=False #for smooth look

################################################################################
##                                Settings                                    ##
################################################################################


if bpy.context.scene.objects.values():
    bpy.ops.object.mode_set(mode='OBJECT')
bpy.context.scene.cursor_location=(0,0,0)
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
for m in bpy.data.meshes:
    m.user_clear()
    bpy.data.meshes.remove(m)

def addSphere(loc=(0,0,0)):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=dSegs, ring_count=rCount, size=s, location=loc)
    
def addCylinder(loc1=(0,0,0), loc2=(0,0,0)):
    print(loc1)
    vec = Vector((loc2[0]-loc1[0],loc2[1]-loc1[1],loc2[2]-loc1[2]))
    pos = Vector((loc2[0]+loc1[0],loc2[1]+loc1[1],loc2[2]+loc1[2]))/2
    up = Vector((0,0,1))
    if vec != -up:
        rot=up.rotation_difference(vec)
    else:
        rot=Quaternion((1,0,0),pi)
    d = vec.length    
    bpy.ops.mesh.primitive_cylinder_add(vertices=dSegs, radius=r, depth=d, location=pos)
    bpy.ops.transform.rotate(value=rot.angle, axis=rot.axis)

def createBox(b):
    bpy.ops.object.select_all(action='DESELECT')
    selectedObjects = []
    v1 = (b.vertices[0].x, b.vertices[0].y, b.vertices[0].z)
    v2 = (b.vertices[1].x, b.vertices[1].y, b.vertices[1].z)
    v3 = (b.vertices[2].x, b.vertices[2].y, b.vertices[2].z)
    v4 = (b.vertices[3].x, b.vertices[3].y, b.vertices[3].z)
    v5 = (b.vertices[4].x, b.vertices[4].y, b.vertices[4].z)
    v6 = (b.vertices[5].x, b.vertices[5].y, b.vertices[5].z)
    v7 = (b.vertices[6].x, b.vertices[6].y, b.vertices[6].z)
    v8 = (b.vertices[7].x, b.vertices[7].y, b.vertices[7].z)

    addCylinder(loc1=v1, loc2=v2)    
    selectedObjects.append(bpy.context.active_object)
    addCylinder(loc1=v3, loc2=v4)
    selectedObjects.append(bpy.context.active_object)
    addCylinder(loc1=v5, loc2=v6)
    selectedObjects.append(bpy.context.active_object)
    addCylinder(loc1=v7, loc2=v8)
    selectedObjects.append(bpy.context.active_object)

    addCylinder(loc1=v1, loc2=v3)
    selectedObjects.append(bpy.context.active_object)
    addCylinder(loc1=v2, loc2=v4)
    selectedObjects.append(bpy.context.active_object)
    addCylinder(loc1=v5, loc2=v7)
    selectedObjects.append(bpy.context.active_object)
    addCylinder(loc1=v6, loc2=v8)
    selectedObjects.append(bpy.context.active_object)

    addCylinder(loc1=v1, loc2=v5)
    selectedObjects.append(bpy.context.active_object)
    addCylinder(loc1=v2, loc2=v6)
    selectedObjects.append(bpy.context.active_object)
    addCylinder(loc1=v3, loc2=v7)
    selectedObjects.append(bpy.context.active_object)
    addCylinder(loc1=v4, loc2=v8)
    selectedObjects.append(bpy.context.active_object)


    addSphere(loc=v1)
    selectedObjects.append(bpy.context.active_object)
    addSphere(loc=v2)
    selectedObjects.append(bpy.context.active_object)
    addSphere(loc=v3)
    selectedObjects.append(bpy.context.active_object)
    addSphere(loc=v4)
    selectedObjects.append(bpy.context.active_object)
    addSphere(loc=v5)
    selectedObjects.append(bpy.context.active_object)
    addSphere(loc=v6)
    selectedObjects.append(bpy.context.active_object)
    addSphere(loc=v7)
    selectedObjects.append(bpy.context.active_object)
    addSphere(loc=v8)
    selectedObjects.append(bpy.context.active_object)

    bpy.ops.object.select_all(action='DESELECT')
    for o in selectedObjects:
        o.select = True
    bpy.ops.object.join()

    bpy.ops.object.mode_set(mode='OBJECT')
    if smooth:
        bpy.ops.object.shade_smooth()
    
class Vertex:
    def __init__(self):
       self.x = 0.0;
       self.y = 0.0;
       self.z = 0.0;

    def printing(self):
        print("Vertex(%f, %f, %f)" % (self.x, self.y, self.z))

class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

class Box:
    def __init__(self):
        print(bcolors.OKGREEN + "add new box" + bcolors.ENDC)
        self.center = Vertex();
        self.vertices = [];
        self.boxName='';

    def addVertex(self, v):
        print(bcolors.OKBLUE + "add Vertex(%f, %f, %f)" % (v.x, v.y, v.z) + bcolors.ENDC)
        self.vertices.append(v);

    def compCenter(self):
        x = 0.0;
        y = 0.0;
        z = 0.0;
        for v in self.vertices:
            self.center.x += v.x;
            self.center.y += v.y;
            self.center.z += v.z;
        self.center.x = x/8.0;
        self.center.y = y/8.0;
        self.center.z = z/8.0;

boxes = []

for line in f:
    if "# box begin" in line:
        boxes.append(Box());
    elif "#" in line: continue;
    else:
        vals = line.rstrip().split(" ");
        v = Vertex();
        v.x = float(vals[0])
        v.y = float(vals[1])
        v.z = float(vals[2])
        boxes[-1].addVertex(v);
        boxes[-1].compCenter();


for b in boxes:
    createBox(b);
    