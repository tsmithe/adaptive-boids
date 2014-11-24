import Quadtree
import random
import matplotlib.pyplot as plt

width = 600
height = 600
boids=[]
for i in xrange(1,100):
    x = width*random.random()
    y = height*random.random()
    boids.append(Quadtree.Boid(x,y,1))

tree = Quadtree.Quadtree(0,Quadtree.Bounds(0,0,width,height))

for obj in boids:
    tree.insert(obj)
    
x=[]
y=[]
colors=[]
chosen = random.choice(boids)
closeby = tree.retrieve([],chosen)
for obj in boids:
    x.append(obj.x)
    y.append(obj.y)
    
    if obj == chosen:
        colors.append([0, 1, 0])
    elif obj in closeby:
        colors.append([1, 0, 0])
    else:
        colors.append([0, 0, 1])
    
plt.clf()
plt.scatter(x, y, c=colors)
plt.show()