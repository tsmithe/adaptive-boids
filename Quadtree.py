# Java implementation of Quadtree: http://gamedevelopment.tutsplus.com/tutorials/quick-tip-use-quadtrees-to-detect-likely-collisions-in-2d-space--gamedev-374

class Boid: # Temporary boid class
    def __init__(self,x,y,radius):
        self.x = x
        self.y = y
        self.radius = radius
        self.boundary = Bounds(x-radius,y-radius,2*radius,2*radius)
        self.direction = 0

class Bounds:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

class Quadtree:
    maxobjs = 10;
    maxlevels = 5;
    
    def __init__(self, level, bounds):
        self.level = level
        self.bounds = bounds
        self.objs = []
        self.nodes = [None, None, None, None]

    def clear(self):
        self.objs = []
        if not self.nodes[0] is None:
            for i in range(1,4):
                self.nodes[i].clear()
                self.nodes[i] = None
        
    def split(self):
        subwidth = self.bounds.width/2
        subheight = self.bounds.height/2
        x = self.bounds.x
        y = self.bounds.y
        self.nodes[0] = Quadtree(self.level+1, Bounds(x, y, subwidth, subheight))
        self.nodes[1] = Quadtree(self.level+1, Bounds(x+subwidth, y, subwidth, subheight))
        self.nodes[2] = Quadtree(self.level+1, Bounds(x, y+subheight, subwidth, subheight))
        self.nodes[3] = Quadtree(self.level+1, Bounds(x+subwidth, y+subheight, subwidth, subheight))
        
    def getindex(self, obj):
        index = -1
        subwidth = self.bounds.width/2
        subheight = self.bounds.height/2
        x = self.bounds.x
        y = self.bounds.y
        horizontalmidpoint = x+subwidth
        verticalmidpoint = y+subheight
        
        if obj.y + obj.radius < verticalmidpoint:
            topquadrant = True
        else:
            topquadrant = False
        
        if obj.y - obj.radius > verticalmidpoint:
            bottomquadrant = True
        else:
            bottomquadrant = False
            
        if obj.x + obj.radius < horizontalmidpoint:
            if topquadrant:
                index = 1
            elif bottomquadrant:
                index = 2
        elif obj.x - obj.radius > horizontalmidpoint:
            if topquadrant:
                index = 0
            elif bottomquadrant:
                index = 3
                
        return index
    
    def insert(self,obj):
        if not self.nodes[0] is None:
            index = self.getindex(obj)
            
            if index != -1:
                self.nodes[index].insert(obj)
                return
        
        self.objs.append(obj)
        
        if len(self.objs) > self.maxobjs and self.level < self.maxlevels:
            if self.nodes[0] is None:
                self.split()
            
            for obj in self.objs:
                index = self.getindex(obj)
                if index != -1:
                    self.nodes[index].insert(obj)
                    self.objs.remove(obj)
        
    def retrieve(self, returned_objs, obj):
        index = self.getindex(obj)
        if index != -1 and not self.nodes[0] is None:
            self.nodes[index].retrieve(returned_objs, obj)
            
        returned_objs.extend(self.objs)
            
        return returned_objs
        
    def inspect(self):
        print('Level:')
        print(self.level)
        print('Bounds:')
        print([self.bounds.x, self.bounds.y, self.bounds.width, self.bounds.height])
        print('Number of boids:')
        print(len(self.objs))
        if not self.nodes[0] is None:
            self.nodes[0].inspect()
            self.nodes[1].inspect()
            self.nodes[2].inspect()
            self.nodes[3].inspect()
        
        