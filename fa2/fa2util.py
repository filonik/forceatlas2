# This file allows separating the most CPU intensive routines from the
# main code.  This allows them to be optimized with Cython.  If you
# don't have Cython, this will run normally.  However, if you use
# Cython, you'll get speed boosts from 10-100x automatically.
#
# THE ONLY CATCH IS THAT IF YOU MODIFY THIS FILE, YOU MUST ALSO MODIFY
# fa2util.pxd TO REFLECT ANY CHANGES IN FUNCTION DEFINITIONS!
#
# Copyright (C) 2017 Bhargav Chippada <bhargavchippada19@gmail.com>
#
# Available under the GPLv3

from math import sqrt

import numpy as np


DEFAULT_DIM = 2
DEFAULT_DTYPE = np.double


# This will substitute for the nLayout object
class Node:
    def __init__(self, dim=DEFAULT_DIM, dtype=DEFAULT_DTYPE):
        self.mass = 0.0
        #self.old_dx = 0.0
        #self.old_dy = 0.0
        self.old_delta = np.zeros((dim,), dtype=dtype)
        #self.dx = 0.0
        #self.dy = 0.0
        self.delta = np.zeros((dim,), dtype=dtype)
        #self.x = 0.0
        #self.y = 0.0
        self.position = np.zeros((dim,), dtype=dtype)


# This is not in the original java code, but it makes it easier to deal with edges
class Edge:
    def __init__(self):
        self.node1 = -1
        self.node2 = -1
        self.weight = 0.0


# Here are some functions from ForceFactory.java
# =============================================

# Repulsion function.  `n1` and `n2` should be nodes.  This will
# adjust the dx and dy values of `n1`  `n2`
def linRepulsion(n1, n2, coefficient=0):
    #xDist = n1.x - n2.x
    #yDist = n1.y - n2.y
    diff = n1.position - n2.position
    #distance2 = xDist * xDist + yDist * yDist  # Distance squared
    distance2 = np.inner(diff, diff)
    
    if distance2 > 0:
        factor = coefficient * n1.mass * n2.mass / distance2
        #n1.dx += xDist * factor
        #n1.dy += yDist * factor
        n1.delta += diff * factor
        #n2.dx -= xDist * factor
        #n2.dy -= yDist * factor
        n2.delta -= diff * factor


# Repulsion function. 'n' is node and 'r' is region
def linRepulsion_region(n, r, coefficient=0):
    #xDist = n.x - r.massCenterX
    #yDist = n.y - r.massCenterY
    diff = n.position - r.massCenter
    #distance2 = xDist * xDist + yDist * yDist
    distance2 = np.inner(diff, diff)

    if distance2 > 0:
        factor = coefficient * n.mass * r.mass / distance2
        #n.dx += xDist * factor
        #n.dy += yDist * factor
        n.delta += diff * factor


# Gravity repulsion function.  For some reason, gravity was included
# within the linRepulsion function in the original gephi java code,
# which doesn't make any sense (considering a. gravity is unrelated to
# nodes repelling each other, and b. gravity is actually an
# attraction)
def linGravity(n, g):
    #xDist = n.x
    #yDist = n.y
    diff = n.position
    #distance = sqrt(xDist * xDist + yDist * yDist)
    distance = np.linalg.norm(diff)
    
    if distance > 0:
        factor = n.mass * g / distance
        #n.dx -= xDist * factor
        #n.dy -= yDist * factor
        n.delta -= diff * factor


# Strong gravity force function. `n` should be a node, and `g`
# should be a constant by which to apply the force.
def strongGravity(n, g, coefficient=0):
    #xDist = n.x
    #yDist = n.y
    diff = n.position

    #if xDist != 0 and yDist != 0:
    if np.all(diff != 0):
        factor = coefficient * n.mass * g
        #n.dx -= xDist * factor
        #n.dy -= yDist * factor
        n.delta -= diff * factor


# Attraction function.  `n1` and `n2` should be nodes.  This will
# adjust the dx and dy values of `n1` and `n2`.  It does
# not return anything.
def linAttraction(n1, n2, e, distributedAttraction, coefficient=0):
    #xDist = n1.x - n2.x
    #yDist = n1.y - n2.y
    diff = n1.position - n2.position
    if not distributedAttraction:
        factor = -coefficient * e
    else:
        factor = -coefficient * e / n1.mass
    #n1.dx += xDist * factor
    #n1.dy += yDist * factor
    n1.delta += diff * factor
    #n2.dx -= xDist * factor
    #n2.dy -= yDist * factor
    n2.delta -= diff * factor


# The following functions iterate through the nodes or edges and apply
# the forces directly to the node objects.  These iterations are here
# instead of the main file because Python is slow with loops.
def apply_repulsion(nodes, coefficient):
    i = 0
    for n1 in nodes:
        j = i
        for n2 in nodes:
            if j == 0:
                break
            linRepulsion(n1, n2, coefficient)
            j -= 1
        i += 1


def apply_gravity(nodes, gravity, useStrongGravity=False):
    if not useStrongGravity:
        for n in nodes:
            linGravity(n, gravity)
    else:
        for n in nodes:
            strongGravity(n, gravity)


def apply_attraction(nodes, edges, distributedAttraction, coefficient, edgeWeightInfluence):
    # Optimization, since usually edgeWeightInfluence is 0 or 1, and pow is slow
    if edgeWeightInfluence == 0:
        for edge in edges:
            linAttraction(nodes[edge.node1], nodes[edge.node2], 1, distributedAttraction, coefficient)
    elif edgeWeightInfluence == 1:
        for edge in edges:
            linAttraction(nodes[edge.node1], nodes[edge.node2], edge.weight, distributedAttraction, coefficient)
    else:
        for edge in edges:
            linAttraction(nodes[edge.node1], nodes[edge.node2], pow(edge.weight, edgeWeightInfluence),
                          distributedAttraction, coefficient)


# For Barnes Hut Optimization
class Region:
    def __init__(self, nodes, dim=DEFAULT_DIM, dtype=DEFAULT_DTYPE):
        self.mass = 0.0
        #self.massCenterX = 0.0
        #self.massCenterY = 0.0
        self.massCenter = np.zeros((dim,), dtype=dtype)
        self.size = 0.0
        self.nodes = nodes
        self.subregions = []
        self.updateMassAndGeometry()

    def updateMassAndGeometry(self):
        if len(self.nodes) > 1:
            self.mass = 0
            #massSumX = 0
            #massSumY = 0
            massSum = np.zeros_like(self.massCenter)
            for n in self.nodes:
                self.mass += n.mass
                #massSumX += n.x * n.mass
                #massSumY += n.y * n.mass
                massSum += n.position * n.mass
            #self.massCenterX = massSumX / self.mass
            #self.massCenterY = massSumY / self.mass
            self.massCenter = massSum / self.mass
            
            self.size = 0.0
            for n in self.nodes:
                #distance = sqrt((n.x - self.massCenterX) ** 2 + (n.y - self.massCenterY) ** 2)
                distance = np.linalg.norm(n.position - self.massCenter)
                self.size = max(self.size, 2 * distance)

    def buildSubRegions(self):
        if len(self.nodes) > 1:
            '''
            leftNodes = []
            rightNodes = []
            for n in self.nodes:
                if n.x < self.massCenterX:
                    leftNodes.append(n)
                else:
                    rightNodes.append(n)

            topleftNodes = []
            bottomleftNodes = []
            for n in leftNodes:
                if n.y < self.massCenterY:
                    topleftNodes.append(n)
                else:
                    bottomleftNodes.append(n)

            toprightNodes = []
            bottomrightNodes = []
            for n in rightNodes:
                if n.y < self.massCenterY:
                    toprightNodes.append(n)
                else:
                    bottomrightNodes.append(n)

            if len(topleftNodes) > 0:
                if len(topleftNodes) < len(self.nodes):
                    subregion = Region(topleftNodes)
                    self.subregions.append(subregion)
                else:
                    for n in topleftNodes:
                        subregion = Region([n])
                        self.subregions.append(subregion)

            if len(bottomleftNodes) > 0:
                if len(bottomleftNodes) < len(self.nodes):
                    subregion = Region(bottomleftNodes)
                    self.subregions.append(subregion)
                else:
                    for n in bottomleftNodes:
                        subregion = Region([n])
                        self.subregions.append(subregion)

            if len(toprightNodes) > 0:
                if len(toprightNodes) < len(self.nodes):
                    subregion = Region(toprightNodes)
                    self.subregions.append(subregion)
                else:
                    for n in toprightNodes:
                        subregion = Region([n])
                        self.subregions.append(subregion)

            if len(bottomrightNodes) > 0:
                if len(bottomrightNodes) < len(self.nodes):
                    subregion = Region(bottomrightNodes)
                    self.subregions.append(subregion)
                else:
                    for n in bottomrightNodes:
                        subregion = Region([n])
                        self.subregions.append(subregion)
            '''
            '''
            leftNodes = []
            rightNodes = []
            for n in self.nodes:
                if n.position[0] < self.massCenter[0]:
                    leftNodes.append(n)
                else:
                    rightNodes.append(n)

            topleftNodes = []
            bottomleftNodes = []
            for n in leftNodes:
                if n.position[1] < self.massCenter[1]:
                    topleftNodes.append(n)
                else:
                    bottomleftNodes.append(n)

            toprightNodes = []
            bottomrightNodes = []
            for n in rightNodes:
                if n.position[1] < self.massCenter[1]:
                    toprightNodes.append(n)
                else:
                    bottomrightNodes.append(n)

            if len(topleftNodes) > 0:
                if len(topleftNodes) < len(self.nodes):
                    subregion = Region(topleftNodes)
                    self.subregions.append(subregion)
                else:
                    for n in topleftNodes:
                        subregion = Region([n])
                        self.subregions.append(subregion)

            if len(bottomleftNodes) > 0:
                if len(bottomleftNodes) < len(self.nodes):
                    subregion = Region(bottomleftNodes)
                    self.subregions.append(subregion)
                else:
                    for n in bottomleftNodes:
                        subregion = Region([n])
                        self.subregions.append(subregion)

            if len(toprightNodes) > 0:
                if len(toprightNodes) < len(self.nodes):
                    subregion = Region(toprightNodes)
                    self.subregions.append(subregion)
                else:
                    for n in toprightNodes:
                        subregion = Region([n])
                        self.subregions.append(subregion)

            if len(bottomrightNodes) > 0:
                if len(bottomrightNodes) < len(self.nodes):
                    subregion = Region(bottomrightNodes)
                    self.subregions.append(subregion)
                else:
                    for n in bottomrightNodes:
                        subregion = Region([n])
                        self.subregions.append(subregion)
            '''
            dim = self.massCenter.shape[0]
            dtype = self.massCenter.dtype
            
            powers = np.array([2**i for i in range(dim)], dtype=np.int_)
            def convert_to_number(x, y):
                return (x * y).sum()
            
            # Manual unrolling of general case for optimisation...
            if dim == 2:
                nodes_0 = []
                nodes_1 = []
                nodes_2 = []
                nodes_3 = []
                
                for n in self.nodes:
                    i = convert_to_number(n.position < self.massCenter, powers)
                    if i == 0:
                        nodes_0.append(n)
                    elif i == 1:
                        nodes_1.append(n)
                    elif i == 2:
                        nodes_2.append(n)
                    elif i == 3:
                        nodes_3.append(n)
                
                if len(nodes_0) > 0:
                    if len(nodes_0) < len(self.nodes):
                        subregion = Region(nodes_0, dim, dtype)
                        self.subregions.append(subregion)
                    else:
                        for n in nodes_0:
                            subregion = Region([n], dim, dtype)
                            self.subregions.append(subregion)
                            
                if len(nodes_1) > 0:
                    if len(nodes_1) < len(self.nodes):
                        subregion = Region(nodes_1, dim, dtype)
                        self.subregions.append(subregion)
                    else:
                        for n in nodes_1:
                            subregion = Region([n], dim, dtype)
                            self.subregions.append(subregion)
                            
                if len(nodes_2) > 0:
                    if len(nodes_2) < len(self.nodes):
                        subregion = Region(nodes_2, dim, dtype)
                        self.subregions.append(subregion)
                    else:
                        for n in nodes_2:
                            subregion = Region([n], dim, dtype)
                            self.subregions.append(subregion)
                            
                if len(nodes_3) > 0:
                    if len(nodes_3) < len(self.nodes):
                        subregion = Region(nodes_3, dim, dtype)
                        self.subregions.append(subregion)
                    else:
                        for n in nodes_3:
                            subregion = Region([n], dim, dtype)
                            self.subregions.append(subregion)
            
            elif dim == 3:
                nodes_0 = []
                nodes_1 = []
                nodes_2 = []
                nodes_3 = []
                nodes_4 = []
                nodes_5 = []
                nodes_6 = []
                nodes_7 = []
                
                for n in self.nodes:
                    i = convert_to_number(n.position < self.massCenter, powers)
                    if i == 0:
                        nodes_0.append(n)
                    elif i == 1:
                        nodes_1.append(n)
                    elif i == 2:
                        nodes_2.append(n)
                    elif i == 3:
                        nodes_3.append(n)
                    elif i == 4:
                        nodes_4.append(n)
                    elif i == 5:
                        nodes_5.append(n)
                    elif i == 6:
                        nodes_6.append(n)
                    elif i == 7:
                        nodes_7.append(n)
                
                if len(nodes_0) > 0:
                    if len(nodes_0) < len(self.nodes):
                        subregion = Region(nodes_0, dim, dtype)
                        self.subregions.append(subregion)
                    else:
                        for n in nodes_0:
                            subregion = Region([n], dim, dtype)
                            self.subregions.append(subregion)
                            
                if len(nodes_1) > 0:
                    if len(nodes_1) < len(self.nodes):
                        subregion = Region(nodes_1, dim, dtype)
                        self.subregions.append(subregion)
                    else:
                        for n in nodes_1:
                            subregion = Region([n], dim, dtype)
                            self.subregions.append(subregion)
                            
                if len(nodes_2) > 0:
                    if len(nodes_2) < len(self.nodes):
                        subregion = Region(nodes_2, dim, dtype)
                        self.subregions.append(subregion)
                    else:
                        for n in nodes_2:
                            subregion = Region([n], dim, dtype)
                            self.subregions.append(subregion)
                            
                if len(nodes_3) > 0:
                    if len(nodes_3) < len(self.nodes):
                        subregion = Region(nodes_3, dim, dtype)
                        self.subregions.append(subregion)
                    else:
                        for n in nodes_3:
                            subregion = Region([n], dim, dtype)
                            self.subregions.append(subregion)
                            
                if len(nodes_4) > 0:
                    if len(nodes_4) < len(self.nodes):
                        subregion = Region(nodes_4, dim, dtype)
                        self.subregions.append(subregion)
                    else:
                        for n in nodes_4:
                            subregion = Region([n], dim, dtype)
                            self.subregions.append(subregion)
                            
                if len(nodes_5) > 0:
                    if len(nodes_5) < len(self.nodes):
                        subregion = Region(nodes_5, dim, dtype)
                        self.subregions.append(subregion)
                    else:
                        for n in nodes_5:
                            subregion = Region([n], dim, dtype)
                            self.subregions.append(subregion)
                            
                if len(nodes_6) > 0:
                    if len(nodes_6) < len(self.nodes):
                        subregion = Region(nodes_6, dim, dtype)
                        self.subregions.append(subregion)
                    else:
                        for n in nodes_6:
                            subregion = Region([n], dim, dtype)
                            self.subregions.append(subregion)
                            
                if len(nodes_7) > 0:
                    if len(nodes_7) < len(self.nodes):
                        subregion = Region(nodes_7, dim, dtype)
                        self.subregions.append(subregion)
                    else:
                        for n in nodes_7:
                            subregion = Region([n], dim, dtype)
                            self.subregions.append(subregion)
            
            else:
                nodes_by_region = [[]] * (2**dim)
                
                for n in self.nodes:
                    i = convert_to_number(n.position < self.massCenter, powers)
                    nodes_by_region[i].append(n)
                
                for nodes in nodes_by_region:
                    if len(nodes) > 0:
                        if len(nodes) < len(self.nodes):
                            subregion = Region(nodes, dim, dtype)
                            self.subregions.append(subregion)
                        else:
                            for n in nodes:
                                subregion = Region([n], dim, dtype)
                                self.subregions.append(subregion)
            
            for subregion in self.subregions:
                subregion.buildSubRegions()

    def applyForce(self, n, theta, coefficient=0):
        if len(self.nodes) < 2:
            linRepulsion(n, self.nodes[0], coefficient)
        else:
            #distance = sqrt((n.x - self.massCenterX) ** 2 + (n.y - self.massCenterY) ** 2)
            distance = np.linalg.norm(n.position - self.massCenter)
            if distance * theta > self.size:
                linRepulsion_region(n, self, coefficient)
            else:
                for subregion in self.subregions:
                    subregion.applyForce(n, theta, coefficient)

    def applyForceOnNodes(self, nodes, theta, coefficient=0):
        for n in nodes:
            self.applyForce(n, theta, coefficient)


# Adjust speed and apply forces step
def adjustSpeedAndApplyForces(nodes, speed, speedEfficiency, jitterTolerance):
    # Auto adjust speed.
    totalSwinging = 0.0  # How much irregular movement
    totalEffectiveTraction = 0.0  # How much useful movement
    for n in nodes:
        #swinging = sqrt((n.old_dx - n.dx) * (n.old_dx - n.dx) + (n.old_dy - n.dy) * (n.old_dy - n.dy))
        swinging = np.linalg.norm(n.old_delta - n.delta)
        totalSwinging += n.mass * swinging
        #totalEffectiveTraction += .5 * n.mass * sqrt((n.old_dx + n.dx) * (n.old_dx + n.dx) + (n.old_dy + n.dy) * (n.old_dy + n.dy))
        totalEffectiveTraction += .5 * n.mass * np.linalg.norm(n.old_delta + n.delta)

    # Optimize jitter tolerance.  The 'right' jitter tolerance for
    # this network. Bigger networks need more tolerance. Denser
    # networks need less tolerance. Totally empiric.
    estimatedOptimalJitterTolerance = .05 * sqrt(len(nodes))
    minJT = sqrt(estimatedOptimalJitterTolerance)
    maxJT = 10
    jt = jitterTolerance * max(minJT,
                               min(maxJT, estimatedOptimalJitterTolerance * totalEffectiveTraction / (
                                   len(nodes) * len(nodes))))

    minSpeedEfficiency = 0.05

    # Protective against erratic behavior
    if totalSwinging / totalEffectiveTraction > 2.0:
        if speedEfficiency > minSpeedEfficiency:
            speedEfficiency *= .5
        jt = max(jt, jitterTolerance)

    if totalSwinging == 0:
        targetSpeed = float('inf')
    else:
        targetSpeed = jt * speedEfficiency * totalEffectiveTraction / totalSwinging

    if totalSwinging > jt * totalEffectiveTraction:
        if speedEfficiency > minSpeedEfficiency:
            speedEfficiency *= .7
    elif speed < 1000:
        speedEfficiency *= 1.3

    # But the speed shoudn't rise too much too quickly, since it would
    # make the convergence drop dramatically.
    maxRise = .5
    speed = speed + min(targetSpeed - speed, maxRise * speed)

    # Apply forces.
    #
    # Need to add a case if adjustSizes ("prevent overlap") is
    # implemented.
    for n in nodes:
        #swinging = n.mass * sqrt((n.old_dx - n.dx) * (n.old_dx - n.dx) + (n.old_dy - n.dy) * (n.old_dy - n.dy))
        swinging = n.mass * np.linalg.norm(n.old_delta - n.delta)
        factor = speed / (1.0 + sqrt(speed * swinging))
        #n.x = n.x + (n.dx * factor)
        #n.y = n.y + (n.dy * factor)
        n.position += n.delta * factor

    values = {}
    values['speed'] = speed
    values['speedEfficiency'] = speedEfficiency

    return values


try:
    import cython

    if not cython.compiled:
        print("Warning: uncompiled fa2util module.  Compile with cython for a 10-100x speed boost.")
except:
    print("No cython detected.  Install cython and compile the fa2util module for a 10-100x speed boost.")
