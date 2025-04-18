#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.environ as pyo

# ---------------------------------------------
# @loop1
M = pyo.ConcreteModel()
M.x = pyo.Var(range(5))

s = 0
for i in range(5):
    s = s + M.x[i]
# @loop1
print(s)

# ---------------------------------------------
# @loop2
s = sum(M.x[i] for i in range(5))
# @loop2
print(s)

# ---------------------------------------------
# @loop3
s = sum(M.x[i] for i in range(5)) ** 2
# @loop3
print(s)

# ---------------------------------------------
# @prod
M = pyo.ConcreteModel()
M.x = pyo.Var(range(5))
M.z = pyo.Var()

# The product M.x[0] * M.x[1] * ... * M.x[4]
e1 = pyo.prod(M.x[i] for i in M.x)

# The product M.x[0]*M.z
e2 = pyo.prod([M.x[0], M.z])

# The product M.z*(M.x[0] + ... + M.x[4])
e3 = pyo.prod([sum(M.x[i] for i in M.x), M.z])
# @prod
print(e1)
print(e2)
print(e3)

# ---------------------------------------------
# @quicksum
M = pyo.ConcreteModel()
M.x = pyo.Var(range(5))

# Summation using the Python sum() function
e1 = sum(M.x[i] ** 2 for i in M.x)

# Summation using the Pyomo quicksum function
e2 = pyo.quicksum(M.x[i] ** 2 for i in M.x)
# @quicksum
print(e1)
print(e2)

# ---------------------------------------------
# @warning
M = pyo.ConcreteModel()
M.x = pyo.Var(range(5))

e = pyo.quicksum(M.x[i] ** 2 if i > 0 else M.x[i] for i in range(5))
# @warning
print(e)

# ---------------------------------------------
# @sum_product1
M = pyo.ConcreteModel()
M.z = pyo.RangeSet(5)
M.x = pyo.Var(range(10))
M.y = pyo.Var(range(10))

# Sum the elements of x
e1 = pyo.sum_product(M.x)

# Sum the product of elements in x and y
e2 = pyo.sum_product(M.x, M.y)

# Sum the product of elements in x and y, over the index set z
e3 = pyo.sum_product(M.x, M.y, index=M.z)
# @sum_product1
print(e1)
print(e2)
print(e3)

# ---------------------------------------------
# @sum_product2
# Sum the product of x_i/y_i
e1 = pyo.sum_product(M.x, denom=M.y)

# Sum the product of 1/(x_i*y_i)
e2 = pyo.sum_product(denom=(M.x, M.y))
# @sum_product2
print(e1)
print(e2)
