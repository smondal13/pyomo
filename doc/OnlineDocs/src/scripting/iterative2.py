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

# iterative2.py

import pyomo.environ as pyo

# Create a solver
opt = pyo.SolverFactory('cplex')

#
# A simple model with binary variables and
# an empty constraint list.
#
model = pyo.AbstractModel()
model.n = pyo.Param(default=4)
model.x = pyo.Var(pyo.RangeSet(model.n), within=pyo.Binary)


def o_rule(model):
    return pyo.summation(model.x)


model.o = pyo.Objective(rule=o_rule)
model.c = pyo.ConstraintList()

# Create a model instance and optimize
instance = model.create_instance()
results = opt.solve(instance)
instance.display()

# "flip" the value of x[2] (it is binary)
# then solve again
# @Flip_value_before_solve_again

if pyo.value(instance.x[2]) == 0:
    instance.x[2].fix(1)
else:
    instance.x[2].fix(0)

results = opt.solve(instance)
# @Flip_value_before_solve_again
instance.display()
