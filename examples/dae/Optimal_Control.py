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

# Sample Problem 1 (Ex 1 from Dynopt Guide)
#
# 	min X2(tf)
# 	s.t.	X1_dot = u			X1(0) = 1
# 		X2_dot = X1^2 + u^2		X2(0) = 0
# 		tf = 1

import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar

m = pyo.ConcreteModel()

m.t = ContinuousSet(bounds=(0, 1))

m.x1 = pyo.Var(m.t, bounds=(0, 1))
m.x2 = pyo.Var(m.t, bounds=(0, 1))
m.u = pyo.Var(m.t, initialize=0)

m.x1dot = DerivativeVar(m.x1)
m.x2dot = DerivativeVar(m.x2)

m.obj = pyo.Objective(expr=m.x2[1])


def _x1dot(M, i):
    if i == 0:
        return pyo.Constraint.Skip
    return M.x1dot[i] == M.u[i]


m.x1dotcon = pyo.Constraint(m.t, rule=_x1dot)


def _x2dot(M, i):
    if i == 0:
        return pyo.Constraint.Skip
    return M.x2dot[i] == M.x1[i] ** 2 + M.u[i] ** 2


m.x2dotcon = pyo.Constraint(m.t, rule=_x2dot)


def _init(M):
    yield M.x1[0] == 1
    yield M.x2[0] == 0
    yield pyo.ConstraintList.End


m.init_conditions = pyo.ConstraintList(rule=_init)
