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

"""Analytical example from Section 6.3.3 of F. Trespalacions Ph.D. Thesis (2015)

Analytical example for a nonconvex GDP with 2 disjunctions, each with 2 disjuncts.

Ref:
    ANALYTICAL NONCONVEX GDP EXAMPLE.
    FRANCISCO TRESPALACIOS , PH.D. THESIS (EQ 6.6) , 2015.
    CARNEGIE-MELLON UNIVERSITY , PITTSBURGH , PA.

Solution is 4.46 with (Z, x1, x2) = (4.46, 1.467, 0.833),
with the first and second disjuncts active in
the first and second disjunctions, respectively.

Pyomo model implementation by @bernalde and @qtothec.

"""

import pyomo.environ as pyo
from pyomo.gdp import Disjunction


def build_simple_nonconvex_gdp():
    """Build the Analytical Problem."""
    m = pyo.ConcreteModel(name="Example 6.3.3")

    # Variables x1 and x2
    m.x1 = pyo.Var(bounds=(0, 5), doc="variable x1")
    m.x2 = pyo.Var(bounds=(0, 3), doc="variable x2")
    m.obj = pyo.Objective(expr=5 + 0.2 * m.x1 - m.x2, doc="Minimize objective")

    m.disjunction1 = Disjunction(
        expr=[
            [
                m.x2 <= 0.4 * pyo.exp(m.x1 / 2.0),
                m.x2 <= 0.5 * (m.x1 - 2.5) ** 2 + 0.3,
                m.x2 <= 6.5 / (m.x1 / 0.3 + 2.0) + 1.0,
            ],
            [
                m.x2 <= 0.3 * pyo.exp(m.x1 / 1.8),
                m.x2 <= 0.7 * (m.x1 / 1.2 - 2.1) ** 2 + 0.3,
                m.x2 <= 6.5 / (m.x1 / 0.8 + 1.1),
            ],
        ]
    )
    m.disjunction2 = Disjunction(
        expr=[
            [
                m.x2 <= 0.9 * pyo.exp(m.x1 / 2.1),
                m.x2 <= 1.3 * (m.x1 / 1.5 - 1.8) ** 2 + 0.3,
                m.x2 <= 6.5 / (m.x1 / 0.8 + 1.1),
            ],
            [
                m.x2 <= 0.4 * pyo.exp(m.x1 / 1.5),
                m.x2 <= 1.2 * (m.x1 - 2.5) ** 2 + 0.3,
                m.x2 <= 6.0 / (m.x1 / 0.6 + 1.0) + 0.5,
            ],
        ]
    )

    return m


if __name__ == "__main__":
    model = build_simple_nonconvex_gdp()
    model.pprint()
    res = pyo.SolverFactory('gdpopt.gloa').solve(model, tee=True)

    model.display()
    print(res)
