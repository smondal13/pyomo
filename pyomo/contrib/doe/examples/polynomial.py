# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import pyomo.environ as pyo

from pyomo.contrib.doe import DesignOfExperiments
from pyomo.contrib.parmest.experiment import Experiment


class PolynomialExperiment(Experiment):
    """A polynomial experiment with an analytically verifiable Jacobian."""

    def __init__(self, data=None, x1=1.0, x2=1.0):
        self.data = data
        self.x1 = x1
        self.x2 = x2
        self.model = None

    def get_labeled_model(self):
        if self.model is None:
            self.create_model()
            self.finalize_model()
            self.label_experiment()
        return self.model

    def create_model(self):
        """Define ``y = a*x1 + b*x2 + c*x1*x2 + d``."""
        model = self.model = pyo.ConcreteModel()

        # Input variables (independent variables)
        model.x1 = pyo.Var(bounds=(-5, 5), initialize=self.x1)
        model.x2 = pyo.Var(bounds=(-5, 5), initialize=self.x2)

        # Model coefficients (unknown parameters)
        model.a = pyo.Var(bounds=(-5, 5), initialize=2)
        model.b = pyo.Var(bounds=(-5, 5), initialize=-1)
        model.c = pyo.Var(bounds=(-5, 5), initialize=0.5)
        model.d = pyo.Var(bounds=(-5, 5), initialize=-1)

        # Unknown parameters are fixed at nominal values for local sensitivities.
        model.a.fix()
        model.b.fix()
        model.c.fix()
        model.d.fix()

        # Model output (dependent variable)
        model.y = pyo.Var(initialize=0)

        @model.Constraint()
        def output_equation(m):
            return m.y == m.a * m.x1 + m.b * m.x2 + m.c * m.x1 * m.x2 + m.d

    def finalize_model(self):
        """Finalize the model; this algebraic example needs no transformation."""

    def label_experiment(self):
        """Label the model components used by DesignOfExperiments."""
        model = self.model

        # Set measurement labels
        model.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        model.experiment_outputs[model.y] = None

        # Assume independent measurement error with unit standard deviation.
        model.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        model.measurement_error[model.y] = 1

        # Identify design variables (experiment inputs) for the model
        model.experiment_inputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        model.experiment_inputs[model.x1] = None
        model.experiment_inputs[model.x2] = None

        # Add unknown parameter labels using their nominal values.
        model.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        model.unknown_parameters.update(
            (param, pyo.value(param)) for param in (model.a, model.b, model.c, model.d)
        )


def run_polynomial_doe(solver=None):
    """Compute the polynomial example FIM using symbolic differentiation."""
    experiment = PolynomialExperiment(data=None)
    if solver is None:
        solver = pyo.SolverFactory("ipopt")

    doe_obj = DesignOfExperiments(
        experiment=[experiment],
        gradient_method="pynumero",
        fd_formula=None,
        objective_option="determinant",
        scale_constant_value=1,
        scale_nominal_param_value=False,
        solver=solver,
    )

    return doe_obj.compute_FIM()


if __name__ == "__main__":
    run_polynomial_doe()
