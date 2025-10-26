.. _transport:

=======================
EpwTransportWorkChain
=======================

This work chain is used to compute the superconductor critical temperature on different level of theories.

It is an aggregation of the following work chains:

- PwRelaxWorkChain: relaxation of the structure
- PwBandsWorkChain: Calculation of the quantum ESPRESSO electron bands
- EpwB2WWorkChain: transformation from coarse-grid Bloch basis to Wannier basis
- EpwBandsWorkChain: Interpolation of the electron and phonon bands
- EpwA2FWorkChain: Calculation of the Eliashberg spectral function
- EpwIBTEWorkChain: Solve the iterative Boltzmann transport equation

It follows the following steps:

.. code-block:: python

    spec.outline(
        cls.setup,
        cls.validate_inputs,
        if_(cls.should_run_pw_relax)(
            cls.run_pw_relax,
            cls.inspect_pw_relax,
        ),
        if_(cls.should_run_pw_bands)(
            cls.run_pw_bands,
            cls.inspect_pw_bands,
        ),
        if_(cls.should_run_b2w)(
            cls.run_b2w,
            cls.inspect_b2w,
        ),
        if_(cls.should_run_bands)(
            cls.run_bands,
            cls.inspect_bands,
        ),
        if_(cls.should_run_a2f)(
            cls.run_a2f,
            cls.inspect_a2f,
        ),
        if_(cls.should_run_ibte)(
            cls.run_ibte,
            cls.inspect_ibte,
        ),
        cls.results
    )

Firstly, the workchain may do structural relaxation and calculation of qe electron bands, if necessary.

Then it will run the wannier90 and phonon calculations, if necessary.

Then it will run the EpwB2WWorkChain to transform the coarse-grid Bloch basis to Wannier basis.

Then it will run the EpwBandsWorkChain to interpolate the electron and phonon bands.

Then it will run the EpwA2FWorkChain to calculate the Eliashberg spectral function.

Then it will run the EpwIBTEWorkChain to solve the iterative Boltzmann transport equation.