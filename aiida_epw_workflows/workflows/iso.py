# -*- coding: utf-8 -*-
"""Work chain for computing the critical temperature based off an `EpwWorkChain`."""

from aiida import orm
from .intp import EpwBaseIntpWorkChain
from .base import EpwBaseWorkChain
from ..tools.calculators import calculate_iso_tc

### Note: We should later add the support for:
#      Error in routine crit_temp_solver (1):
#      Convergence (tc_linear) was not reached


class EpwIsoWorkChain(EpwBaseIntpWorkChain):
    """Work chain to compute the superconductivity based on Migdal-Eliashberg theory.
    This work chain will use isotropic approximation and solve the linearized Eliashberg equation.
    """

    _INTP_NAMESPACE = 'iso'

    _forced_parameters =  EpwBaseIntpWorkChain._forced_parameters.copy()
    _forced_parameters['INPUTEPW']  = EpwBaseIntpWorkChain._forced_parameters['INPUTEPW'] | {
          'eliashberg': True,
          'ephwrite': True,
          'liso': True,
          'limag': True,
          'lpade': False,
          'laniso': False,
          'tc_linear': True,
          'tc_linear_solver': 'power'
        }

    _MIN_TEMP = 1.0

    @classmethod
    def define(cls, spec):
        """Define the work chain specification.
        """
        super().define(spec)

        spec.input('estimated_Tc_iso', valid_type=orm.Float, default=lambda: orm.Float(40.0),
            help='The estimated Tc for the iso calculation.')
        spec.input('linearized_Eliashberg', valid_type=orm.Bool, default=lambda: orm.Bool(True),
            help='Whether to use the linearized Eliashberg function.')

        spec.output('Tc_iso', valid_type=orm.Float,
                    help='The isotropic Tc interpolated from the a2f file.')

        spec.exit_code(401, 'ERROR_SUB_PROCESS_B2W',
            message='The `EpwIsoWorkChain` failed at `b2w` step.')
        spec.exit_code(402, 'ERROR_SUB_PROCESS_ISO',
            message='The `EpwIsoWorkChain` failed at `iso` step.')
        spec.exit_code(403, 'ERROR_TEMPERATURE_OUT_OF_RANGE',
            message='There is no gap at some temperature in the `iso` step.')
            
    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files
        from . import protocols
        return files(protocols) / f'{cls._INTP_NAMESPACE}.yaml'

    @classmethod
    def get_builder_restart(
        cls,
        from_iso_workchain
        ):
        """Return a builder prepopulated with inputs extracted from the iso workchain.
        :param from_iso_workchain: The iso workchain from which to restart.
        :type from_iso_workchain: :class:`aiida.orm.Node`
        :return: A builder instance with the inputs prepopulated.
        :rtype: :class:`aiida.engine.ProcessBuilder`
        """
        return super()._get_builder_restart(
            from_intp_workchain=from_iso_workchain,
            )


    @classmethod
    def get_builder_from_protocol(
            cls,
            codes,
            structure,
            protocol=None,
            overrides=None,
            **kwargs
        ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.
        :param codes: The codes should be a dictionary with the following keys:
            - pw: The code for the pw.x calculation.
            - ph: The code for the ph.x calculation.
            - epw: The code for the epw.x calculation.
            - pw2wannier90: The code for the pw2wannier90.x calculation.
            - wannier: The code for the wannier90.x calculation.
        :type codes: dict
        :param structure: The structure to use for the calculations.
        :type structure: :class:`aiida.orm.StructureData`
        :param protocol: The protocol to use for the calculations.
        :type protocol: str
        :param overrides: The overrides to use for the calculations.
        """
        builder = super().get_builder_from_protocol(
            codes,
            structure,
            protocol,
            overrides,
            **kwargs
        )

        return builder

    def prepare_process(self):
        """Prepare the `EpwBaseWorkChain`.
        It will set the necessary inputs parameters for an isotropic calculation.
        It will set the necessary retrieve items from an isotropic calculation:
        - a2f
        - a2f_proj
        - dos
        - phdos
        - phdos_proj
        - lambda_FS
        - lambda_k_pairs
        """

        super().prepare_process()

        parameters = self.ctx.inputs.epw.parameters.get_dict()
        temps = f'{self._MIN_TEMP} {self.inputs.estimated_Tc_iso.value*1.5}'
        parameters['INPUTEPW']['temps'] = temps
        parameters['INPUTEPW']['nstemp'] = min(
            int(self.inputs.estimated_Tc_iso.value*1.5),
            EpwBaseWorkChain._MAX_NSTEMP
            )

        self.ctx.inputs.epw.parameters = orm.Dict(parameters)

        try:
            settings = self.ctx.inputs.epw.settings.get_dict()
        except AttributeError:
            settings = {}

        settings['ADDITIONAL_RETRIEVE_LIST'] = [
            'aiida.lambda_FS',
            'aiida.lambda_k_pairs'
            ]

        self.ctx.inputs.epw.settings = orm.Dict(settings)

    def inspect_process(self):
        """Verify that the `EpwBaseWorkChain` finished successfully.
        It will calculate the isotropic Tc from the `EpwBaseWorkChain` outputs.
        """
        intp_workchain = self.ctx.workchain_intp

        if not intp_workchain.is_finished_ok:
            self.report(f'`EpwBaseWorkChain`<{intp_workchain.pk}> failed with exit status {intp_workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_ISO

        self.report(f'`EpwBaseWorkChain`<{intp_workchain.pk}> finished successfully')

        inputs = {
                'max_eigenvalue':intp_workchain.outputs.max_eigenvalue,
                'metadata': {
                    'call_link_label': 'calculate_iso_tc'
                }
            }

        Tc_iso = calculate_iso_tc(**inputs)
        self.report(f'The isotropic Tc is {Tc_iso.value} K.')

        self.ctx.Tc_iso = Tc_iso

    def results(self):
        """Add the most important results `Tc_iso` to the outputs of the work chain.
        """

        super().results()

        self.out('Tc_iso', self.ctx.Tc_iso)

