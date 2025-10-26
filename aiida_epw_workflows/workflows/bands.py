# -*- coding: utf-8 -*-
"""Work chain for computing the critical temperature based off an `EpwWorkChain`."""
from aiida import orm


from .intp import EpwBaseIntpWorkChain

class EpwBandsWorkChain(EpwBaseIntpWorkChain):
    """Work chain to interpolate the electron and phonon band structure.
    It will run the `EpwB2WWorkChain` for the electron-phonon coupling matrix on Wannier basis.
    and then the `EpwBaseWorkChain` for interpolation along the high-symmetry lines.
    """

    _INTP_NAMESPACE = 'bands'
    _ALL_NAMESPACES = [EpwBaseIntpWorkChain._B2W_NAMESPACE, _INTP_NAMESPACE]

    _forced_parameters =  EpwBaseIntpWorkChain._forced_parameters.copy()
    _forced_parameters['INPUTEPW']  = EpwBaseIntpWorkChain._forced_parameters['INPUTEPW'] | {
          'band_plot': True,
          'mp_mesh_k': False,
        }

    _MIN_FREQ = -1.0 # meV ~ 8.1 cm-1

    @classmethod
    def validate_inputs(cls, inputs, ctx=None):
        """Validate the inputs."""
        return None

    @classmethod
    def define(cls, spec):
        """Define the work chain specification."""
        super().define(spec)


        spec.inputs[cls._INTP_NAMESPACE].validator = cls.validate_inputs
        spec.inputs.validator = cls.validate_inputs

        spec.input(
            'bands_kpoints', valid_type=orm.KpointsData, required=False,
            help='The kpoints to use for the band structure.'
            )

        spec.input(
            'bands_kpoints_distance', valid_type=orm.Float, required=False,
            help='The distance between the kpoints in the band structure.'
            )

        spec.output(
            "seekpath_parameters",
            valid_type=orm.Dict,
            required=False,
            help="The parameters used in the SeeKpath call to normalize the input or relaxed structure.",
        )

        spec.exit_code(
            402, 'ERROR_SUB_PROCESS_BANDS',
            message='The `epw.x` workflow failed.'
            )

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files
        from . import protocols
        return files(protocols) / f'{cls._INTP_NAMESPACE}.yaml'

    @classmethod
    def get_builder_restart(
        cls,
        from_bands_workchain
        ):
        """Return a builder prepopulated with inputs from a previous `EpwBandsWorkChain`.
        :param from_bands_workchain: The `EpwBandsWorkChain` node from which to restart.
        :type from_bands_workchain: :class:`aiida.orm.Node`
        :return: A builder instance with the inputs prepopulated.
        :rtype: :class:`aiida.engine.ProcessBuilder`
        """
        return super()._get_builder_restart(
            from_intp_workchain=from_bands_workchain,
            )

    @classmethod
    def get_builder_from_protocol(
            cls,
            codes,
            structure,
            protocol=None,
            overrides=None,
            kpoints=None,
            **kwargs
        ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.
        :param codes: The codes to use for the calculations.
        :type codes: dict
        :param structure: The structure to use for the calculations.
        :type structure: :class:`aiida.orm.StructureData`
        :param protocol: The protocol to use for the calculations.
        :type protocol: str
        :param overrides: The overrides to use for the calculations.
        :type overrides: dict
        :param kwargs: Additional keyword arguments.
        :type kwargs: dict
        :return: A builder instance with the inputs prepopulated.
        :rtype: :class:`aiida.engine.ProcessBuilder`
        """
        builder = super().get_builder_from_protocol(
            codes,
            structure,
            protocol,
            overrides,
            **kwargs
            )

        if kpoints:
            builder.kpoints = kpoints

        return builder

    def setup(self):
        """Setup the work chain.
        The default k/q points setup only works for uniform k/q grids.
        Here we use the `seekpath` plugin to generate the high-symmetry k/q points.
        """
        super().setup()

        from aiida_quantumespresso.calculations.functions.seekpath_structure_analysis import seekpath_structure_analysis

        if 'kpoints' in self.inputs:
            self.ctx.bands_kpoints = self.inputs.bands_kpoints
        else:
            inputs = {
                "structure": self.inputs.structure,
                'metadata': {
                    'call_link_label': 'seekpath'
                }
            }

            if 'bands_kpoints_distance' in self.inputs:
                inputs['reference_distance'] = self.inputs.bands_kpoints_distance

            result = seekpath_structure_analysis(**inputs)
            self.ctx.bands_kpoints = result['explicit_kpoints']

            self.out('seekpath_parameters', result['parameters'])

    def prepare_process(self):
        """Prepare the `EpwBaseWorkChain`.
        We remove the `qfpoints_distance` and `kfpoints_factor` from the inputs.
        We set up the necessary inputs parameters for the `EpwBaseWorkChain`.
        """
        super().prepare_process()

        self.ctx.inputs.pop('qfpoints_distance')
        self.ctx.inputs.pop('kfpoints_factor')

        self.ctx.inputs.qfpoints = self.ctx.bands_kpoints
        self.ctx.inputs.kfpoints = self.ctx.bands_kpoints

    def inspect_process(self):
        """Verify that the epw.x workflow finished successfully."""
        intp_workchain = self.ctx.workchain_intp

        if not intp_workchain.is_finished_ok:
            self.report(f'`EpwBaseWorkChain`<{intp_workchain.pk}> failed with exit status {intp_workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_BANDS

        self.report(f'`EpwBaseWorkChain`<{intp_workchain.pk}> finished successfully')