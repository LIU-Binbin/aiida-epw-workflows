# -*- coding: utf-8 -*-
"""Work chain for computing the spectral function."""
from aiida import orm
from aiida.engine import calcfunction, ToContext, if_


from .intp import EpwBaseIntpWorkChain

class EpwA2fWorkChain(EpwBaseIntpWorkChain):
    """Work chain to compute the spectral function.
    It will run the `EpwB2WWorkChain` for the electron-phonon coupling matrix on Wannier basis.
    and then the `EpwBaseWorkChain` for interpolation to a fine k/q-grid. the spectral function is computed
    from the interpolated grids.
    """

    _INTP_NAMESPACE = 'a2f'
    _ALL_NAMESPACES = [EpwBaseIntpWorkChain._B2W_NAMESPACE, _INTP_NAMESPACE]

    # _forced_parameters =  EpwBaseIntpWorkChain._forced_parameters.copy()
    # _forced_parameters['INPUTEPW']  = EpwBaseIntpWorkChain._forced_parameters['INPUTEPW'] | {
    #       'eliashberg': True,
    #       'ephwrite': True,
    #     }

    INPUTEPW_A2F_DEFAULT = {
        'phonselfen': True,
        'a2f': True,
        'mp_mesh_k': False,
    }

    INPUTEPW_A2F_ELIASHBERG = {
        'eliashberg': True,
        'ephwrite': True,
    }

    @classmethod
    def validate_inputs(cls, inputs, ctx=None):
        """Validate the inputs."""
        return None

    @classmethod
    def define(cls, spec):
        """Define the work chain specification."""
        super().define(spec)

        
        spec.input('eliashberg', valid_type=orm.Bool, default=lambda: orm.Bool(True),
                    help='Whether to calculate superconductivity from spectral function.')
        spec.inputs[cls._INTP_NAMESPACE].validator = cls.validate_inputs
        spec.inputs.validator = cls.validate_inputs
        
        spec.exit_code(
            401, 'ERROR_SUB_PROCESS_B2W',
            message='The `EpwA2fWorkChain` failed at `b2w` step.')

        spec.exit_code(
            402, 'ERROR_SUB_PROCESS_A2F',
            message='The `EpwA2fWorkChain` failed at `a2f` step.'
            )

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols.
        :return: The path to the protocol file.
        """
        from importlib_resources import files
        from . import protocols
        return files(protocols) / f'{cls._INTP_NAMESPACE}.yaml'

    @classmethod
    def get_builder_restart(
        cls,
        from_a2f_workchain
        ):
        """Return a builder prepopulated with inputs extracted from the a2f workchain.
        :param from_a2f_workchain: The a2f workchain.
        :return: The builder.
        """
        return super()._get_builder_restart(
            from_intp_workchain=from_a2f_workchain,
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
        :param structure: The structure.
        :param protocol: The protocol.
        :param overrides: The overrides.
        :return: The builder.
        """
        inputs = cls.get_protocol_inputs(protocol, overrides)
        builder = super().get_builder_from_protocol(
            codes,
            structure,
            protocol,
            overrides,
            **kwargs
            )

        builder.eliashberg = orm.Bool(inputs.get('eliashberg', True))
        return builder

    def prepare_process(self):
        """Prepare for the `EpwBaseWorkChain`.
        Now it is only used to append some additional retrieve items:
        - aiida.a2f
        - aiida.a2f_proj
        - aiida.dos
        - aiida.phdos
        - aiida.phdos_proj
        - aiida.lambda_FS
        - aiida.lambda_k_pairs
        """

        super().prepare_process()
        try:
            settings = self.ctx.inputs.epw.settings.get_dict()
        except AttributeError:
            settings = {}

        settings['ADDITIONAL_RETRIEVE_LIST'] = [
            'aiida.lambda_FS',
            'aiida.lambda_k_pairs'
            ]

        self.ctx.inputs.epw.settings = orm.Dict(settings)

        parameters = self.ctx.inputs.epw.parameters.get_dict()
        if self.inputs.eliashberg:
            parameters['INPUTEPW'].update(self.INPUTEPW_A2F_ELIASHBERG)
        else:
            parameters['INPUTEPW'].update(self.INPUTEPW_A2F_DEFAULT)
        
        self.ctx.inputs.epw.parameters = orm.Dict(parameters)

    def inspect_process(self):
        """Verify that the `EpwBaseWorkChain` finished successfully."""
        intp_workchain = self.ctx.workchain_intp

        if not intp_workchain.is_finished_ok:
            self.report(f'`EpwBaseWorkChain`<{intp_workchain.pk}> failed with exit status {intp_workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_A2F

        self.report(f'`EpwBaseWorkChain`<{intp_workchain.pk}> finished successfully')

    def results(self):
        """Only the basic results are retrieved:
        - output_parameters
        - remote_folder
        """

        super().results()

