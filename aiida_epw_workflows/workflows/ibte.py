# -*- coding: utf-8 -*-
"""Work chain for computing the spectral function."""
from aiida import orm
from aiida.engine import calcfunction, ToContext, if_


from .intp import EpwBaseIntpWorkChain

class EpwIBTEWorkChain(EpwBaseIntpWorkChain):
    """Work chain to compute the spectral function.
    It will run the `EpwB2WWorkChain` for the electron-phonon coupling matrix on Wannier basis.
    and then the `EpwBaseWorkChain` for interpolation to a fine k/q-grid. the spectral function is computed
    from the interpolated grids.
    """

    _INTP_NAMESPACE = 'ibte'
    _ALL_NAMESPACES = [EpwBaseIntpWorkChain._B2W_NAMESPACE, _INTP_NAMESPACE]

    _forced_parameters =  EpwBaseIntpWorkChain._forced_parameters.copy()
    _forced_parameters['INPUTEPW']  = EpwBaseIntpWorkChain._forced_parameters['INPUTEPW'] | {
          'etf_mem': 3,
          'scattering': True,
          'scattering_serta': True,
          'int_mob': False,
          'carrier': True,
          'iterative_bte': True,
          'epmatkqread': False,
        }

    @classmethod
    def validate_inputs(cls, inputs, ctx=None):
        """Validate the inputs."""
        return None

    @classmethod
    def define(cls, spec):
        """Define the work chain specification."""
        super().define(spec)

        spec.input('is_polar', valid_type=orm.Bool, default=lambda: orm.Bool(False))
        spec.input('use_serta', valid_type=orm.Bool, default=lambda: orm.Bool(True))
        spec.inputs[cls._INTP_NAMESPACE].validator = cls.validate_inputs
        spec.inputs.validator = cls.validate_inputs
        spec.exit_code(
            401, 'ERROR_SUB_PROCESS_B2W',
            message='The `EpwIBTEWorkChain` failed at `b2w` step.')
        spec.exit_code(
            402, 'ERROR_SUB_PROCESS_IBTE',
            message='The `EpwIBTEWorkChain` failed at `ibte` step.')

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
        from_ibte_workchain
        ):
        """Return a builder prepopulated with inputs extracted from the a2f workchain.
        :param from_a2f_workchain: The a2f workchain.
        :return: The builder.
        """
        return super()._get_builder_restart(
            from_intp_workchain=from_ibte_workchain,
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
        builder = super().get_builder_from_protocol(
            codes,
            structure,
            protocol,
            overrides,
            **kwargs
            )

        return builder

    @staticmethod
    def calculate_degaussq(workchain_a2f):
        """Calculate the degaussq for the a2f calculation based on the `ph.x` results.
        :param workchain_a2f: The a2f workchain.
        :return: The degaussq.
        """
        import numpy

        output_parameters = workchain_a2f.inputs.parent_folder_epw.creator.inputs.parent_folder_ph.creator.outputs.output_parameters.get_dict()
        number_of_qpoints = output_parameters.get('number_of_qpoints')
        dynamical_matricies = numpy.array([
            output_parameters.get(f'dynamical_matricies_{iq+1}').get('frequencies') for iq in range(number_of_qpoints)
            ])

        max_frequency = numpy.max(dynamical_matricies)
        degaussq = max_frequency / 100

        return degaussq

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
            'IBTEvel_sup_0.fmt',
            'inv_tau_0.fmt',
            'inv_taucb_0.fmt',
            'm_effective.fmt',
            ]

        self.ctx.inputs.epw.settings = orm.Dict(settings)

        parameters = self.ctx.inputs.epw.parameters.get_dict()
        
        if self.inputs.is_polar:
            parameters['INPUTEPW']['lpolar'] = True
        if self.inputs.use_serta:
            parameters['INPUTEPW']['scattering'] = True
            parameters['INPUTEPW']['scattering_serta'] = True
        
        etf_mem = parameters['INPUTEPW']['etf_mem']
        if etf_mem == 3:
            parameters['INPUTEPW']['efermi_read'] = True
            parameters['INPUTEPW']['fermi_energy'] = self.ctx.inputs.parent_folder_epw.creator.outputs.output_parameters['fermi_energy_coarse']

        self.ctx.inputs.epw.parameters = orm.Dict(parameters)
    
    def inspect_process(self):
        """Verify that the `EpwBaseWorkChain` finished successfully."""
        intp_workchain = self.ctx.workchain_intp

        if not intp_workchain.is_finished_ok:
            self.report(f'`EpwBaseWorkChain`<{intp_workchain.pk}> failed with exit status {intp_workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_IBTE

        self.report(f'`EpwBaseWorkChain`<{intp_workchain.pk}> finished successfully')

    def results(self):
        """Only the basic results are retrieved:
        - output_parameters
        - remote_folder
        """

        super().results()

