# -*- coding: utf-8 -*-
"""Workflow skeletons for charge-carrier mobility calculations with EPW."""

from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import WorkChain, ToContext, if_
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

from .b2w import EpwB2WWorkChain
from .base import EpwBaseWorkChain
from .intp import EpwBaseIntpWorkChain


class EpwMobilityIntpWorkChain(EpwBaseIntpWorkChain):
    """Base interpolation workflow tailored for mobility post-processing.

    The class inherits from :class:`EpwBaseIntpWorkChain` and only injects the
    EPW flags that are typically required to generate scattering rates on fine
    meshes.  The concrete post-processing of the interpolated data will be
    implemented in future iterations.
    """

    _INTP_NAMESPACE = 'mobility'

    _forced_parameters = EpwBaseIntpWorkChain._forced_parameters.copy()
    _forced_parameters['INPUTEPW'] = (
        EpwBaseIntpWorkChain._forced_parameters['INPUTEPW']
        | {
            'phinterp': True,
            'phonselfen': True,
            'transport': True,
        }
    )

    @classmethod
    def define(cls, spec):
        """Expose the parent specification and add mobility placeholders."""
        super().define(spec)

        spec.input(
            'temperatures',
            valid_type=orm.List,
            required=False,
            help='Optional list of temperatures (in K) for mobility evaluation.',
        )
        spec.output(
            'mobility_results',
            valid_type=orm.Dict,
            required=False,
            help='Placeholder node for mobility-related quantities.',
        )

    def prepare_process(self):
        """Inject mobility defaults before submitting the fine-grid EPW run."""
        super().prepare_process()

        parameters = self.ctx.inputs.epw.parameters.get_dict()
        inputepw = parameters.setdefault('INPUTEPW', {})

        if 'temps' not in inputepw and 'temperatures' in self.inputs:
            temps_list = self.inputs.temperatures.get_list()
            temps = ' '.join(str(t) for t in temps_list)
            if temps:
                inputepw['temps'] = temps
                inputepw['nstemp'] = min(len(temps_list), EpwBaseWorkChain._MAX_NSTEMP)

        self.ctx.inputs.epw.parameters = orm.Dict(parameters)

        try:
            settings_dict = self.ctx.inputs.epw.settings.get_dict()
        except AttributeError:
            settings_dict = {}

        retrieve_list = settings_dict.get('ADDITIONAL_RETRIEVE_LIST', [])
        retrieve_list.extend([
            'aiida.imself',
            'aiida.elself',
        ])
        settings_dict['ADDITIONAL_RETRIEVE_LIST'] = sorted(set(retrieve_list))
        self.ctx.inputs.epw.settings = orm.Dict(settings_dict)


class EpwMobilityWorkChain(ProtocolMixin, WorkChain):
    """High-level workflow orchestrating B2W and mobility interpolation steps."""

    _NAMESPACE = 'mobility'
    _B2W_NAMESPACE = EpwB2WWorkChain._NAMESPACE
    _MOBILITY_NAMESPACE = EpwMobilityIntpWorkChain._INTP_NAMESPACE

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files
        from . import protocols

        return files(protocols) / f'{cls._NAMESPACE}.yaml'

    @classmethod
    def define(cls, spec):
        """Define the workflow outline and exposed inputs/outputs."""
        super().define(spec)

        spec.input('structure', valid_type=orm.StructureData)
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(True))

        spec.expose_inputs(
            EpwB2WWorkChain,
            namespace=cls._B2W_NAMESPACE,
            exclude=(
                'structure',
                'clean_workdir',
            ),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the coarse-grid EPW and Wannier preparation.',
            },
        )

        spec.expose_inputs(
            EpwMobilityIntpWorkChain,
            namespace=cls._MOBILITY_NAMESPACE,
            exclude=(
                'structure',
                'clean_workdir',
                f'{cls._MOBILITY_NAMESPACE}.parent_folder_nscf',
                f'{cls._MOBILITY_NAMESPACE}.parent_folder_ph',
                f'{cls._MOBILITY_NAMESPACE}.parent_folder_chk',
            ),
            namespace_options={
                'required': True,
                'populate_defaults': False,
                'help': 'Inputs for the fine-grid mobility interpolation.',
            },
        )

        spec.outline(
            cls.setup,
            cls.validate_inputs,
            if_(cls.should_run_b2w)(
                cls.run_b2w,
                cls.inspect_b2w,
            ),
            cls.run_mobility,
            cls.inspect_mobility,
            cls.results,
        )

        spec.expose_outputs(
            EpwB2WWorkChain,
            namespace=cls._B2W_NAMESPACE,
            namespace_options={
                'required': False,
                'help': 'Outputs from the coarse-grid preparation workflow.',
            },
        )

        spec.expose_outputs(
            EpwMobilityIntpWorkChain,
            namespace=cls._MOBILITY_NAMESPACE,
            namespace_options={
                'required': True,
                'help': 'Outputs from the mobility interpolation workflow.',
            },
        )

        spec.exit_code(401, 'ERROR_SUB_PROCESS_B2W', 'The `EpwB2WWorkChain` sub process failed.')
        spec.exit_code(402, 'ERROR_SUB_PROCESS_MOBILITY', 'The mobility interpolation sub process failed.')
        spec.exit_code(403, 'ERROR_MISSING_PARENT_FOLDER', 'Missing `parent_folder_epw` when skipping the B2W step.')

    def setup(self):
        """Cache exposed input namespaces for later manipulation."""
        self.ctx.inputs_mobility = AttributeDict(
            self.exposed_inputs(EpwMobilityIntpWorkChain, namespace=self._MOBILITY_NAMESPACE)
        )

        if self._B2W_NAMESPACE in self.inputs:
            self.ctx.inputs_b2w = AttributeDict(
                self.exposed_inputs(EpwB2WWorkChain, namespace=self._B2W_NAMESPACE)
            )

    def validate_inputs(self):
        """Ensure that the interpolation step can locate its parent folders."""
        if self._B2W_NAMESPACE not in self.inputs:
            mobility_inputs = self.ctx.inputs_mobility[self._MOBILITY_NAMESPACE]
            if 'parent_folder_epw' not in mobility_inputs:
                return self.exit_codes.ERROR_MISSING_PARENT_FOLDER

    def should_run_b2w(self):
        """Return whether the coarse-grid workflow should be executed."""
        return self._B2W_NAMESPACE in self.inputs

    def run_b2w(self):
        """Launch the coarse-grid EPW/Wannier preparation workflow."""
        inputs = self.ctx.inputs_b2w
        inputs.structure = self.inputs.structure
        inputs.clean_workdir = self.inputs.clean_workdir

        if 'metadata' not in inputs:
            inputs.metadata = AttributeDict()
        inputs.metadata.call_link_label = self._B2W_NAMESPACE

        workchain = self.submit(EpwB2WWorkChain, **inputs)
        self.report(f'Launching `EpwB2WWorkChain`<{workchain.pk}>')

        return ToContext(workchain_b2w=workchain)

    def inspect_b2w(self):
        """Check the result of the coarse-grid workflow."""
        workchain = self.ctx.workchain_b2w

        if not workchain.is_finished_ok:
            self.report(f'`EpwB2WWorkChain`<{workchain.pk}> failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_B2W

        self.report(f'`EpwB2WWorkChain`<{workchain.pk}> finished successfully')

        self.out_many(
            self.exposed_outputs(
                workchain,
                EpwB2WWorkChain,
                namespace=self._B2W_NAMESPACE,
            )
        )

        parent_folder_epw = workchain.outputs.epw_base.remote_stash
        self.ctx.inputs_mobility[self._MOBILITY_NAMESPACE].parent_folder_epw = parent_folder_epw

    def run_mobility(self):
        """Submit the mobility interpolation workflow."""
        inputs = self.ctx.inputs_mobility
        inputs.structure = self.inputs.structure
        inputs.clean_workdir = self.inputs.clean_workdir

        if 'metadata' not in inputs:
            inputs.metadata = AttributeDict()
        inputs.metadata.call_link_label = self._MOBILITY_NAMESPACE

        workchain = self.submit(EpwMobilityIntpWorkChain, **inputs)
        self.report(f'Launching `EpwMobilityIntpWorkChain`<{workchain.pk}>')

        return ToContext(workchain_mobility=workchain)

    def inspect_mobility(self):
        """Validate the result of the mobility interpolation workflow."""
        workchain = self.ctx.workchain_mobility

        if not workchain.is_finished_ok:
            self.report(f'`EpwMobilityIntpWorkChain`<{workchain.pk}> failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_MOBILITY

        self.report(f'`EpwMobilityIntpWorkChain`<{workchain.pk}> finished successfully')

    def results(self):
        """Expose the mobility outputs to the outside world."""
        self.out_many(
            self.exposed_outputs(
                self.ctx.workchain_mobility,
                EpwMobilityIntpWorkChain,
                namespace=self._MOBILITY_NAMESPACE,
            )
        )
