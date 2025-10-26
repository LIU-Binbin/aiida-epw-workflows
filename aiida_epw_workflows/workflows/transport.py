
from aiida import orm
from aiida.engine import WorkChain, ToContext, if_
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin
from aiida.common import AttributeDict, LinkType, NotExistentAttributeError
from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain
from aiida_quantumespresso.workflows.pw.bands import PwBandsWorkChain

from .b2w import EpwB2WWorkChain
from .bands import EpwBandsWorkChain

from .ibte import EpwIBTEWorkChain
from .a2f import EpwA2fWorkChain

class EpwTransportWorkChain(ProtocolMixin, WorkChain):
    """Workchain to calculate transport properties using EPW."""

    _NAMESPACE = 'transport'
    _PW_RELAX_NAMESPACE = "pw_relax"
    _PW_BANDS_NAMESPACE = "pw_bands"
    _B2W_NAMESPACE = EpwB2WWorkChain._NAMESPACE
    _BANDS_NAMESPACE = EpwBandsWorkChain._INTP_NAMESPACE
    _A2F_NAMESPACE = EpwA2fWorkChain._INTP_NAMESPACE
    _IBTE_NAMESPACE = EpwIBTEWorkChain._INTP_NAMESPACE


    @classmethod
    def define(cls, spec):
        # yapf: disable
        super().define(spec)
        spec.input('structure', valid_type=orm.StructureData)
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(False))
        spec.input('kfpoints_factor', required=False, valid_type=orm.Int, default=lambda: orm.Int(1))
        spec.input('qfpoints', required=False, valid_type=orm.KpointsData)
        spec.input('qfpoints_distance', required=False, valid_type=orm.Float, default=lambda: orm.Float(0.1))

        spec.expose_inputs(
            PwRelaxWorkChain,
            namespace=cls._PW_RELAX_NAMESPACE,
            exclude=(
                'structure',
                'clean_workdir',
            ),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the `PwRelaxWorkChain`.'
            }
        )
        spec.expose_inputs(
            PwBandsWorkChain,
            namespace=cls._PW_BANDS_NAMESPACE,
            exclude=(
                'structure',
                'clean_workdir',
                'bands_kpoints',
                'bands_kpoints_distance',
            ),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the `PwBandsWorkChain`.'
            }
        )

        spec.expose_inputs(
            EpwB2WWorkChain,
            namespace=cls._B2W_NAMESPACE,
            exclude=(
                'clean_workdir',
            ),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the `EpwB2WWorkChain`.'
            }
        )

        spec.expose_inputs(
            EpwBandsWorkChain,
            namespace=cls._BANDS_NAMESPACE,
            exclude=(
                'clean_workdir',
                'structure',
                f'{cls._BANDS_NAMESPACE}.parent_folder_nscf',
                f'{cls._BANDS_NAMESPACE}.parent_folder_chk',
                f'{cls._BANDS_NAMESPACE}.parent_folder_ph',
            ),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the `EpwBandsWorkChain`.'
            }
        )        
        
        spec.expose_inputs(
            EpwA2fWorkChain,
            namespace=cls._A2F_NAMESPACE,
            exclude=(
                'clean_workdir',
                'structure',
                f'{cls._A2F_NAMESPACE}.parent_folder_nscf',
                f'{cls._A2F_NAMESPACE}.parent_folder_chk',
                f'{cls._A2F_NAMESPACE}.parent_folder_ph',
            ),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the `EpwA2fWorkChain`.'
            }
        )

        spec.expose_inputs(
            EpwIBTEWorkChain,
            namespace=cls._IBTE_NAMESPACE,
            exclude=(
                'clean_workdir',
                'structure',
                f'{cls._IBTE_NAMESPACE}.parent_folder_nscf',
                f'{cls._IBTE_NAMESPACE}.parent_folder_chk',
                f'{cls._IBTE_NAMESPACE}.parent_folder_ph',
            ),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the `EpwIBTEWorkChain`.'
            }
        )

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
        
        spec.output(
            "seekpath_parameters",
            valid_type=orm.Dict,
            required=False,
            help="The parameters used in the SeeKpath call to normalize the input or relaxed structure.",
        )

        spec.expose_outputs(
            PwRelaxWorkChain,
            namespace=cls._PW_RELAX_NAMESPACE,
            namespace_options={
                'required': False,
            }
        )
        spec.expose_outputs(
            PwBandsWorkChain,
            namespace=cls._PW_BANDS_NAMESPACE,
            namespace_options={
                'required': False,
            }
        )

        spec.expose_outputs(
            EpwB2WWorkChain,
            namespace=cls._B2W_NAMESPACE,
            namespace_options={
                'required': False,
                'help': 'Outputs from the anisotropic `EpwCalculation`.'
            }
        )

        spec.expose_outputs(
            EpwBandsWorkChain,
            namespace=cls._BANDS_NAMESPACE,
            namespace_options={
                'required': False,
                'help': 'Outputs from the `EpwBandsWorkChain`.'
            }
        )

        spec.expose_outputs(
            EpwA2fWorkChain,
            namespace=cls._A2F_NAMESPACE,
            namespace_options={
                'required': False,
                'help': 'Outputs from the `EpwA2fWorkChain`.'
            }
        )
        spec.expose_outputs(
            EpwIBTEWorkChain,
            namespace=cls._IBTE_NAMESPACE,
            namespace_options={
                'required': False,
                'help': 'Outputs from the `EpwIBTEWorkChain`.'
            }
        )
        spec.exit_code(401, 'ERROR_SUB_PROCESS_PW_RELAX',
            message='The `pw_relax` sub process failed')
        spec.exit_code(402, 'ERROR_SUB_PROCESS_PW_BANDS',
            message='The `pw_bands` sub process failed')
        spec.exit_code(403, 'ERROR_SUB_PROCESS_B2W',
            message='The `b2w` sub process failed')
        spec.exit_code(404, 'ERROR_SUB_PROCESS_BANDS',
            message='The `bands` sub process failed')
        spec.exit_code(405, 'ERROR_SUB_PROCESS_A2F',
            message='The `a2f` sub process failed')
        spec.exit_code(406, 'ERROR_SUB_PROCESS_IBTE',
            message='The `ibte` sub process failed')

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files
        from . import protocols
        return files(protocols) / f'{cls._NAMESPACE}.yaml'

    @classmethod
    def get_protocol_overrides(cls) -> dict:
        """Get the ``overrides`` for default protocol.
        :return: The overrides.
        """
        from importlib_resources import files
        import yaml

        from . import protocols

        path = files(protocols) / f"{cls._NAMESPACE}.yaml"
        with path.open() as file:
            return yaml.safe_load(file)

    @classmethod
    def get_builder_from_protocol(
            cls,
            codes,
            structure,
            protocol='moderate',
            overrides=None,
            **kwargs
        ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.
        """
        inputs = cls.get_protocol_inputs(protocol, overrides)
        args = (codes, structure, protocol)

        builder = cls.get_builder()

        pw_relax_builder = PwRelaxWorkChain.get_builder_from_protocol(
            code=codes['pw'],
            structure=structure,
            overrides=inputs.get(cls._PW_RELAX_NAMESPACE, {}),
            **kwargs
        )

        pw_relax_builder.pop('structure', None)
        pw_relax_builder.pop('clean_workdir', None)
        pw_relax_builder.pop('base_final_scf', None)

        builder[cls._PW_RELAX_NAMESPACE]._data = pw_relax_builder._data

        # Set up the pw bands sub-workchain
        pw_bands_builder = PwBandsWorkChain.get_builder_from_protocol(
            code=codes['pw'],
            structure=structure,
            overrides=inputs.get(cls._PW_BANDS_NAMESPACE, {}),
        )
        pw_bands_builder.pop('relax', None)

        builder[cls._PW_BANDS_NAMESPACE]._data = pw_bands_builder._data

        b2w_builder = EpwB2WWorkChain.get_builder_from_protocol(
            *args,
            overrides=inputs.get(cls._B2W_NAMESPACE, None),
            **kwargs
        )

        builder[cls._B2W_NAMESPACE]._data = b2w_builder._data

        for (epw_namespace, epw_workchain_class) in (
            (cls._BANDS_NAMESPACE, EpwBandsWorkChain),
            (cls._A2F_NAMESPACE, EpwA2fWorkChain),
            (cls._IBTE_NAMESPACE, EpwIBTEWorkChain),
        ):
            epw_builder = epw_workchain_class.get_builder_from_protocol(
                *args,
                overrides=inputs.get(epw_namespace, None),
                **kwargs
            )

            epw_builder.pop(epw_workchain_class._B2W_NAMESPACE)

            builder[epw_namespace]._data = epw_builder._data


        builder.structure = structure
        builder.kfpoints_factor = orm.Int(inputs.get('kfpoints_factor', 1))
        builder.qfpoints_distance = orm.Float(inputs.get('qfpoints_distance', 0.5))
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])

        return builder

    def setup(self):
        self.ctx.current_structure = self.inputs.structure


        if (
            self._PW_BANDS_NAMESPACE in self.inputs
            or
            self._BANDS_NAMESPACE in self.inputs
            ):
            from aiida_quantumespresso.calculations.functions.seekpath_structure_analysis import seekpath_structure_analysis

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


        if self._BANDS_NAMESPACE in self.inputs:
            self.ctx.inputs_bands = AttributeDict(
                self.exposed_inputs(
                    EpwBandsWorkChain,
                    namespace=self._BANDS_NAMESPACE
                    )
                )

        if self._A2F_NAMESPACE in self.inputs:
            self.ctx.inputs_a2f = AttributeDict(
                self.exposed_inputs(
                    EpwA2fWorkChain,
                    namespace=self._A2F_NAMESPACE
                    )
                )

        if self._IBTE_NAMESPACE in self.inputs:
            self.ctx.inputs_ibte = AttributeDict(
                self.exposed_inputs(
                    EpwIBTEWorkChain,
                    namespace=self._IBTE_NAMESPACE
                    )
                )

    def validate_inputs(self):
        pass
    def should_run_pw_relax(self):
        """Check if the pw relax workflow should be run.
        If 'pw_relax' is not in the inputs, it will return False.
        """

        return self._PW_RELAX_NAMESPACE in self.inputs

    def run_pw_relax(self):
        """Run the pw relax workflow."""

        inputs = AttributeDict(
            self.exposed_inputs(
                PwRelaxWorkChain,
                namespace=self._PW_RELAX_NAMESPACE
            )
        )

        inputs.metadata.call_link_label = self._PW_RELAX_NAMESPACE
        inputs.structure = self.ctx.current_structure

        workchain_node = self.submit(PwRelaxWorkChain, **inputs)
        self.report(f'launching `PwRelaxWorkChain`<{workchain_node.pk}>')

        return ToContext(workchain_pw_relax=workchain_node)

    def inspect_pw_relax(self):
        """Verify that the pw relax workflow finished successfully."""

        workchain = self.ctx.workchain_pw_relax

        if not workchain.is_finished_ok:
            self.report(f'`PwRelaxWorkChain`<{workchain.pk}> failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_PW_RELAX

        self.report(f'`PwRelaxWorkChain`<{workchain.pk}> finished successfully')
        self.ctx.current_structure = workchain.outputs.output_structure
        self.ctx.nbnd = workchain.outputs.output_parameters.get('number_of_bands')
        self.out_many(
            self.exposed_outputs(
                workchain,
                PwRelaxWorkChain,
                namespace=self._PW_RELAX_NAMESPACE,
            ),
        )

        # Immediately clean the workdir of the pw relax workchain
        self._clean_workdir(workchain)

    def should_run_pw_bands(self):
        """Check if the pw bands workflow should be run.
        If 'pw_bands' is not in the inputs or the 'kpoints_nscf' is not in the context, it will return False.
        """

        return self._PW_BANDS_NAMESPACE in self.inputs

    def run_pw_bands(self):
        """Run the pw bands workflow."""

        inputs = AttributeDict(
            self.exposed_inputs(
                PwBandsWorkChain,
                namespace=self._PW_BANDS_NAMESPACE
                )
        )

        inputs.metadata.call_link_label = self._PW_BANDS_NAMESPACE
        inputs.structure = self.ctx.current_structure
        inputs.bands_kpoints = self.ctx.bands_kpoints

        parameters = inputs.scf.pw.parameters.get_dict()
        if self.ctx.nbnd:
            parameters['SYSTEM']['nbnd'] = self.ctx.nbnd
        inputs.scf.pw.parameters = orm.Dict(parameters)

        workchain_node = self.submit(PwBandsWorkChain, **inputs)
        self.report(f'launching `PwBandsWorkChain`<{workchain_node.pk}>')

        return ToContext(workchain_pw_bands=workchain_node)

    def inspect_pw_bands(self):
        """Verify that the pw bands workflow finished successfully."""

        workchain = self.ctx.workchain_pw_bands

        if not workchain.is_finished_ok:
            self.report(f'`PwBandsWorkChain`<{workchain.pk}> failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_PW_BANDS
        self.report(f'`PwBandsWorkChain`<{workchain.pk}> finished successfully')
        self.out_many(
            self.exposed_outputs(
                workchain,
                PwBandsWorkChain,
                namespace=self._PW_BANDS_NAMESPACE,
            ),
        )

        # Immediately clean the workdir of the pw bands workchain
        self._clean_workdir(workchain)

    def should_run_b2w(self):
        return self._B2W_NAMESPACE in self.inputs

    def run_b2w(self):
        """Run the b2w workflow."""
        inputs = AttributeDict(
            self.exposed_inputs(
                EpwB2WWorkChain,
                namespace=self._B2W_NAMESPACE
            )
        )
        inputs.metadata.call_link_label = self._B2W_NAMESPACE
        inputs.structure = self.ctx.current_structure
        workchain_node = self.submit(EpwB2WWorkChain, **inputs)

        self.report(f'launching EpwB2WWorkChain<{workchain_node.pk}>')

        return ToContext(workchain_b2w=workchain_node)


    def inspect_b2w(self):
        """Inspect the b2w workflow."""
        b2w_workchain = self.ctx.workchain_b2w

        if not b2w_workchain.is_finished_ok:
            self.report(f'`EpwB2WWorkChain`<{b2w_workchain.pk}> failed with exit status {b2w_workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_B2W

        self.report(f'`EpwB2WWorkChain`<{b2w_workchain.pk}> finished successfully')
        self.out_many(
            self.exposed_outputs(
                b2w_workchain,
                EpwB2WWorkChain,
                namespace=self._B2W_NAMESPACE
            )
        )

    def should_run_bands(self):
        """Check if the bands workflow should continue or not."""
        if self._BANDS_NAMESPACE in self.inputs:
            if self.should_run_b2w():
                b2w_workchain = self.ctx.workchain_b2w

                parent_folder_epw = b2w_workchain.outputs.epw_base.remote_folder

                self.ctx.inputs_bands[self._BANDS_NAMESPACE].parent_folder_epw = parent_folder_epw

            return True
        else:
            return False

    def run_bands(self):
        """Run the bands workflow."""
        inputs = self.ctx.inputs_bands
        inputs.structure = self.ctx.current_structure
        inputs.metadata.call_link_label = self._BANDS_NAMESPACE

        workchain_node = self.submit(EpwBandsWorkChain, **inputs)

        self.report(f'launching EpwBandsWorkChain<{workchain_node.pk}>')

        return ToContext(workchain_bands=workchain_node)

    def inspect_bands(self):
        """Inspect the bands workflow."""
        bands_workchain = self.ctx.workchain_bands

        if not bands_workchain.is_finished_ok:
            self.report(f'`EpwBandsWorkChain`<{bands_workchain.pk}> failed with exit status {bands_workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_BANDS
        self.report(f'`EpwBandsWorkChain`<{bands_workchain.pk}> finished successfully')
        self.out_many(
            self.exposed_outputs(
                bands_workchain,
                EpwBandsWorkChain,
                namespace=self._BANDS_NAMESPACE
            )
        )

    def should_run_a2f(self):
        """Check if the a2f workflow should continue or not."""
        if self._A2F_NAMESPACE in self.inputs:
            if self.should_run_b2w():
                b2w_workchain = self.ctx.workchain_b2w
                parent_folder_epw = b2w_workchain.outputs.epw_base.remote_folder
                self.ctx.inputs_a2f[self._A2F_NAMESPACE].parent_folder_epw = parent_folder_epw
            return True
        else:
            return False

    def run_a2f(self):
        """Run the a2f workflow."""
        inputs = self.ctx.inputs_a2f
        inputs.structure = self.ctx.current_structure
        inputs.a2f.qfpoints_distance = self.inputs.qfpoints_distance

        inputs.metadata.call_link_label = self._A2F_NAMESPACE
        workchain_node = self.submit(EpwA2fWorkChain, **inputs)

        self.report(f'launching EpwA2fWorkChain<{workchain_node.pk}>')

        return ToContext(workchain_a2f=workchain_node)

    def inspect_a2f(self):
        """Inspect the a2f workflow."""
        a2f_workchain = self.ctx.workchain_a2f
        if not a2f_workchain.is_finished_ok:
            self.report(f'`EpwA2fWorkChain`<{a2f_workchain.pk}> failed with exit status {a2f_workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_A2F

        self.report(f'`EpwA2fWorkChain`<{a2f_workchain.pk}> finished successfully')
        self.out_many(
            self.exposed_outputs(
                a2f_workchain,
                EpwA2fWorkChain,
                namespace=self._A2F_NAMESPACE
            )
        )

    def should_run_ibte(self):
        """Check if the ibte workflow should continue or not."""
        if self._IBTE_NAMESPACE in self.inputs:
            if self.should_run_b2w():
                b2w_workchain = self.ctx.workchain_b2w
                parent_folder_epw = b2w_workchain.outputs.epw_base.remote_folder
                self.ctx.inputs_ibte[self._IBTE_NAMESPACE].parent_folder_epw = parent_folder_epw
            return True
        else:
            return False

    def run_ibte(self):
        """Run the ibte workflow."""
        inputs = self.ctx.inputs_ibte
        inputs.structure = self.ctx.current_structure
        # TODO: We should then use the input fine grid from a2f workchain for full consistency.
        inputs.ibte.qfpoints_distance = self.inputs.qfpoints_distance
        inputs.metadata.call_link_label = self._IBTE_NAMESPACE
        workchain_node = self.submit(EpwIBTEWorkChain, **inputs)

        self.report(f'launching EpwIBTEWorkChain<{workchain_node.pk}>')

        return ToContext(workchain_ibte=workchain_node)

    def inspect_ibte(self):
        """Inspect the ibte workflow."""
        ibte_workchain = self.ctx.workchain_ibte
        if not ibte_workchain.is_finished_ok:
            self.report(f'`EpwIBTEWorkChain`<{ibte_workchain.pk}> failed with exit status {ibte_workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_IBTE

        self.report(f'`EpwIBTEWorkChain`<{ibte_workchain.pk}> finished successfully')
        self.out_many(
            self.exposed_outputs(
                ibte_workchain,
                EpwIBTEWorkChain,
                namespace=self._IBTE_NAMESPACE
            )
        )

    def results(self):
        pass

    @staticmethod
    def _clean_workdir(node):
        """Clean the working directories of all child calculations if `clean_workdir=True` in the inputs."""

        cleaned_calcs = []

        for called_descendant in node.called_descendants:
            if isinstance(called_descendant, orm.CalcJobNode):
                try:
                    called_descendant.outputs.remote_folder._clean()  # pylint: disable=protected-access
                    cleaned_calcs.append(called_descendant.pk)
                except (IOError, OSError, KeyError, NotExistentAttributeError):
                    pass

        return cleaned_calcs

    def on_terminated(self):
        super().on_terminated()

        if self.inputs.clean_workdir.value is False:
            self.report('remote folders will not be cleaned')
            return

        cleaned_calcs = self._clean_workdir(self.node)

        if cleaned_calcs:
            self.report(f"cleaned remote folders of calculations: {' '.join(map(str, cleaned_calcs))}")

