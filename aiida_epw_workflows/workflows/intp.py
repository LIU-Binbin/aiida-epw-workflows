# -*- coding: utf-8 -*-
from aiida import orm
from aiida.engine import WorkChain, ToContext, if_

from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

from .base import EpwBaseWorkChain
from .b2w import EpwB2WWorkChain
import warnings

from ..common.types import RestartType

class EpwBaseIntpWorkChain(ProtocolMixin, WorkChain):
    """Base work chain for two-step interpolation workflows.
    It will run the `EpwB2WWorkChain` for electron-phonon coupling matrix on Wannier representation.
    And then an `EpwBaseWorkChain` for interpolation from Wannier to Bloch representation (fine grid).
    Based on the fine-grid electron-phonon coupling matrix, various quantities can be computed.
    """

    # --- Child classes should override these placeholders ---
    _B2W_NAMESPACE = EpwB2WWorkChain._NAMESPACE
    _INTP_NAMESPACE = 'intp'  # e.g., 'a2f' for A2fWorkChain
    # ---------------------------------------------------------

    # --- INTP workchain and B2W workchain share these blocked keywords ---
    _blocked_keywords = [
        ('INPUTEPW', 'use_ws'),
        ('INPUTEPW', 'muc'),
        ('INPUTEPW', 'nbndsub'),
        ('INPUTEPW', 'bands_skipped'),
        # ('INPUTEPW', 'degaussq'),
        ('INPUTEPW', 'vme'),
    ]
    # ---------------------------------------------------------

    # --- Child classes should override this _forced_parameters according to the purpose of the child workchain ---

    _forced_parameters = {
        'INPUTEPW': {
            'elph'        : True,
            'ep_coupling' : True,
            'epbread'     : False,
            'epbwrite'    : False,
            'epwread'     : True,
            'epwwrite'    : False,
            'mp_mesh_k'   : True,
            'wannierize'  : False,
        }
    }
    # ---------------------------------------------------------

    _restart_from_ephmat = {
        'INPUTEPW': {
            'elph'        : False,
            'ep_coupling' : False,
            'ephwrite'    : False,
            'restart'     : True,
        }
    }

    @classmethod
    def define(cls, spec):
        """Define the work chain specification."""
        super().define(spec)

        # TODO: Seems we don't need structure input port here
        spec.input('structure', valid_type=orm.StructureData)
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(False))
        spec.input('restart_type', valid_type=orm.EnumData, default=lambda: orm.EnumData(RestartType.FROM_SCRATCH))

        spec.expose_inputs(
            EpwB2WWorkChain, namespace=cls._B2W_NAMESPACE, exclude=(
                'structure',
                'clean_workdir',
            ),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the interpolation `EpwCalculation`s.'
            }
        )

        # spec.inputs[cls._B2W_NAMESPACE].validator = None
        spec.expose_inputs(
            EpwBaseWorkChain, namespace=cls._INTP_NAMESPACE, exclude=(
                'structure',
                'clean_workdir',
                'parent_folder_nscf',
                'parent_folder_ph',
                'parent_folder_chk',
                # 'parent_folder_epw',
            ),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the a2f `EpwBaseWorkChain`.'
            }
        )
        spec.outline(
            cls.setup,
            if_(cls.should_run_b2w)(
                cls.run_b2w,
                cls.inspect_b2w,
            ),
            cls.prepare_process,
            cls.run_process,
            cls.inspect_process,
            cls.results
        )
        spec.expose_outputs(
            EpwB2WWorkChain,
            namespace=cls._B2W_NAMESPACE,
            namespace_options={
                'required': False,
                'help': 'Outputs for the `EpwB2WWorkChain`.'
            }
        )
        spec.expose_outputs(
            EpwBaseWorkChain,
            namespace=cls._INTP_NAMESPACE,
            exclude=(
                'output_parameters',
                'remote_folder',
                'retrieved',
            )
        )

        spec.output('output_parameters', valid_type=orm.Dict)
        spec.output('remote_folder', valid_type=orm.RemoteData)
        spec.output('retrieved', valid_type=orm.FolderData)

    @staticmethod
    def get_descendant(
        intp: orm.WorkChainNode,
        link_label_filter: str
        ) -> orm.WorkChainNode:
        """Get the descendant workchains of the intp workchain.
        :param intp: The intp workchain.
        :param link_label_filter: The link label filter.
        :return: The descendant workchain.
        """
        try:
            return intp.base.links.get_outgoing(
                link_label_filter=link_label_filter
                ).first().node
        except AttributeError:
            return None

    @classmethod
    def get_protocol_overrides(cls) -> dict:
        """Get the ``overrides`` for default protocol.
        :return: The overrides.
        """
        from importlib_resources import files
        import yaml

        from . import protocols

        path = files(protocols) / f"{cls._INTP_NAMESPACE}.yaml"
        with path.open() as file:
            return yaml.safe_load(file)

    @classmethod
    def _get_builder_restart(
        cls,
        from_intp_workchain=None,
        ):
        """Return a builder prepopulated with inputs extracted from the intp workchain.
        This is a hook method for the `get_builder_restart` method of derived classes.
        :param from_intp_workchain: The intp workchain.
        :return: The builder.
        """
        builder = from_intp_workchain.get_builder_restart()
        # parent_builder = from_intp_workchain.get_builder_restart()

        b2w = EpwBaseIntpWorkChain.get_descendant(
            from_intp_workchain,
            cls._B2W_NAMESPACE
            )

        if (
            cls._B2W_NAMESPACE not in from_intp_workchain.inputs
            or
            b2w.is_finished_ok
            ):
            print("EpwB2WWorkChain subprocess is finished or not present, skipping")
            builder[cls._INTP_NAMESPACE].parent_folder_epw = b2w.outputs.epw_base.remote_stash
            builder.pop(cls._B2W_NAMESPACE)
        else:
            print("EpwB2WWorkChain subprocess is not finished, restarting from it")
            b2w_builder = EpwB2WWorkChain.get_builder_restart(
                from_b2w_workchain=b2w
                )
            builder[cls._B2W_NAMESPACE]._data = b2w_builder._data

        intp = EpwBaseIntpWorkChain.get_descendant(
            from_intp_workchain,
            cls._INTP_NAMESPACE
            )

        if intp:
            if intp.is_finished_ok:
                warnings.warn(
                    f"The EpwBaseWorkChain subprocess <{from_intp_workchain.pk}> has already finished.",
                    stacklevel=2
                    )
                return
            else:
                print("EpwBaseWorkChain subprocess is finished, setting the parent folder")
                builder[cls._INTP_NAMESPACE].parent_folder_epw = intp.inputs.parent_folder_epw

        return builder

    @classmethod
    def get_builder_restart_from_b2w(
        cls,
        from_b2w_workchain: orm.WorkChainNode,
        protocol=None,
        overrides=None,
        **kwargs
        ):
        """Return a builder prepopulated with inputs extracted from the b2w workchain.
        :param from_b2w_workchain: The b2w workchain.
        :param protocol: The protocol.
        :param overrides: The overrides.
        :return: The builder.
        """

        inputs = cls.get_protocol_inputs(protocol, overrides)

        builder = cls.get_builder()

        if not from_b2w_workchain or not from_b2w_workchain.process_class == EpwB2WWorkChain:
            raise ValueError('Currently we only accept `EpwB2WWorkChain`')

        structure = from_b2w_workchain.inputs.structure
        code = from_b2w_workchain.inputs[EpwB2WWorkChain._EPW_NAMESPACE]['epw'].code

        builder.structure = structure

        if from_b2w_workchain.is_finished_ok:
            builder.pop(cls._B2W_NAMESPACE)
        else:
            b2w_builder = EpwB2WWorkChain.get_builder_restart(
                from_b2w_workchain=from_b2w_workchain,
                protocol=protocol,
                overrides=overrides.get(cls._B2W_NAMESPACE, None),
                **kwargs
                )

            # Actually there is no exclusion of EpwB2WWorkChain namespace
            # So we need to set the _data manually
            builder[cls._B2W_NAMESPACE]._data = b2w_builder._data

        intp_builder = EpwBaseWorkChain.get_builder_from_protocol(
            code=code,
            structure=structure,
            protocol=protocol,
            overrides=inputs.get(cls._INTP_NAMESPACE, None),
            **kwargs
            )

        intp_builder.parent_folder_epw = from_b2w_workchain.outputs.epw.remote_folder

        builder[cls._INTP_NAMESPACE]._data = intp_builder._data

        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])

        return builder

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
        builder = cls.get_builder()
        builder.structure = structure

        b2w_builder = EpwB2WWorkChain.get_builder_from_protocol(
            codes=codes,
            structure=structure,
            protocol=protocol,
            overrides=inputs.get(cls._B2W_NAMESPACE, {}),
            **kwargs
        )

        # b2w_builder.w90_intp.pop('open_grid')
        # b2w_builder.w90_intp.pop('projwfc')

        builder[cls._B2W_NAMESPACE]._data = b2w_builder._data

        intp_builder = EpwBaseWorkChain.get_builder_from_protocol(
            code=codes['epw'],
            structure=structure,
            protocol=protocol,
            overrides=inputs.get(cls._INTP_NAMESPACE, None),
        )

        builder[cls._INTP_NAMESPACE]._data = intp_builder._data

        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])

        return builder

    def setup(self):
        """Setup steps, i.e. initialise context variables."""
        self.ctx.degaussq = None
        inputs = self.exposed_inputs(EpwBaseWorkChain, namespace=self._INTP_NAMESPACE)

        inputs.structure = self.inputs.structure

        self.ctx.inputs = inputs

    def should_run_b2w(self):
        """Check if the epw loop should continue or not.
        If 'intp' is not in the inputs, it will return False.
        """
        return self._B2W_NAMESPACE in self.inputs

    def run_b2w(self):
        """Run the ``restart`` EPW calculation for the current interpolation distance."""

        inputs = self.exposed_inputs(EpwB2WWorkChain, namespace=self._B2W_NAMESPACE)
        inputs.structure = self.inputs.structure

        inputs.metadata.call_link_label = self._B2W_NAMESPACE
        workchain_node = self.submit(EpwB2WWorkChain, **inputs)

        self.report(f'launching `EpwB2WWorkChain`<{workchain_node.pk}>')

        return ToContext(workchain_b2w=workchain_node)

    def inspect_b2w(self):
        """Verify that the epw.x workflow finished successfully.
        If the epw workflow passed, it will generate the parent folder for the following EpwBaseWorkChain.
        It will expose the outputs of the EpwB2WWorkChain as an intermediate results.
        """
        b2w_workchain = self.ctx.workchain_b2w

        if not b2w_workchain.is_finished_ok:
            self.report(f'`EpwB2WWorkChain`<{b2w_workchain.pk}> failed with exit status {b2w_workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_B2W

        self.report(f'`EpwB2WWorkChain`<{b2w_workchain.pk}> finished successfully')

        # We set the parent folder here to keep the logic of restart from EpwB2WWorkChain
        # Since the only connection between the EpwBaseIntpWorkChain and EpwB2WWorkChain is the parent_folder_epw
        # And the parameter of EpwB2WWorkChain is deduced from the parent_folder_epw
        self.ctx.inputs.parent_folder_epw = b2w_workchain.outputs.epw_base.remote_stash

        self.out_many(
            self.exposed_outputs(
                b2w_workchain,
                EpwB2WWorkChain,
                namespace=self._B2W_NAMESPACE
                )
        )

    def prepare_process(self):
        """Prepare for the `EpwBaseWorkChain` workchain.
        It will update the parameters of the EpwBaseWorkChain with the parameters of the previous EpwB2WWorkChain.
        """

        parameters = self.ctx.inputs.epw.parameters.get_dict()

        b2w_parameters = self.ctx.inputs.parent_folder_epw.creator.inputs.parameters.get_dict()

        for namespace, keyword in self._blocked_keywords:
            if keyword in b2w_parameters[namespace]:
                parameters[namespace][keyword] = b2w_parameters[namespace][keyword]

        for namespace, keywords in self._forced_parameters.items():
            for keyword, value in keywords.items():
                parameters[namespace][keyword] = value

        if self.inputs.restart_type == RestartType.FROM_EPHMAT:
            for namespace, keywords in self._restart_from_ephmat.items():
                for keyword, value in keywords.items():
                    parameters[namespace][keyword] = value

        self.ctx.inputs.epw.parameters = orm.Dict(parameters)

    def run_process(self):
        """Run the `EpwBaseWorkChain`
        """
        inputs = self.ctx.inputs

        inputs.metadata.call_link_label = self._INTP_NAMESPACE
        inputs.structure = self.inputs.structure
        
        workchain_node = self.submit(EpwBaseWorkChain, **inputs)

        self.report(f'launching EpwBaseWorkChain<{workchain_node.pk}> in {self._INTP_NAMESPACE} mode')

        return ToContext(workchain_intp=workchain_node)

    def results(self):
        """It will expose the basic outputs of the `EpwBaseWorkChain` as a final results.
        """

        self.out_many(
            self.exposed_outputs(
                self.ctx.workchain_intp,
                EpwBaseWorkChain,
                namespace=self._INTP_NAMESPACE
                )
        )

        self.out('output_parameters', self.ctx.workchain_intp.outputs.output_parameters)
        self.out('remote_folder', self.ctx.workchain_intp.outputs.remote_folder)
        self.out('retrieved', self.ctx.workchain_intp.outputs.retrieved)

    def on_terminated(self):
        """Clean up the work chain."""
        super().on_terminated()

        if self.inputs.clean_workdir.value is False:
            self.report('remote folders will not be cleaned')
            return

        cleaned_calcs = []

        for called_descendant in self.node.called_descendants:
            if isinstance(called_descendant, orm.CalcJobNode):
                try:
                    called_descendant.outputs.remote_folder._clean()  # pylint: disable=protected-access
                    cleaned_calcs.append(called_descendant.pk)
                except (IOError, OSError, KeyError):
                    pass

        if cleaned_calcs:
            self.report(f"cleaned remote folders of calculations: {' '.join(map(str, cleaned_calcs))}")
