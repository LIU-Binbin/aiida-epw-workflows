# -*- coding: utf-8 -*-
from aiida import orm
from aiida.common import AttributeDict, LinkType
from aiida.engine import WorkChain, ToContext, if_, while_, append_

from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin
from aiida.engine import calcfunction

from .b2w import EpwB2WWorkChain
from .bands import EpwBandsWorkChain
from .a2f import EpwA2fWorkChain
from .iso import EpwIsoWorkChain
from .aniso import EpwAnisoWorkChain
from .base import EpwBaseWorkChain

from ..common.types import RestartType

@calcfunction
def split_list(list_node: orm.List) -> dict:
    return {f'el_{no}': orm.Float(el) for no, el in enumerate(list_node.get_list())}

class EpwSuperConWorkChain(ProtocolMixin, WorkChain):
    """Work chain to compute superconductivity based on different levels of approximations.
    It will run the  `EpwBandsWorkChain`, `EpwA2fWorkChain`, `EpwIsoWorkChain`, and `EpwAnisoWorkChain` consecutively.
    """

    _NAMESPACE       = 'supercon'

    _CONV_NAMESPACE  = 'a2f_conv'

    _B2W_NAMESPACE   = EpwB2WWorkChain._NAMESPACE
    _BANDS_NAMESPACE = EpwBandsWorkChain._INTP_NAMESPACE
    _A2F_NAMESPACE   = EpwA2fWorkChain._INTP_NAMESPACE
    _ISO_NAMESPACE   = EpwIsoWorkChain._INTP_NAMESPACE
    _ANISO_NAMESPACE = EpwAnisoWorkChain._INTP_NAMESPACE

    _MIN_FREQ = -1.0 # meV ~ 8.1 cm-1

    @classmethod
    def define(cls, spec):
        """Define the work chain specification."""
        super().define(spec)

        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(True))
        spec.input('structure', valid_type=orm.StructureData)
        spec.input('interpolation_distances', required=False, valid_type=orm.List)
        spec.input('kfpoints_factor', required=False, valid_type=orm.Int, default=lambda: orm.Int(1))
        spec.input('convergence_threshold', required=False, valid_type=orm.Float)
        spec.input('always_run_final', required=False, valid_type=orm.Bool, default=lambda: orm.Bool(True))
        spec.input('use_ir', required=False, valid_type=orm.Bool, default=lambda: orm.Bool(False))

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
        # Here A2f and A2f_conv are exclusive, they share the same namespace
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
            EpwIsoWorkChain,
            namespace=cls._ISO_NAMESPACE,
            exclude=(
                'clean_workdir',
                'structure',
                f"{cls._ISO_NAMESPACE}.parent_folder_nscf",
                f"{cls._ISO_NAMESPACE}.parent_folder_chk",
                f"{cls._ISO_NAMESPACE}.parent_folder_ph",
            ),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the `EpwIsoWorkChain`.'
            }
        )
        spec.expose_inputs(
            EpwAnisoWorkChain,
            namespace=cls._ANISO_NAMESPACE,
            exclude=(
                'clean_workdir',
                'structure',
                f"{cls._ANISO_NAMESPACE}.parent_folder_nscf",
                f"{cls._ANISO_NAMESPACE}.parent_folder_chk",
                f"{cls._ANISO_NAMESPACE}.parent_folder_ph",
            ),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the `EpwAnisoWorkChain`.'
            }
        )
        spec.outline(
            cls.setup,
            cls.validate_inputs,
            if_(cls.should_run_b2w)(
                cls.run_b2w,
                cls.inspect_b2w,
            ),
            if_(cls.should_run_bands)(
                cls.run_bands,
                cls.inspect_bands,
            ),
            cls.prepare_intp,
            while_(cls.should_run_conv)(
                cls.run_conv,
                cls.inspect_conv,
            ),
            if_(cls.should_run_a2f)(
                cls.run_a2f,
                cls.inspect_a2f,
            ),
            if_(cls.should_run_iso)(
                cls.run_iso,
                cls.inspect_iso,
            ),
            if_(cls.should_run_aniso)(
                cls.run_aniso,
                cls.inspect_aniso,
            ),
            cls.results
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
            EpwIsoWorkChain,
            namespace=cls._ISO_NAMESPACE,
            exclude=('dos', 'phdos', 'phdos_proj', 'a2f', 'a2f_proj', 'lambda_FS', 'lambda_k_pairs'),
            namespace_options={
                'required': False,
                'help': 'Outputs from the `EpwIsoWorkChain`.'
            }
        )

        spec.expose_outputs(
            EpwAnisoWorkChain,
            namespace=cls._ANISO_NAMESPACE,
            exclude=('dos', 'phdos', 'phdos_proj', 'a2f', 'a2f_proj', 'lambda_FS', 'lambda_k_pairs'),
            namespace_options={
                'required': False,
                'help': 'Outputs from the `EpwAnisoWorkChain`.'
            }
        )

        spec.exit_code(401, 'ERROR_SUB_PROCESS_B2W',
            message='The `b2w` sub process failed')
        spec.exit_code(402, 'ERROR_CONVERGENCE_NOT_REACHED',
            message='The convergence is not reached in current interpolation list.')
        spec.exit_code(403, 'ERROR_SUB_PROCESS_BANDS',
            message='The `bands` sub process failed')
        spec.exit_code(404, 'ERROR_IMAGINARY_PHONON_FREQ',
            message='The phonon frequency is too low.')
        spec.exit_code(405, 'ERROR_SUB_PROCESS_A2F',
            message='The `a2f` sub process failed')
        spec.exit_code(406, 'ERROR_ALLEN_DYNES_TC_TOO_LOW',
            message='The Allen-Dynes Tc is too low.')
        spec.exit_code(407, 'ERROR_SUB_PROCESS_ISO',
            message='The `iso` sub process failed')
        spec.exit_code(408, 'ERROR_ISOTROPIC_TC_TOO_LOW',
            message='The isotropic Tc is too low.')
        spec.exit_code(409, 'ERROR_SUB_PROCESS_ANISO',
            message='The `aniso` sub process failed')

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
    def get_builder_restart_from_b2w(
        cls,
        from_b2w_workchain: orm.WorkChainNode,
        protocol=None,
        overrides=None,
        **kwargs
        ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.
        It will restart from the b2w workchain and then run the a2f, iso, and aniso workchains.
        """
        inputs = cls.get_protocol_inputs(protocol, overrides)

        builder = cls.get_builder()

        if not from_b2w_workchain or not from_b2w_workchain.process_class == EpwB2WWorkChain:
            raise ValueError('Currently we only accept `EpwB2WWorkChain`')

        if from_b2w_workchain.is_finished_ok:
            builder.pop(EpwA2fWorkChain._B2W_NAMESPACE)
            parent_folder_epw = from_b2w_workchain.outputs.epw.remote_stash
        else:
            b2w_builder = EpwB2WWorkChain.get_builder_restart(
                from_b2w_workchain=from_b2w_workchain,
                protocol=protocol,
                overrides=overrides.get(EpwA2fWorkChain._B2W_NAMESPACE, None),
                **kwargs
                )

            # Actually there is no exclusion of EpwB2WWorkChain namespace
            # So we need to set the _data manually
            builder[EpwA2fWorkChain._B2W_NAMESPACE]._data = b2w_builder._data


        for (epw_namespace, epw_workchain_class) in (
            (cls._BANDS_NAMESPACE, EpwBandsWorkChain),
            (cls._A2F_NAMESPACE, EpwA2fWorkChain),
            (cls._ISO_NAMESPACE, EpwIsoWorkChain),
            (cls._ANISO_NAMESPACE, EpwAnisoWorkChain),
        ):
            epw_builder = epw_workchain_class.get_builder_restart_from_b2w(
                from_b2w_workchain=from_b2w_workchain,
                protocol=protocol,
                overrides=overrides.get(epw_namespace, None),
                **kwargs
                )

            if epw_workchain_class._B2W_NAMESPACE in epw_builder:
                epw_builder.pop(epw_workchain_class._B2W_NAMESPACE)

            # if parent_folder_epw:
            #     epw_builder[epw_namespace].parent_folder_epw = parent_folder_epw

            builder[epw_namespace]._data = epw_builder._data


        builder.structure = from_b2w_workchain.inputs.structure
        builder.interpolation_distances = orm.List(inputs.get('interpolation_distances', None))
        builder.kfpoints_factor = orm.Int(inputs.get('kfpoints_factor', 1))
        builder.convergence_threshold = orm.Float(inputs['convergence_threshold'])
        builder.always_run_final = orm.Bool(inputs.get('always_run_final', True))
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])

        return builder

    @classmethod
    def get_builder_from_a2f(
        cls,
        from_a2f_workchain: orm.WorkChainNode,
        protocol=None,
        overrides=None,
        **kwargs
        ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol."""

        builder = cls.get_builder()

        if not from_a2f_workchain or not from_a2f_workchain.process_class == EpwA2fWorkChain:
            raise ValueError('Currently we only accept `EpwA2fWorkChain`')

        structure = from_a2f_workchain.inputs.structure
        code = from_a2f_workchain.inputs[EpwA2fWorkChain._INTP_NAMESPACE].epw.code

        parent_folder_epw = from_a2f_workchain.outputs.epw.remote_folder

        for (epw_namespace, epw_workchain_class) in (
            (cls._ISO_NAMESPACE, EpwIsoWorkChain),
            (cls._ANISO_NAMESPACE, EpwAnisoWorkChain),
        ):
            epw_builder = epw_workchain_class.get_builder()

            intp_builder = EpwBaseWorkChain.get_builder_from_protocol(
                code,
                structure,
                protocol,
                overrides=overrides.get(epw_namespace, None),
                **kwargs)

            epw_builder[epw_namespace]._data = intp_builder._data

            epw_builder[epw_namespace].parent_folder_epw = parent_folder_epw

            builder[epw_namespace]._data = epw_builder._data

        builder.pop('interpolation_distances')
        builder.pop('convergence_threshold')
        builder.pop('always_run_final')

        return builder

    @classmethod
    def get_builder_restart(
        cls,
        from_supercon_workchain: orm.WorkChainNode,
        ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol."""

        ## Here the restart builder contains info:
        ## all of the metadatas of the calculations.
        ## the parent_folders should be reset according to the progress of the previous workchain.

        builder = from_supercon_workchain.get_builder_restart()

        if not from_supercon_workchain or not from_supercon_workchain.process_class == EpwSuperConWorkChain:
            raise ValueError('Currently we only accept `EpwSuperConWorkChain`')

        # if from_supercon_workchain.is_finished_ok:
        #     raise Warning('The `EpwSuperConWorkChain` is already finished.')

        from ..tools.links import get_descendants
        from aiida.common.links import LinkType

        descendants = get_descendants(
            from_supercon_workchain,
            LinkType.CALL_WORK
            )

        try:
            for sub_workchain in (
                EpwBandsWorkChain,
                EpwA2fWorkChain,
                EpwIsoWorkChain,
                EpwAnisoWorkChain,
                ):
                builder[sub_workchain._INTP_NAMESPACE].pop(sub_workchain._B2W_NAMESPACE)
        except KeyError:
            pass

        # Firstly we should check whether we should restart from b2w workchain.
        # If B2W_NAMESPACE is in the inputs, it means that the previous workchain did a b2w calculation.
        # So we need to check the status of the b2w workchain.
        if cls._B2W_NAMESPACE in from_supercon_workchain.inputs:
            b2w_workchain = descendants[cls._B2W_NAMESPACE][0]
                # If the b2w workchain is finished, simply pop it.
            if b2w_workchain.is_finished_ok:
                builder.pop(cls._B2W_NAMESPACE)
                builder[cls._BANDS_NAMESPACE][cls._BANDS_NAMESPACE]['parent_folder_epw'] = b2w_workchain.outputs.epw.remote_stash
                builder[cls._A2F_NAMESPACE][cls._A2F_NAMESPACE]['parent_folder_epw'] = b2w_workchain.outputs.epw.remote_stash

            # If the b2w workchain is not finished, we only need to restart from b2w workchain leaving the following inputs unchanged.
            else:
                b2w_builder = EpwB2WWorkChain.get_builder_restart(
                    from_b2w_workchain=b2w_workchain,
                    )

                builder[cls._B2W_NAMESPACE]._data = b2w_builder._data

                return builder

        # If B2W_NAMESPACE is not in the inputs, it means that the previous workchain did not do a b2w calculation.
        # So the previous workchain itself is a restart one and b2w is finished. We can pop the b2w namespace.
        else:
            print('b2w namespace is not in the inputs')
            builder.pop(cls._B2W_NAMESPACE)

        if cls._BANDS_NAMESPACE in from_supercon_workchain.inputs:
            bands_workchain = descendants[cls._BANDS_NAMESPACE][0]
            if bands_workchain.is_finished_ok:
                builder.pop(cls._BANDS_NAMESPACE)
            else:
                # bands_builder = EpwBandsWorkChain.get_builder_restart(
                #     from_bands_workchain=bands_workchain,
                # )
                # builder[cls._BANDS_NAMESPACE]._data = bands_builder._data
                # builder[cls._BANDS_NAMESPACE][cls._BANDS_NAMESPACE]['parent_folder_epw'] = bands_workchain.inputs.epw.remote_folder
                return builder
        else:
            builder.pop(cls._BANDS_NAMESPACE)
            print('bands namespace is not in the inputs')
        # If interpolation list is not an input port, it must be that the previous
        # workchain has finished convergence test. Or it start from a given grid without convergence test.

        if 'interpolation_distances' in from_supercon_workchain.inputs:
            initial_interpolation_distances = from_supercon_workchain.inputs.interpolation_distances.get_list()
            # Then we need to know whether convergence is finished.
            a2f_conv_workchains = descendants[cls._CONV_NAMESPACE]

            is_converged, _ = cls.check_convergence(
                a2f_conv_workchains,
                from_supercon_workchain.inputs.convergence_threshold
            )

            for a2f_conv_workchain in a2f_conv_workchains:
                if a2f_conv_workchain.is_finished_ok:
                    initial_interpolation_distances.remove(a2f_conv_workchain.inputs.a2f.qfpoints_distance.value)

            if len(initial_interpolation_distances) > 0:
                builder[cls._A2F_NAMESPACE][cls._A2F_NAMESPACE]['parent_folder_epw'] = a2f_conv_workchains[0].inputs[cls._A2F_NAMESPACE].parent_folder_epw
                builder['interpolation_distances'] = orm.List(initial_interpolation_distances)
                builder['convergence_threshold'] = orm.Float(from_supercon_workchain.inputs.convergence_threshold)
                builder['always_run_final'] = orm.Bool(from_supercon_workchain.inputs.always_run_final)
                return builder

            else:
                always_run_final = from_supercon_workchain.inputs.always_run_final

                if always_run_final:
                    builder[cls._ISO_NAMESPACE][cls._ISO_NAMESPACE]['parent_folder_epw'] = a2f_conv_workchains[-1].outputs.remote_folder

                    builder.pop(cls._A2F_NAMESPACE)
                    builder.pop('interpolation_distances')
                    builder.pop('convergence_threshold')
                    builder.pop('always_run_final')
                else:
                    raise ValueError('Convergence can not reach, either add more interpolation distances or set `always_run_final` to True.')
        # If there is no interpolation distance, either it's finished or we don't do it at all.

        else:
            if cls._A2F_NAMESPACE in from_supercon_workchain.inputs:
                a2f_workchain = descendants[cls._A2F_NAMESPACE][0]
                if a2f_workchain and a2f_workchain.is_finished_ok:
                    builder.pop(cls._A2F_NAMESPACE)
                else:
                    # a2f_builder = EpwA2fWorkChain.get_builder_restart(
                    #     from_a2f_workchain=a2f_workchain,
                    #     )
                    # builder[cls._A2F_NAMESPACE]._data = a2f_builder._data
                    return builder
            else:
                builder.pop(cls._A2F_NAMESPACE)

        if cls._ISO_NAMESPACE in from_supercon_workchain.inputs:
            if not cls._ISO_NAMESPACE in descendants:
                raise Warning(
                    '`iso` in input namespace but no workchain found.'
                    'Something in the workchain should be fixed.'
                    )

            iso_workchain = descendants[cls._ISO_NAMESPACE][0]
            if iso_workchain.is_finished_ok:
                builder.pop(cls._ISO_NAMESPACE)
            else:
                # iso_builder = EpwIsoWorkChain.get_builder_restart(
                #     from_iso_workchain=iso_workchain,
                # )
                # builder[cls._ISO_NAMESPACE]._data = iso_builder._data
                return builder

        else:
            builder.pop(cls._ISO_NAMESPACE)

        if cls._ANISO_NAMESPACE in from_supercon_workchain.inputs:
            if not cls._ANISO_NAMESPACE in descendants:
                raise Warning(
                    '`aniso` in input namespace but no workchain found.'
                    'Something in the workchain should be fixed.'
                    )

            aniso_workchain = descendants[cls._ANISO_NAMESPACE][0]
            if aniso_workchain.is_finished_ok:
                print('The `EpwSuperConWorkChain` has already finished.')
                builder[cls._ANISO_NAMESPACE][cls._ANISO_NAMESPACE].parent_folder_epw = aniso_workchain.inputs[cls._ANISO_NAMESPACE].parent_folder_epw
                return builder
            else:
                # aniso_builder = EpwAnisoWorkChain.get_builder_restart(
                #     from_aniso_workchain=aniso_workchain,
                #     )
                # builder[cls._ANISO_NAMESPACE]._data = aniso_builder._data
                return builder
        else:
            raise Warning('The `EpwSuperConWorkChain` has already finished.')

    @classmethod
    def get_builder_restart_from_ph(
            cls,
            from_ph_workchain: orm.WorkChainNode,
            codes,
            protocol=None,
            overrides=None,
            **kwargs
        ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :TODO:
        """

        structure = from_ph_workchain.inputs.ph.parent_folder.creator.inputs.structure

        inputs = cls.get_protocol_inputs(protocol, overrides)

        b2w_builder = EpwB2WWorkChain.get_builder_restart_from_phonon(
            from_ph_workchain=from_ph_workchain,
            codes=codes,
            protocol=protocol,
            overrides=inputs.get(EpwA2fWorkChain._B2W_NAMESPACE, None),
            **kwargs
        )

        # b2w_builder.w90_intp.pop('open_grid')
        # b2w_builder.w90_intp.pop('projwfc')

        builder = cls.get_builder_from_protocol(
            codes=codes,
            structure=structure,
            protocol=protocol,
            overrides=inputs,
            **kwargs
        )

        builder[EpwA2fWorkChain._B2W_NAMESPACE]._data = b2w_builder._data

        return builder

    @classmethod
    def get_builder_restart_from_epw_stash(
            cls,
            from_epw_stash_folder,
            protocol=None,
            overrides=None,
            **kwargs
        ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :TODO:
        """

        if not isinstance(from_epw_stash_folder, (orm.RemoteData, orm.RemoteStashFolderData)):
            raise ValueError('The `from_epw_stash_folder` must be a `RemoteData` or `RemoteStashFolderData` node.')

        structure = from_epw_stash_folder.creator.caller.inputs.structure
        code = from_epw_stash_folder.creator.inputs.code

        args = (code, structure, protocol)

        inputs = cls.get_protocol_inputs(protocol, overrides)


        builder = cls.get_builder()

        builder.pop(cls._B2W_NAMESPACE)

        from aiida_quantumespresso.workflows.protocols.utils import recursive_merge

        for epw_class in (
            EpwA2fWorkChain,
            EpwIsoWorkChain,
            EpwAnisoWorkChain,
        ):
            epw_namespace = epw_class._INTP_NAMESPACE

            overrides_intp = recursive_merge(
                epw_class.get_protocol_overrides().get('default_inputs').get(epw_namespace, None),
                overrides.get(epw_namespace, None).get(epw_namespace, None)
            )
            epw_builder = EpwBaseWorkChain.get_builder_from_protocol(
                *args,
                overrides=overrides_intp,
                **kwargs
            )

            builder[epw_namespace].pop(EpwA2fWorkChain._B2W_NAMESPACE)
            builder[epw_namespace].structure = structure
            builder[epw_namespace][epw_namespace]._data = epw_builder._data

        builder[cls._A2F_NAMESPACE][cls._A2F_NAMESPACE].parent_folder_epw = from_epw_stash_folder

        builder.interpolation_distances = orm.List(inputs.get('interpolation_distances', None))
        builder.convergence_threshold = orm.Float(inputs['convergence_threshold'])
        builder.always_run_final = orm.Bool(inputs.get('always_run_final', True))
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])

        return builder

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

        b2w_builder = EpwB2WWorkChain.get_builder_from_protocol(
            *args,
            overrides=inputs.get(cls._B2W_NAMESPACE, None),
            **kwargs
        )

        builder[cls._B2W_NAMESPACE]._data = b2w_builder._data

        for (epw_namespace, epw_workchain_class) in (
            (cls._BANDS_NAMESPACE, EpwBandsWorkChain),
            (cls._A2F_NAMESPACE, EpwA2fWorkChain),
            (cls._ISO_NAMESPACE, EpwIsoWorkChain),
            (cls._ANISO_NAMESPACE, EpwAnisoWorkChain),
        ):
            epw_builder = epw_workchain_class.get_builder_from_protocol(
                *args,
                overrides=inputs.get(epw_namespace, None),
                **kwargs
            )

            epw_builder.pop(epw_workchain_class._B2W_NAMESPACE)

            builder[epw_namespace]._data = epw_builder._data

        builder.structure = structure
        builder.interpolation_distances = orm.List(inputs.get('interpolation_distances', None))
        builder.kfpoints_factor = orm.Int(inputs.get('kfpoints_factor', 1))
        builder.convergence_threshold = orm.Float(inputs['convergence_threshold'])
        builder.always_run_final = orm.Bool(inputs.get('always_run_final', True))
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])

        return builder

    def validate_inputs(self):
        """Validate the inputs."""
        if 'interpolation_distances' in self.inputs:
            if isinstance(self.inputs.interpolation_distances, orm.List) and len(self.inputs.interpolation_distances) > 1:
                if not self.inputs.convergence_threshold:
                    raise ValueError('The `convergence_threshold` must be provided if the `interpolation_distances` is provided.')
            else:
                raise ValueError('The `interpolation_distances` must be a list with at least one element.')

            if EpwA2fWorkChain._INTP_NAMESPACE not in self.inputs:
                raise ValueError('The `a2f` must be provided as an input port.')
        else:
            if 'convergence_threshold' in self.inputs:
                raise ValueError('The `convergence_threshold` must not be provided if the `interpolation_distances` is not provided.')

    def setup(self):
        """Setup steps, i.e. initialise context variables."""
        # Here we expose the inputs in advance so that we can modify the
        # parent_folder_epw in the should_run_b2w, should_run_a2f, should_run_iso, should_run_aniso methods.

        if self._BANDS_NAMESPACE in self.inputs:
            self.ctx.inputs_bands = AttributeDict(self.exposed_inputs(EpwBandsWorkChain, namespace=self._BANDS_NAMESPACE))
        if self._A2F_NAMESPACE in self.inputs:
            self.ctx.inputs_a2f = AttributeDict(self.exposed_inputs(EpwA2fWorkChain, namespace=self._A2F_NAMESPACE))
            self.ctx.inputs_a2f[self._A2F_NAMESPACE].kfpoints_factor = self.inputs.kfpoints_factor

        if self._ISO_NAMESPACE in self.inputs:
            self.ctx.inputs_iso = AttributeDict(self.exposed_inputs(EpwIsoWorkChain, namespace=self._ISO_NAMESPACE))
            self.ctx.inputs_iso[self._ISO_NAMESPACE].kfpoints_factor = self.inputs.kfpoints_factor

        self.ctx.inputs_aniso = AttributeDict(self.exposed_inputs(EpwAnisoWorkChain, namespace=self._ANISO_NAMESPACE))
        self.ctx.inputs_aniso[self._ANISO_NAMESPACE].kfpoints_factor = self.inputs.kfpoints_factor

        if 'interpolation_distances' in self.inputs:
            self.report("Will check convergence")
            self.ctx.interpolation_distances = self.inputs.get('interpolation_distances').get_list()
            self.ctx.interpolation_distances.sort()
            self.ctx.final_a2f = None
            self.ctx.allen_dynes_values = []
            self.ctx.is_converged = False

    def should_run_b2w(self):
        """Check if the b2w workflow should continue or not."""

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
        workchain_node = self.submit(EpwB2WWorkChain, **inputs)

        self.report(f'launching EpwB2WWorkChain<{workchain_node.pk}>')

        return ToContext(workchain_b2w=workchain_node)

    def inspect_b2w(self):
        """Inspect the b2w workflow."""
        b2w_workchain = self.ctx.workchain_b2w

        if not b2w_workchain.is_finished_ok:
            self.report(f'`epw.x` failed with exit status {b2w_workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_B2W

        self.out_many(
            self.exposed_outputs(
                self.ctx.workchain_b2w,
                EpwB2WWorkChain,
                namespace=self._B2W_NAMESPACE
            )
        )

    def should_run_bands(self):
        """Check if the bands workflow should continue or not."""
        if self._BANDS_NAMESPACE in self.inputs:
            if self.should_run_b2w():
                b2w_workchain = self.ctx.workchain_b2w

                parent_folder_epw = b2w_workchain.outputs.epw.remote_stash

                self.ctx.inputs_bands[self._BANDS_NAMESPACE].parent_folder_epw = parent_folder_epw

            return True
        else:
            return False

    def run_bands(self):
        """Run the bands workflow."""
        inputs = self.ctx.inputs_bands
        inputs.structure = self.inputs.structure
        inputs.metadata.call_link_label = self._BANDS_NAMESPACE

        workchain_node = self.submit(EpwBandsWorkChain, **inputs)

        self.report(f'launching EpwBandsWorkChain<{workchain_node.pk}>')

        return ToContext(workchain_bands=workchain_node)

    def inspect_bands(self):
        """Inspect the bands workflow."""
        bands_workchain = self.ctx.workchain_bands

        if not bands_workchain.is_finished_ok:
            self.report(f'`epw.x` failed with exit status {bands_workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_BANDS

        self.out_many(
            self.exposed_outputs(
                self.ctx.workchain_bands,
                EpwBandsWorkChain,
                namespace=self._BANDS_NAMESPACE
            )
        )

        import numpy
        ph_bands = bands_workchain.outputs.bands.ph_band_structure.get_bands()
        min_freq = numpy.min(ph_bands)
        max_freq = numpy.max(ph_bands)

        if min_freq < EpwBandsWorkChain._MIN_FREQ:
            return self.exit_codes.ERROR_IMAGINARY_PHONON_FREQ

        for namespace, key in (
            (self._A2F_NAMESPACE, 'inputs_a2f'),
            (self._ISO_NAMESPACE, 'inputs_iso'),
            (self._ANISO_NAMESPACE, 'inputs_aniso'),
        ):
            if key in self.ctx:
                parameters = self.ctx[key][namespace].epw.parameters.get_dict()
                parameters['INPUTEPW']['degaussq'] = max_freq / 100
                self.ctx[key][namespace].epw.parameters = orm.Dict(parameters)

    def prepare_intp(self):
        """Prepare the inputs for the interpolation workflow."""

        if 'interpolation_distances' not in self.inputs:
            self.report('No interpolation distances provided, will not do interpolation.')
            return

        parameters = self.ctx.inputs_a2f[self._A2F_NAMESPACE].epw.parameters.get_dict()

        if self.should_run_b2w():
            b2w_workchain = self.ctx.workchain_b2w

            parent_folder_epw = b2w_workchain.outputs.epw.remote_stash

            self.ctx.inputs_a2f[self._A2F_NAMESPACE].parent_folder_epw = parent_folder_epw


    @staticmethod
    def check_convergence(
        a2f_conv_workchains: list[orm.WorkChainNode],
        convergence_threshold: float
        ) -> tuple[bool, str]:
        """Check if the convergence is reached."""

        try:
            prev_allen_dynes = a2f_conv_workchains[-2].outputs.output_parameters['Allen_Dynes_Tc']
            new_allen_dynes = a2f_conv_workchains[-1].outputs.output_parameters['Allen_Dynes_Tc']
            is_converged = (
                abs(prev_allen_dynes - new_allen_dynes) / new_allen_dynes
                < convergence_threshold
            )
            return (
                is_converged,
                f'Checking convergence: old {prev_allen_dynes}; new {new_allen_dynes} -> Converged = {is_converged}')
        except (AttributeError, IndexError, KeyError):
            return (False, 'Not enough data to check convergence.')

    def should_run_conv(self):
        """Check if the conv loop should continue or not."""

        if 'interpolation_distances' not in self.inputs:
            return False

        is_converged, msg = self.check_convergence(
            self.ctx.a2f_conv,
            self.inputs.convergence_threshold
        )

        self.ctx.is_converged = is_converged
        self.report(msg)

        ## TODO: If Allen-Dynes Tc is converged, clean the remote folder:
        ## rm out/aiida.ephmat/*, out/aiida.epwmatwp,
        ## Leaving only the final a2f calculation for later use.

        if self.ctx.is_converged:
            self.ctx.inputs_iso[self._ISO_NAMESPACE].parent_folder_epw = self.ctx.a2f_conv[-1].outputs.remote_folder
            return False

        else:
            if len(self.ctx.interpolation_distances) == 0 and not self.ctx.is_converged:
                if self.inputs.always_run_final.value:
                    self.ctx.inputs_iso[self._ISO_NAMESPACE].parent_folder_epw = self.ctx.a2f_conv[-1].outputs.remote_folder
                    self.report('Allen-Dynes Tc is not converged, but will run the following workchains!.')
                    return False
                else:
                    return self.exit_codes.ERROR_CONVERGENCE_NOT_REACHED
            else:
                return True

    def run_conv(self):
        """Run the ``restart`` EPW calculation for the current interpolation distance."""

        inputs = self.ctx.inputs_a2f
        inputs.structure = self.inputs.structure
        inputs.a2f.qfpoints_distance = self.ctx.interpolation_distances.pop()

        inputs.metadata.call_link_label = self._CONV_NAMESPACE
        workchain_node = self.submit(EpwA2fWorkChain, **inputs)

        self.report(f'launching convergence test: EpwA2fWorkChain<{workchain_node.pk}> [qfpoints_distance = {inputs.a2f.qfpoints_distance}]')

        return ToContext(a2f_conv=append_(workchain_node))

    def inspect_conv(self):
        """Verify that the conv workflow finished successfully."""
        a2f_workchain = self.ctx.a2f_conv[-1]

        if not a2f_workchain.is_finished_ok:
            self.report(f'`epw.x` failed with exit status {a2f_workchain.exit_status}')
            self.ctx.a2f_conv.pop()
        else:
            ## TODO: Use better way to get the mesh
            a2f_calculation = a2f_workchain.called_descendants[-1]
            mesh = 'x'.join(str(i) for i in a2f_calculation.inputs.qfpoints.get_kpoints_mesh()[0])

            try:
                self.report(f"Allen-Dynes: {a2f_workchain.outputs.output_parameters['Allen_Dynes_Tc']} at {mesh}")
            except KeyError:
                self.report(f"Could not find Allen-Dynes temperature in parsed output parameters!")

    def should_run_a2f(self):
        """Check if the a2f workflow should continue or not."""
        if (
            self._A2F_NAMESPACE in self.inputs
            and
            'interpolation_distances' not in self.inputs
        ):
            if self.should_run_b2w():
                b2w_workchain = self.ctx.workchain_b2w
                parent_folder_epw = b2w_workchain.outputs.epw.remote_stash
                self.ctx.inputs_a2f[self._A2F_NAMESPACE].parent_folder_epw = parent_folder_epw
            return True
        else:
            return False

    def run_a2f(self):
        """Run the a2f workflow."""
        inputs = self.ctx.inputs_a2f
        inputs.structure = self.inputs.structure

        inputs.metadata.call_link_label = self._A2F_NAMESPACE
        workchain_node = self.submit(EpwA2fWorkChain, **inputs)

        self.report(f'launching EpwA2fWorkChain<{workchain_node.pk}>')

        return ToContext(workchain_a2f=workchain_node)

    def inspect_a2f(self):
        """Inspect the a2f workflow."""
        a2f_workchain = self.ctx.workchain_a2f
        if not a2f_workchain.is_finished_ok:
            self.report(f'`epw.x` failed with exit status {a2f_workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_A2F

        self.out_many(
            self.exposed_outputs(
                self.ctx.workchain_a2f,
                EpwA2fWorkChain,
                namespace=self._A2F_NAMESPACE
            )
        )

        self.ctx.inputs_iso[self._ISO_NAMESPACE].parent_folder_epw = a2f_workchain.outputs.remote_folder

    def should_run_iso(self):
        """Check if the iso workflow should continue or not."""

        if self._ISO_NAMESPACE in self.inputs:
            if self.should_run_a2f():
                a2f_workchain = self.ctx.workchain_a2f
                parent_folder_epw = a2f_workchain.outputs.remote_folder
                self.ctx.inputs_iso[self._ISO_NAMESPACE].parent_folder_epw = parent_folder_epw

                Allen_Dynes_Tc = self.ctx.inputs_iso[self._ISO_NAMESPACE].parent_folder_epw.creator.outputs.output_parameters['Allen_Dynes_Tc']

                if Allen_Dynes_Tc < EpwIsoWorkChain._MIN_TEMP:
                    return self.exit_codes.ERROR_ALLEN_DYNES_TC_TOO_LOW

                self.ctx.inputs_iso.estimated_Tc_iso = orm.Float(Allen_Dynes_Tc * 1.5)

            return True
        else:
            return False

    def run_iso(self):
        """Run the iso workflow."""
        inputs = self.ctx.inputs_iso
        inputs.structure = self.inputs.structure
        inputs.restart_type = orm.EnumData(RestartType.FROM_EPHMAT)
        parent_folder_epw = inputs[self._ISO_NAMESPACE].parent_folder_epw

        # TODO: This will only work if qfpoints and kfpoints are defined by qfpoint_distance and kfpoints_factor.
        #       We should find a better way to handle this.
        inputs[self._ISO_NAMESPACE].qfpoints_distance = parent_folder_epw.creator.caller.inputs.qfpoints_distance
        inputs[self._ISO_NAMESPACE].kfpoints_factor = parent_folder_epw.creator.caller.inputs.kfpoints_factor

        inputs.metadata.call_link_label = self._ISO_NAMESPACE
        workchain_node = self.submit(EpwIsoWorkChain, **inputs)
        self.report(f'launching EpwIsoWorkChain<{workchain_node.pk}>')

        return ToContext(workchain_iso=workchain_node)

    def inspect_iso(self):
        """Inspect the iso workflow."""
        iso_workchain = self.ctx.workchain_iso
        if not iso_workchain.is_finished_ok:
            self.report(f'`epw.x` failed with exit status {iso_workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_ISO

        self.out_many(
            self.exposed_outputs(
                self.ctx.workchain_iso,
                EpwIsoWorkChain,
                namespace=self._ISO_NAMESPACE
            )
        )

        self.ctx.inputs_aniso[self._ANISO_NAMESPACE].parent_folder_epw = iso_workchain.outputs.remote_folder

    def should_run_aniso(self):
        """Check if the aniso workflow should continue or not."""

        if self._ANISO_NAMESPACE in self.inputs:
            if self.should_run_iso():
                iso_workchain = self.ctx.workchain_iso
                Tc_iso = iso_workchain.outputs.Tc_iso.value

                if Tc_iso < EpwAnisoWorkChain._MIN_TEMP:
                    return self.exit_codes.ERROR_ISOTROPIC_TC_TOO_LOW

                self.ctx.inputs_aniso.estimated_Tc_aniso = orm.Float(Tc_iso * 2.0)

            return True
        else:
            return False

    def run_aniso(self):
        """Run the aniso workflow."""
        inputs = self.ctx.inputs_aniso
        inputs.structure = self.inputs.structure
        inputs.restart_type = orm.EnumData(RestartType.FROM_EPHMAT)

        parent_folder_epw = inputs[self._ANISO_NAMESPACE].parent_folder_epw

        inputs[self._ANISO_NAMESPACE].qfpoints_distance = parent_folder_epw.creator.caller.inputs.qfpoints_distance
        inputs[self._ANISO_NAMESPACE].kfpoints_factor = parent_folder_epw.creator.caller.inputs.kfpoints_factor

        inputs.metadata.call_link_label = self._ANISO_NAMESPACE
        workchain_node = self.submit(EpwAnisoWorkChain, **inputs)
        self.report(f'launching EpwAnisoWorkChain<{workchain_node.pk}>')

        return ToContext(workchain_aniso=workchain_node)

    def inspect_aniso(self):
        """Inspect the aniso workflow."""
        aniso_workchain = self.ctx.workchain_aniso
        if not aniso_workchain.is_finished_ok:
            self.report(f'`epw.x` failed with exit status {aniso_workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_ANISO

        self.out_many(
            self.exposed_outputs(
                self.ctx.workchain_aniso,
                EpwAnisoWorkChain,
                namespace=self._ANISO_NAMESPACE
            )
        )
    def results(self):
        """TODO"""

        if 'a2f_conv' in self.ctx:
            self.out_many(
                self.exposed_outputs(
                    self.ctx.a2f_conv[-1],
                    EpwA2fWorkChain,
                    namespace=self._CONV_NAMESPACE
                )
            )

    def on_terminated(self):
        """Clean the working directories of all child calculations if `clean_workdir=True` in the inputs."""
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

