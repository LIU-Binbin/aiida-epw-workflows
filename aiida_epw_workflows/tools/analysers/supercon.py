from doctest import FAIL_FAST
from email import message
from re import S
import re
from socket import NI_NOFQDN
from pathlib import Path
from tkinter import E

from aiida import orm
from aiida.common.links import LinkType
from aiida.engine import ProcessState
import numpy
from ..workchains import clean_workdir
from .base import BaseWorkChainAnalyser
from enum import Enum
from aiida.tools import delete_nodes
from .b2w import EpwB2WWorkChainAnalyser
from ..calculators import _calculate_iso_tc, check_convergence
from ..plot import (
    plot_epw_interpolated_bands,
    plot_a2f,
    plot_eldos,
    plot_gap_function,
    plot_bands_comparison,
    plot_bands,
    plot_aniso,
    plot_phdos
)


class EpwSuperConWorkChainState(Enum):
    FINISHED_OK = 0
    WAITING = 1
    RUNNING = 2
    EXCEPTED = 3
    KILLED = 4
    PW_RELAX_FAILED = 4001
    PW_BANDS_FAILED = 4002
    B2W_W90_INTP_SCF_FAILED = 4004
    B2W_W90_INTP_NSCF_FAILED = 4005
    B2W_W90_INTP_PW2WAN_FAILED = 4006
    B2W_W90_INTP_WANNIER_FAILED = 4007
    B2W_PH_BASE_S_MATRIX_NOT_POSITIVE_DEFINITE = 4008
    B2W_PH_BASE_NODE_FAILURE = 4009
    B2W_PH_BASE_FAILED = 4008
    B2W_PH_BASE_UNSTABLE = 4009
    B2W_MATDYN_BASE_UNSTABLE = 4010
    B2W_EPW_BASE_FAILED = 4011
    BANDS_FAILED = 4012
    BANDS_UNSTABLE = 4013
    CONVERGENCE_NOT_REACHED = 4014
    A2F_FAILED = 4015
    A2F_CONV_FAILED = 4016
    ALLEN_DYNES_TC_TOO_LOW = 4017
    ISO_FAILED = 4018
    ISO_TC_TOO_LOW = 4019
    ANISO_FAILED = 4020
    PW_RELAX_EXCEPTED = 901
    PW_BANDS_EXCEPTED = 902
    B2W_EXCEPTED = 903
    B2W_W90_INTP_EXCEPTED = 904
    B2W_PH_BASE_EXCEPTED = 905
    B2W_EPW_BASE_EXCEPTED = 906
    BANDS_EXCEPTED = 907
    A2F_EXCEPTED = 908
    A2F_CONV_EXCEPTED = 909
    ISO_EXCEPTED = 910
    ANISO_EXCEPTED = 911
    PW_RELAX_KILLED = 912
    PW_BANDS_KILLED = 913
    B2W_W90_INTP_SCF_KILLED = 914
    B2W_W90_INTP_NSCF_KILLED = 915
    B2W_W90_INTP_PW2WAN_KILLED = 916
    B2W_W90_INTP_WANNIER_KILLED = 917
    B2W_PH_BASE_KILLED = 918
    B2W_EPW_BASE_KILLED = 919
    B2W_MATDYN_BASE_KILLED = 920
    BANDS_KILLED = 921
    A2F_CONV_KILLED = 922
    A2F_KILLED = 923
    ISO_KILLED = 924
    ANISO_KILLED = 925
    PW_RELAX_FINISHED_OK = 1001
    PW_BANDS_FINISHED_OK = 1002
    B2W_W90_INTP_SCF_FINISHED_OK = 1003
    B2W_W90_INTP_NSCF_FINISHED_OK = 1004
    B2W_W90_INTP_PW2WAN_FINISHED_OK = 1005
    B2W_W90_INTP_WANNIER_FINISHED_OK = 1006
    B2W_PH_BASE_FINISHED_OK = 1007
    B2W_EPW_BASE_FINISHED_OK = 1008
    B2W_FINISHED_OK = 1009
    BANDS_FINISHED_OK = 1010
    A2F_FINISHED_OK = 1011
    A2F_CONV_FINISHED_OK = 1012
    ISO_FINISHED_OK = 1013
    ANISO_FINISHED_OK = 1014
    UNKNOWN = 999

from collections import OrderedDict

class EpwSuperConWorkChainAnalyser(BaseWorkChainAnalyser):
    """
    Analyser for the EpwSuperConWorkChain.
    """
    _all_descendants = OrderedDict([
        ('pw_relax', None),
        ('pw_bands', None),
        ('b2w',      None),
        ('bands',    None),
        ('a2f_conv', None),
        ('a2f',      None),
        ('iso',      None),
        ('aniso',    None),
    ])

    def __init__(self, workchain: orm.WorkChainNode):
        super().__init__(workchain)
        self.state = EpwSuperConWorkChainState.UNKNOWN
        for link_label, _ in self._all_descendants.items():
            descendants = workchain.base.links.get_outgoing(link_label_filter=link_label).all_nodes()
            if descendants != []:
                self.descendants[link_label] = descendants

    @staticmethod
    def base_check(
        workchain: orm.WorkChainNode,
        excepted_state: EpwSuperConWorkChainState,
        failed_state: EpwSuperConWorkChainState,
        killed_state: EpwSuperConWorkChainState,
        finished_ok_state: EpwSuperConWorkChainState,
        namespace: str,
        ) -> tuple[EpwSuperConWorkChainState, str]:
        if not workchain:
            return (
                finished_ok_state,
                '{namespace} is not found, should be skipped'
            )
        if workchain.process_state == ProcessState.WAITING:
            state = EpwSuperConWorkChainState.WAITING
            message = f'is waiting at {namespace}'
        elif workchain.process_state == ProcessState.RUNNING:
            state = EpwSuperConWorkChainState.RUNNING
            message = f'is running at {namespace}'
        elif workchain.process_state == ProcessState.EXCEPTED:
            state = excepted_state
            message = f'has excepted at {namespace}'
        elif workchain.process_state == ProcessState.KILLED:
            state = killed_state
            message = f'has killed at {namespace}'
        elif workchain.process_state == ProcessState.FINISHED:
            if not workchain.is_finished_ok:
                state = failed_state
                message = f'has failed at {namespace}'
            else:
                state = finished_ok_state
                message = f'{namespace} is finished successfully'
        else:
            state = EpwSuperConWorkChainState.UNKNOWN
            message = f'unknown state at {namespace}'

        return state, message

    @property
    def structure(self):
        if self.node.inputs.structure is None:
            raise ValueError('structure is not found')
        else:
            return self.node.inputs.structure

    @property
    def pw_relax(self):
        if self.descendants['pw_relax'] == []:
            raise ValueError('pw_relax is not found')
        else:
            return self.descendants['pw_relax']

    @property
    def pw_bands(self):
        if self.descendants['pw_bands'] == []:
            raise ValueError('pw_bands is not found')
        else:
            return self.descendants['pw_bands']

    @property
    def b2w(self):
        if self.descendants['b2w'] == []:
            raise ValueError('b2w is not found')
        else:
            return self.descendants['b2w']

    @property
    def b2w_analyser(self):
        if self.descendants['b2w'] == []:
            return None
        else:
            return EpwB2WWorkChainAnalyser(self.descendants['b2w'][-1])

    @property
    def b2w_w90_intp(self):
        if self.descendants['b2w'] == []:
            raise None
        else:
            return self.b2w_analyser.w90_intp

    @property
    def b2w_ph_base(self):
        if self.descendants['b2w'] == []:
            raise None
        else:
            return self.b2w_analyser.ph_base

    @property
    def b2w_q2r_base(self):
        if self.descendants['b2w'] == []:
            raise None
        else:
            return self.b2w_analyser.q2r_base

    @property
    def b2w_matdyn_base(self):
        if self.descendants['b2w'] == []:
            raise None
        else:
            return self.b2w_analyser.matdyn_base
    @property
    def b2w_epw_base(self):
        if self.descendants['b2w'] == []:
            raise None
        else:
            return self.b2w_analyser.epw_base

    @property
    def epw_bands(self):
        if self.descendants['bands'] == []:
            raise None
        else:
            return self.descendants['bands']

    @property
    def a2f_conv(self):
        if 'a2f_conv' not in self.descendants or self.descendants['a2f_conv'] == []:
            return None
        else:
            return self.descendants['a2f_conv']

    @property
    def a2f(self):
        if 'a2f' not in self.descendants or self.descendants['a2f'] == []:
            return None
        else:
            return self.descendants['a2f']

    @property
    def iso(self):
        if self.descendants['iso'] == []:
            raise None
        else:
            return self.descendants['iso']

    @property
    def aniso(self):
        if self.descendants['aniso'] == []:
            return None
        else:
            return self.descendants['aniso']

    def check_pw_relax(self):
        """Check the state of the pw_relax workchain."""


        return self.base_check(
            self.pw_relax[-1],
            EpwSuperConWorkChainState.PW_RELAX_EXCEPTED,
            EpwSuperConWorkChainState.PW_RELAX_FAILED,
            EpwSuperConWorkChainState.PW_RELAX_KILLED,
            EpwSuperConWorkChainState.PW_RELAX_FINISHED_OK,
            'pw_relax'
        )

    def check_pw_bands(self):
        """Check the state of the pw_bands workchain."""
        return self.base_check(
            self.pw_bands[0],
            EpwSuperConWorkChainState.PW_BANDS_EXCEPTED,
            EpwSuperConWorkChainState.PW_BANDS_FAILED,
            EpwSuperConWorkChainState.PW_BANDS_KILLED,
            EpwSuperConWorkChainState.PW_BANDS_FINISHED_OK,
            'pw_bands'
        )

    def check_b2w(self):
        """Check the state of the b2w workchain."""
        b2w_analyser = EpwB2WWorkChainAnalyser(self.descendants['b2w'][-1])
        return b2w_analyser.check_process_state()

    def check_bands(self):
        """Check the state of the bands workchain."""
        return self.base_check(
            self.epw_bands[0],
            EpwSuperConWorkChainState.BANDS_EXCEPTED,
            EpwSuperConWorkChainState.BANDS_FAILED,
            EpwSuperConWorkChainState.BANDS_KILLED,
            EpwSuperConWorkChainState.BANDS_FINISHED_OK,
            'epw_bands'
        )

    def check_a2f(self):
        if 'a2f' in self.descendants:
            return self.base_check(
                self.a2f[0],
                EpwSuperConWorkChainState.A2F_EXCEPTED,
                EpwSuperConWorkChainState.A2F_FAILED,
                EpwSuperConWorkChainState.A2F_KILLED,
                EpwSuperConWorkChainState.A2F_FINISHED_OK,
                'a2f'
            )
        elif 'a2f_conv' in self.descendants:
            for a2f_conv_workchain in self.a2f_conv:
                if a2f_conv_workchain.is_excepted:
                    state = EpwSuperConWorkChainState.A2F_CONV_EXCEPTED
                    message += f'has excepted at a2f_conv<{a2f_conv_workchain.pk}>'
                else:
                    state = EpwSuperConWorkChainState.A2F_CONV_FAILED
                    message += f'has failed at a2f_conv<{a2f_conv_workchain.pk}>'
            return state, message
        else:
            return None, None

    def check_iso(self):
        """Check the state of the iso workchain."""
        return self.base_check(
            self.iso[0],
            EpwSuperConWorkChainState.ISO_EXCEPTED,
            EpwSuperConWorkChainState.ISO_FAILED,
            EpwSuperConWorkChainState.ISO_KILLED,
            EpwSuperConWorkChainState.ISO_FINISHED_OK,
            'iso'
        )

    def check_aniso(self):
        return self.base_check(
            self.aniso[0],
            EpwSuperConWorkChainState.ANISO_EXCEPTED,
            EpwSuperConWorkChainState.ANISO_FAILED,
            EpwSuperConWorkChainState.ANISO_KILLED,
            EpwSuperConWorkChainState.ANISO_FINISHED_OK,
            'aniso'
        )

    def check_process_state(self):
        """Check the state of the workchain."""
        state = EpwSuperConWorkChainState.UNKNOWN
        message = 'status is not known'

        state, message = self.base_check(
            self.node,
            EpwSuperConWorkChainState.EXCEPTED,
            EpwSuperConWorkChainState.UNKNOWN,
            EpwSuperConWorkChainState.KILLED,
            EpwSuperConWorkChainState.FINISHED_OK,
            'supercon'
        )

        if state != EpwSuperConWorkChainState.UNKNOWN:
            return state, message

        for check_func, finished_ok_state in (
            (self.check_pw_relax, EpwSuperConWorkChainState.PW_RELAX_FINISHED_OK),
            (self.check_pw_bands, EpwSuperConWorkChainState.PW_BANDS_FINISHED_OK),
            (self.check_b2w, EpwSuperConWorkChainState.B2W_FINISHED_OK),
            (self.check_bands, EpwSuperConWorkChainState.BANDS_FINISHED_OK),
            (self.check_a2f, EpwSuperConWorkChainState.A2F_FINISHED_OK),
            # (self.check_a2f_conv, EpwSuperConWorkChainState.A2F_CONV_FINISHED_OK),
            (self.check_iso, EpwSuperConWorkChainState.ISO_FINISHED_OK),
            (self.check_aniso, EpwSuperConWorkChainState.ANISO_FINISHED_OK),
        ):
            state, message = check_func()
            if not state == finished_ok_state:
                return state, message

        return EpwSuperConWorkChainState.FINISHED_OK, 'has finished successfully'

    @property
    def outputs_parameters(self):
        from ase.spacegroup import get_spacegroup
        outputs_parameters = {}

        structure = self.structure

        outputs_parameters['Formula'] = structure.get_formula()
        sg = get_spacegroup(structure.get_ase(), symprec=1e-6)
        outputs_parameters['Space group'] = f"[{sg.no}] {sg.symbol}"
        if self.b2w_w90_intp:
            scf_output_parameters = self.b2w_w90_intp[-1].outputs.scf.output_parameters
            outputs_parameters['Coarse Fermi energy'] = scf_output_parameters.get('fermi_energy')
            outputs_parameters['SCF k-points'] = " x ".join(map(str, scf_output_parameters.get('monkhorst_pack_grid')))
            outputs_parameters['Total energy'] = scf_output_parameters.get('energy')
            outputs_parameters['Number of electrons'] = scf_output_parameters.get('number_of_electrons')
            outputs_parameters['WFC cutoff'] = scf_output_parameters.get('wfc_cutoff')
            outputs_parameters['Degauss'] = scf_output_parameters.get('degauss')
        if self.b2w_epw_base:
            iteration_01 = self.get_descendants_by_label(self.b2w_epw_base[-1], 'iteration_01')[0].node
            outputs_parameters['Coarse k-points'] = " x ".join(map(str, iteration_01.inputs.kpoints.get_kpoints_mesh()[0]))
            outputs_parameters['Coarse q-points'] = " x ".join(map(str, iteration_01.inputs.qpoints.get_kpoints_mesh()[0]))
            outputs_parameters['Number of Wannier functions'] = self.b2w_epw_base[-1].outputs.output_parameters.get('nbndsub')
        if self.iso:
            outputs_parameters['w log'] = self.iso[-1].outputs.output_parameters.get('w_log')
            outputs_parameters['lambda'] = self.iso[-1].outputs.output_parameters.get('lambda')
            outputs_parameters['Allen_Dynes_Tc'] = self.iso[-1].outputs.output_parameters.get('Allen_Dynes_Tc')
            outputs_parameters['iso_tc'] = self.iso[-1].outputs.Tc_iso.value
        elif self.a2f:
            a2f_output_parameters = self.a2f[-1].outputs.output_parameters
            outputs_parameters['w log'] = a2f_output_parameters.get('w_log')
            outputs_parameters['lambda'] = a2f_output_parameters.get('lambda')
            outputs_parameters['Allen_Dynes_Tc'] = a2f_output_parameters.get('Allen_Dynes_Tc')
        elif self.a2f_conv:
            a2f_conv_output_parameters = self.a2f_conv[-1].outputs.output_parameters
            outputs_parameters['w log'] = a2f_conv_output_parameters.get('w_log')
            outputs_parameters['lambda'] = a2f_conv_output_parameters.get('lambda')
            outputs_parameters['Allen_Dynes_Tc'] = a2f_conv_output_parameters.get('Allen_Dynes_Tc')

        return outputs_parameters

    def get_state(self):
        pk = self.node.pk
        formula = self.node.inputs.structure.get_formula()
        source_db, source_id = self.node.inputs.structure.base.extras.get_many(('source_db', 'source_id'))
        state, message = self.check_process_state()
        material_info = f'{source_db}-{source_id}<{formula}>'
        message = f'[{pk}]: {material_info:30s} {message}'

        return state, message

    def get_source(self):
        """Get the source of the workchain."""
        if all(key in self.node.base.extras for key in ['source_db', 'source_id']):
            return (self.node.base.extras.get('source_db'), self.node.base.extras.get('source_id'))
        elif all(key in self.node.inputs.structure.base.extras for key in ['source_db', 'source_id']):
            return (self.node.inputs.structure.base.extras.get('source_db'), self.node.inputs.structure.base.extras.get('source_id'))
        else:
            raise ValueError('Source is not set')

    @property
    def a2f_results(self):
        """Get the results of the a2f workchain."""
        a2f_results = {}
        if self.a2f:
            for a2f_workchain in self.a2f:
                qfpoints_distance = a2f_workchain.inputs.a2f.qfpoints_distance.value
                a2f_results[qfpoints_distance] = a2f_workchain.outputs.output_parameters
        elif self.a2f_conv:
            for a2f_workchain in self.a2f_conv:
                qfpoints_distance = a2f_workchain.inputs.a2f.qfpoints_distance.value
                a2f_results[qfpoints_distance] = a2f_workchain.outputs.output_parameters
        else:
            print('No a2f workchain found')
            a2f_results = None
            
        return a2f_results

    @property
    def converged_allen_dynes_Tc(self, threshold=0.1):
        """Get the results of the a2f workchain."""
        if not self.a2f_conv:
            print('No a2f_conv workchain found')
            return None
        else:
            Tcs = [a2f_result.get('Allen_Dynes_Tc') for a2f_result in self.a2f_results.values()]
            print(Tcs)
            _, converged_allen_dynes_Tc = check_convergence(
                Tcs,
                threshold
            )
            return converged_allen_dynes_Tc
        
    # TODO: This function is only used temporarily before the error handler of EpwSuperconWorkChain
    #       is completed.
    @property
    def iso_results(self):
        """Get the results of the iso workchain."""
        from aiida_epw_workflows.parsers.epw import EpwParser
        results = {}
        for iteration, folderdata in self.retrieved['iso']['iso'].items():
            parsed_stdout, _ = EpwParser.parse_stdout(folderdata.get_object_content('aiida.out'), None)
            results[iteration] = parsed_stdout
        
        return results

    @property
    def iso_max_eigenvalues(self):
        """Get the max eigenvalues of the iso workchain."""
        max_eigenvalues = []
        for iteration, parsed_stdout in self.iso_results.items():
            max_eigenvalues.append(parsed_stdout['max_eigenvalue'].get_array('max_eigenvalue'))
        return numpy.concatenate(max_eigenvalues, axis=1)

    # TODO: This function can't treat the case where minimal eigenvalue is larger than 1.0.
    @property
    def iso_tc(self):
        """Get the tc of the iso workchain."""
        try:
            return _calculate_iso_tc(self.iso_max_eigenvalues, allow_extrapolation=True)
        except (AttributeError, KeyError, ValueError):
            return None

    def get_aniso_remote_path(self):
        """Get the remote directory of the aniso workchain."""
        return self.processes_dict['aniso']['aniso']

    @property
    def processes_dict(self):
        """Get the processes dictionary."""
        return EpwSuperConWorkChainAnalyser.get_processes_dict(self.node)

    @property
    def retrieved(self):
        """Get the retrieved dictionary."""
        return EpwSuperConWorkChainAnalyser.get_retrieved(self.node)

    @property
    def source(self):
        """Get the source of the workchain."""
        try:
            source_db, source_id = self.get_source()
            return f'{source_db}-{source_id}'
        except (ValueError, KeyError):
            return None

    def set_source(self):
        """Set the source of the workchain."""
        if all(key in self.node.base.extras for key in ['source_db', 'source_id']):
            raise Warning('Source is already set')
        else:
            source_db, source_id = self.get_source()
            self.node.base.extras.set_many({
                'source_db': source_db,
                'source_id': source_id
            })

    def clean_workchain(self, dry_run=True):
        """Clean the workchain."""

        message = super().clean_workchain([
            EpwSuperConWorkChainState.FINISHED_OK,
            EpwSuperConWorkChainState.WAITING,
            EpwSuperConWorkChainState.RUNNING,
            EpwSuperConWorkChainState.B2W_PH_BASE_UNSTABLE,
            EpwSuperConWorkChainState.B2W_MATDYN_BASE_UNSTABLE,
            EpwSuperConWorkChainState.BANDS_UNSTABLE,
            EpwSuperConWorkChainState.CONVERGENCE_NOT_REACHED,
            EpwSuperConWorkChainState.ALLEN_DYNES_TC_TOO_LOW,
            EpwSuperConWorkChainState.ISO_TC_TOO_LOW,
            ],
            dry_run=dry_run
            )

        return message

    def check_convergence_allen_dynes_tc(
        self,
        convergence_threshold: float
        ) -> tuple[bool, str]:
        """Check if the convergence is reached."""

        a2f_conv_workchains = self.a2f_conv

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

    def check_stability_epw_bands(
        self,
        min_freq: float # meV ~ 8.1 cm-1
        ) -> tuple[bool, str]:
        """Check if the epw.x interpolated phonon band structure is stable."""
        if self.epw_bands is None:
            raise ValueError('No epw bands found.')
        ph_bands = self.epw_bands[-1].outputs.ph_band_structure.get_bands()
        min_freq = numpy.min(ph_bands)
        max_freq = numpy.max(ph_bands)

        if min_freq < min_freq:
            return (False, max_freq)
        else:
            return (True, max_freq)

    def dump_inputs(self, destpath: Path):
        super()._dump_inputs(
            self.processes_dict,
            destpath=destpath
        )

    def show_pw_bands(self):
        """Show the qe bands."""
        bands = self.pw_bands[0].outputs.band_structure
        bands.show_mpl()

    def show_eldos(
        self,
        axis = None,
        **kwargs,
        ):
        if self.a2f:
            plot_eldos(
                a2f_workchain = self.a2f[-1],
                axis = axis,
                **kwargs,
            )
        elif self.a2f_conv:
            plot_eldos(
                a2f_workchain = self.a2f_conv[-1],
                axis = axis,
                **kwargs,
            )

    def show_phdos(
        self,
        axis = None,
        **kwargs,
        ):

        if self.a2f:
            plot_phdos(
                a2f_workchain = self.a2f[-1],
                axis = axis,
                **kwargs,
            )
        elif self.a2f_conv:
            plot_phdos(
                a2f_workchain = self.a2f_conv[-1],
                axis = axis,
                **kwargs,
            )

    def show_a2f(self, axis=None, **kwargs):
        if self.a2f:
            plot_a2f(
                a2f_workchain = self.a2f[-1],
                axis = axis,
                **kwargs,
            )
        elif self.a2f_conv:
            plot_a2f(
                a2f_workchain = self.a2f_conv[-1],
                axis = axis,
                **kwargs,
            )

    def show_epw_bands(self, axes=None, **kwargs):
        if self.epw_bands:
            plot_epw_interpolated_bands(
                epw_workchain = self.epw_bands[-1],
                axes = axes,
                **kwargs,
            )

    def show_gap_function(self, axis=None, **kwargs):
        if self.aniso:
            plot_gap_function(
                aniso_workchain = self.aniso[-1],
                axis = axis,
                **kwargs,
        )

    def plot_all(self):
        kwargs = {
            'label_fontsize': 18,
            'ticklabel_fontsize': 18,
            'legend_fontsize': 12,
        }
        from matplotlib import gridspec
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(18, 8))
        gs = gridspec.GridSpec(2, 3, width_ratios=[4, 1, 4])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[0, 1])
        ax4 = fig.add_subplot(gs[1, 1])
        ax5 = fig.add_subplot(gs[1, 2])
        ax6 = fig.add_subplot(gs[0, 2])

        ax6.axis('off')
        data = list(self.outputs_parameters.items())

        the_table = ax6.table(
            cellText=data,
            loc='center',
            cellLoc='left',
            )

        for _, cell in the_table.get_celld().items():
            cell.set_edgecolor('none')
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(kwargs['legend_fontsize'])
        the_table.scale(1, 1.2)

        self.show_epw_bands(
            axes=numpy.array([ax1, ax2]),
            **kwargs,
        )

        self.show_eldos(
            axis = ax3,
            **kwargs,
            )
        ax3.set_ylabel("")
        ax3.set_yticks([], [])

        self.show_a2f(
            axis = ax4,
            show_data = False,
            **kwargs,
            )
        ax4.set_ylabel("")
        ax4.set_yticks([], [])
        self.show_gap_function(
            axis = ax5,
            **kwargs,
            )

