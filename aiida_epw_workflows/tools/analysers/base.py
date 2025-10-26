from aiida import orm
from aiida.common.links import LinkType
from aiida.engine import ProcessState
from enum import Enum
from collections import OrderedDict
from abc import ABC, abstractmethod
from pathlib import Path
from ..workchains import clean_workdir
from aiida.tools import delete_nodes
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable



@dataclass
class ProcessTree:
    """
    A tree structure to represent the processes of a workchain.
    """
    name: str = 'ROOT'  # The name of the node (e.g. 'pw_relax', 'iteration_01')
    node: Optional[orm.WorkChainNode | orm.CalcJobNode] = None  # The AiiDA node object (WorkChainNode or CalcJobNode)
    children: Dict[str, 'ProcessTree'] = field(default_factory=dict) # The children nodes, indexed by the name
    
    # Overload the constructor to build the tree from the original dictionary
    def __init__(self, aiida_node: orm.WorkChainNode | orm.CalcJobNode, name: str = 'ROOT'):
        """
        Initialize the ProcessTree node and recursively build the child tree.
        
        :param aiida_node: The AiiDA node object (WorkChainNode or CalcJobNode).
        :param name: The name of the current node (for the root node can be any string, for the child nodes is the link_label).
        """
        self.name = name
        self.node = aiida_node
        self.children = {}
        
        # Only WorkChainNode has 'called' subprocesses
        # We use try-except block to handle CalcJobNode or other nodes without .called attribute
        try:
            # Iterate over all subprocesses called by the current node
            subprocesses = list(aiida_node.called)
            subprocesses.sort(key=lambda p: p.ctime)
            
            for subprocess in subprocesses:
                
                # Extract the link_label of the subprocess, as the name of the child node
                # Assume all subprocesses have metadata_inputs and contain call_link_label
                try:
                    link_label = subprocess.base.attributes.all['metadata_inputs']['metadata']['call_link_label']
                except Exception:
                    # If no label, use the pk or uuid of the subprocess as a fallback
                    link_label = f"unlabeled_process_{subprocess.pk}"

                # Recursively create the ProcessTree child node
                # The power of this is that it can handle CalcJobNode stopping the recursion,
                # and WorkChainNode continuing the recursion.
                
                # Key point: Directly call ProcessTree(subprocess, link_label)
                # This will delegate the recursive construction logic to the ProcessTree constructor of the child node
                child_node = ProcessTree(aiida_node=subprocess, name=link_label)
                
                # Add the child node to the children dictionary of the current node
                self.children[link_label] = child_node

        except AttributeError:
            # If the node does not have the .called attribute (e.g. CalcJobNode or other non-WorkChainNode),
            # an AttributeError will be raised, and we stop the recursion, the children dictionary remains empty.
            pass

    def print(self):
        """
        Print the process tree.
        """
        print(self.name)
        for child in self.children.values():
            child.print()

    def print_tree(self, prefix: str = "", is_last: bool = True):
        """
        Manually print the tree structure to the console.
        """
        
        # Determine the prefix and connector line of the current node
        connector = "└── " if is_last else "├── "
        
        # Get the node information
        node_id = getattr(self.node, 'pk', 'N/A')
        node_type = self.node.process_label
        label = f"{self.name} ({node_type} PK: {node_id})"
        
        # Print the current node
        print(prefix + connector + label)
        
        # Determine the indentation of the next layer
        # If the current node is not the last child node, the next layer needs to continue using the vertical line '│ '
        next_prefix = prefix + ("    " if is_last else "│   ")
        
        # Recursively print the child nodes
        children_list = list(self.children.values())
        for i, child in enumerate(children_list):
            is_last_child = (i == len(children_list) - 1)
            child.print_tree(prefix=next_prefix, is_last=is_last_child)

    def print_nodes_info(
            self,
            target_node_type: str, 
            extractor: Callable[[Any], Dict[str, Any]],
            prefix: str = "",
            is_last: bool = True,
        ) -> None:
        """
        Recursively traverse the ProcessTree, collect the information of all matching target type nodes.

        :param target_node_type: The target AiiDA node type string (e.g. 'WorkChainNode').
        :param extractor: A function that takes an AiiDA node and returns a dictionary containing the desired information.
        :return: A list of dictionaries containing the information of all matching nodes.
        """
        
        connector = "└── " if is_last else "├── "
        
        # Get the node information
        node_id = getattr(self.node, 'pk', 'N/A')
        node_type = self.node.node_type
        process_label = self.node.process_label
        label = f"{self.name} ({process_label} PK: {node_id})"
        next_prefix = prefix + ("    " if is_last else "│   ")                        
        # 1. Check if the current node matches the target node type
        if target_node_type == node_type:
            # If matched, use the provided extractor function to extract the information
            info = extractor(self.node)
            print(prefix + connector + label + ": " + info)
        else:
            print(prefix + connector + label)
        # 2. Recursively traverse the child nodes
        children_list = list(self.children.values())
        for i, child in enumerate(children_list):
            is_last_child = (i == len(children_list) - 1)
            child.print_nodes_info(target_node_type, extractor, next_prefix, is_last_child)

    @staticmethod
    def traverse_and_check(
        node: 'ProcessTree',
        current_path: str,
        ):
        """
        Traverse the ProcessTree and check if the node is errored.
        """

        new_path = f"{current_path}/{node.name}" if current_path else node.name
        node_type = str(type(node.node))
        if node_type == 'process.calculation.calcjob.CalcJobNode.' and not node.node.is_finished_ok:
            return new_path
        for child_node in node.children.values():
            ProcessTree.traverse_and_check(child_node, new_path)

    def get_error_path(self) -> str:
        """
        Get the error path of the first errored CalcJobNode in the process tree.
        """
        return ProcessTree.traverse_and_check(self.node, '')

    @staticmethod
    def _copy_tree(node: 'ProcessTree', destpath: Path) -> None:
        """
        Recursively traverse the ProcessTree, find the CalcJobNode and extract its input files to the local directory.

        :param node: The current ProcessTree node.
        :param current_path: The corresponding directory of the current node in the local file system.
        """
        
        # 1. Create the directory of the current node
        # Use the name of the node as the directory name (e.g. 'pw_relax', 'iteration_01')
        node_dir = destpath / node.name

        # 2. Check if the current node is a CalcJobNode
        if node.node.node_type == 'process.calculation.calcjob.CalcJobNode.':
            # Copy the input files of the CalcJobNode to the destination directory
            node_dir.mkdir(parents=True, exist_ok=True)
            
            calcjob_node = node.node
            calcjob_node.base.repository.copy_tree(node_dir)
            calcjob_node.outputs.retrieved.copy_tree(node_dir)
            
        # 3. Recursively process the child nodes
        for child_node in node.children.values():
            ProcessTree._copy_tree(child_node, node_dir)

    def copy_tree(self, destpath: Path) -> Path:
        """
        Extract the input files of all CalcJobNodes from the entire ProcessTree and save them to the local directory.

        :param root_directory_name: The name of the root directory in the local file system.
        :return: The Path object of the created root directory in the local file system.
        """
        
        print(f"Starting extraction to directory: {destpath.resolve()}")
        
        # Start the recursion. From the child nodes of the root node, and use root_path as the parent directory for these child nodes.
        for child_node in self.children.values():
            self._copy_tree(child_node, destpath)
            
        print("Extraction complete.")
        return destpath

class BaseWorkChainAnalyser(ABC):
    """
    BaseAnalyser for the WorkChain.
    """


    def __init__(self, workchain: orm.WorkChainNode):
        self.node = workchain
        self.descendants = {}

    @staticmethod
    @abstractmethod
    def base_check(
        workchain: orm.WorkChainNode,
        excepted_state,
        failed_state,
        killed_state,
        finished_ok_state,
        namespace: str,
        ):
        pass

    @staticmethod
    def get_calcjob_paths(processes_dict, parent_label=''):
        """
        Recursively extract all CalcJob remote paths from the nested dictionary created by get_processes_dict.

        :param processes_dict: The dictionary generated by get_processes_dict.
        :param parent_label: The parent path for building hierarchical labels (used internally for recursion).
        :return: A flattened dictionary { 'full label': 'remote path' }.
        """
        flat_paths = {}
        for label, sub_dict in processes_dict.items():
            if not isinstance(sub_dict, dict):
                continue
            full_label = f"{parent_label}/{label}" if parent_label else label

            if 'calcjob_node' in sub_dict:
                calcjob = sub_dict['calcjob_node']
                remote_path = calcjob.outputs.remote_folder.get_remote_path()
                flat_paths[full_label] = remote_path

            if 'workchain_node' in sub_dict:
                # Pass the current workchain's subprocess dictionary and the new parent label
                nested_paths = BaseWorkChainAnalyser.get_calcjob_paths(
                    sub_dict,
                    parent_label=full_label
                )
                flat_paths.update(nested_paths)

        return flat_paths

    @property
    def process_tree(self):
        """Get the ProcessTree of the workchain."""
        return ProcessTree(self.node)

    @staticmethod
    def get_retrieved(node):
        """Get the retrieved of the all workchains."""
        retrieved = {}

        for subprocess in node.called:
            if 'CalcJobNode' in subprocess.node_type:
                link_label = subprocess.base.attributes.all['metadata_inputs']['metadata']['call_link_label']
                retrieved[link_label] = subprocess.outputs.retrieved if subprocess.outputs.retrieved else None

            elif 'WorkChainNode' in subprocess.node_type:
                link_label = subprocess.base.attributes.all['metadata_inputs']['metadata']['call_link_label']
                retrieved[link_label] = {}
                sub_paths = BaseWorkChainAnalyser.get_retrieved(subprocess)
                retrieved[link_label].update(sub_paths)
            else:
                pass
        return retrieved

    def copy_tree(
        self,
        destpath: Path,
        ):
        """Copy the tree of the workchain to the destination directory."""
        self.process_tree.copy_tree(destpath)
        
    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def get_source(self):
        pass

    @staticmethod
    def get_descendants_by_label(
        workchain: orm.WorkChainNode,
        link_label_filter: str
        ) -> orm.WorkChainNode:
        """Get the descendant workchains of the parent workchain by the link label."""
        try:
            return workchain.base.links.get_outgoing(
                link_label_filter=link_label_filter
                ).all()
        except AttributeError:
            return None

    @abstractmethod
    def clean_workchain(self, exempted_states, dry_run=True):
        """Clean the workchain."""

        state, _ = self.check_process_state()
        message = ''
        if state in exempted_states:
            message += 'Please check if you really want to clean this workchain.'
            return message

        cleaned_calcs = clean_workdir(self.node, dry_run=dry_run)
        message += f'Cleaned the workchain {self.node.pk}:\n'
        message += '  ' + ' '.join(map(str, cleaned_calcs)) + '\n'
        message += f'Deleted the workchain {self.node.pk}:\n'
        deleted_nodes, _ = delete_nodes([self.node.pk], dry_run=dry_run)
        message += '  ' + ' '.join(map(str, deleted_nodes))

        return message

    @staticmethod
    def extract_remote_path(node: orm.CalcJobNode) -> str:
        """
        Extract the remote path of the node.
        """
        return f"remote path: {node.outputs.remote_folder.get_remote_path()}"

    def print_remote_paths(self):
        """
        Print the remote paths of the all CalcJobNodes in the process tree.
        """
        self.process_tree.print_nodes_info(target_node_type='process.calculation.calcjob.CalcJobNode.', extractor=self.extract_remote_path)

    @staticmethod
    def extract_retrieved(node: orm.CalcJobNode) -> str:
        """
        Extract the remote path of the node.
        """
        return f"retrieved: {node.outputs.retrieved.uuid}"

    def print_retrieved(self):
        """
        Print the retrieved of the all CalcJobNodes in the process tree.
        """
        self.process_tree.print_nodes_info(target_node_type='process.calculation.calcjob.CalcJobNode.', extractor=self.extract_retrieved)