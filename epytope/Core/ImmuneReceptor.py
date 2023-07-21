# This code is part of the epytope distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.
"""
.. module:: Core.AdaptiveImmuneReceptor
   :synopsis: Contains the Adaptive Immune Receptor class (TCR, BCR)
   :Note: All internal indices start at 0!
.. moduleauthor:: schubert
"""

from epytope.Core.Base import MetadataLogger
from epytope.Core.ImmuneReceptorChain import ImmuneReceptorChain
from typing import List


class ReceptorFactory(type):
    def __call__(cls, receptor_id: str, chains: List, t_cell: bool = True,  cell_type: str = None, species: str = None,
                 tissue: str = None):
        if cls is AntigenImmuneReceptor:
            if t_cell:
                return TCellReceptor(receptor_id=receptor_id, cell_type=cell_type, chains=chains, species=species,
                                     tissue=tissue)
            else:
                return BCellReceptor(receptor_id=receptor_id, cell_type=cell_type, chains=chains, species=species,
                                     tissue=tissue)
        return type.__call__(cls, receptor_id=receptor_id, cell_type=cell_type, chains=chains, species=species,
                             tissue=tissue)


class AntigenImmuneReceptor(MetadataLogger, metaclass=ReceptorFactory):
    """
    :class:`~epytope.Core.AdaptiveImmuneReceptor.AdaptiveImmuneReceptor` corresponding to exactly one T or B cell
     receptor.
    """

    def __init__(self, receptor_id: str, chains: List, t_cell: bool = True, cell_type:str = None, species: str = None,
                 tissue: str = None):
        """
        :param str receptor_id: a string representing the receptor id
        :param str cell_type: a String representing the type of the B respectively T cell e.g: CD8, IGM
        :param list chains: a list of chains of which the T respectively B cell receptor consists each chain has
        class: `epytope.Core.ImmuneReceptorChain.ImmuneReceptorChain` type
        :param bool t_cell: true if the object is a t cell receptor otherwise it is a B cell receptor, default True
        :param str species: a string representing the species the B respectively T cell receptor originated from
        :param str tissue: a string representing the tissue used to isolate the receptor
        """
        # Init parent type:
        MetadataLogger.__init__(self)
        self.receptor_id = receptor_id
        self.cell_type = cell_type
        self.species = species
        self.chains = [chain for chain in chains if isinstance(chain, ImmuneReceptorChain)]
        self.tissue = tissue

    def __repr__(self):
        return "\n\n".join([str(chain) for chain in self.chains])


class TCellReceptor(AntigenImmuneReceptor):
    """
       This class represents T cell receptor with alpha, beta, gamma and delta chains
    """

    def __init__(self, receptor_id: str, chains: List, cell_type: str = None, species: str = None, tissue: str = None):
        """
        :param str receptor_id: a string representing the receptor id
        :param str cell_type: a String representing the type of the B respectively T cell e.g: CD8, IGM
        :param list chains: a list of chains of which the T respectively B cell receptor consists each chain has
        class: `epytope.Core.ImmuneReceptorChain.ImmuneReceptorChain` type
        :param str species: a string representing the species the T cell receptor originated from
        :param str tissue: a string representing the tissue used to isolate the T-cell
        """
        # Init parent type:
        MetadataLogger.__init__(self)
        self.receptor_id = receptor_id
        self.cell_type = cell_type
        self.tissue= tissue
        self.chains = []
        self.tcr = {}
        self.TRA = None
        self.TRB = None
        self.TRG = None
        self.TRD = None
        self.species = species
        self.tcr["Receptor_ID"] = self.receptor_id
        self.tcr["Species"] = self.species
        self.tcr["T-Cell-Type"] = self.cell_type
        self.tcr["Tissue"] = self.tissue
        for chain in chains:
            if not isinstance(chain, ImmuneReceptorChain):
                raise ValueError(f"{chain} should be an ImmuneReceptorChain object")
            self.chains.append(chain)
            if chain.chain_type == "TRA":
                self.TRA = chain
                self.tcr["TRA"] = str(chain.cdr3)
                self.tcr["TRAV"] = chain.v_gene
                self.tcr["TRAJ"] = chain.j_gene

            elif chain.chain_type == "TRB":
                self.TRB = chain
                self.tcr["TRB"] = str(chain.cdr3)
                self.tcr["TRBV"] = chain.v_gene
                self.tcr["TRBD"] = chain.d_gene
                self.tcr["TRBJ"] = chain.j_gene

            elif chain.chain_type == "TRG":
                self.TRG = chain
            else:
                self.TRD = chain


class BCellReceptor(AntigenImmuneReceptor):
    """
        This class represents B cell receptor with light, heavy and kappa chains
    """

    def __init__(self, receptor_id: str, chains: List, cell_type: str = None, species: str = None, tissue: str = None):
        """
        :param str receptor_id: a string representing the receptor id
        :param str cell_type: a String representing the type of the B respectively T cell e.g: CD8, IGM
        :param list chains: a list of chains of which the T respectively B cell receptor consists each chain has
        class: `epytope.Core.ImmuneReceptorChain.ImmuneReceptorChain` type
        :param str species: a string representing the species the B cell receptor originated from
        :param str tissue: a string representing the tissue used to isolate the B-cell
        """
        # Init parent type:
        MetadataLogger.__init__(self)
        self.receptor_id = receptor_id
        self.cell_type = cell_type
        self.chains = []
        self.tissue = tissue
        for chain in chains:
            if not isinstance(chain, ImmuneReceptorChain):
                raise ValueError(f"{chain} should be an ImmuneReceptorChain object")
            self.chains.append(chain)
        self.IGK = None
        self.IGH = None
        self.IGL = None
        self.species = species
        for chain in self.chains:
            if chain.chain_type == "IGK":
                self.IGK = chain
            elif chain.chain_type == "IGH":
                self.IGH = chain
            else:
                self.IGL = chain
