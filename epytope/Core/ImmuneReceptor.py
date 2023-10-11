# This code is part of the epytope distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.
"""
.. module:: Core.ImmuneReceptor
   :synopsis: Contains the Immune Receptor classes for T cell receptors (TCRs) and B cell receptors (BCRs)
   :Note: All internal indices start at 0!
.. moduleauthor:: albahah, drost
"""

from epytope.Core.Base import MetadataLogger


class ImmuneReceptorFactory(type):
    def __call__(cls, receptor_chains, cell_type=None, organism=None):
        if cls is ImmuneReceptor:
            if cell_type is None:
                pass
            elif "T cell" in cell_type:
                return TCellReceptor(receptor_chains, cell_type, organism)
            elif "B cell" in cell_type:
                return BCellReceptor(receptor_chains, cell_type, organism)
        return type.__call__(cls, receptor_chains, cell_type, organism)


class ImmuneReceptor(MetadataLogger, metaclass=ImmuneReceptorFactory):
    """
    This class represents a Immune receptor (TCR or BCR) and stores additional information.
    """

    def __init__(self, receptor_chains, cell_type=None, organism=None):
        """
        :param receptor_chains: List containing the immune receptor chains (either alpha-beta, gamma-delta, heavy-light)
        :type receptor_chains: list(:class: `~epytope.Core.ImmuneReceptorChain.ImmuneReceptorChain`)
        :param str cell_type: cell type in which the IR was found (e.g. CD8 T cell, kappa B cell)
        :param str organism: origin of the immune receptor
        """
        MetadataLogger.__init__(self)
        self.cell_type = cell_type if cell_type is not None else 'Immune cell'
        self.organism = organism

        self.chains = {}
        for chain in receptor_chains:
            chain_type = chain.chain_type
            if chain_type in self.chains:
                print(f"Duplicated chains of same type detected in {receptor_chains}")
            else:
                self.chains[chain_type] = chain

    def get_vj_chain(self):
        for name in ['TRA', 'TRG', 'IL']:
            if name in self.chains:
                return self.chains[name]
        return None

    def get_vdj_chain(self):
        for name in ['TRB', 'TRD', 'IH']:
            if name in self.chains:
                return self.chains[name]
        return None

    def get_chain(self, chain_type):
        if chain_type == "VJ":
            return self.get_vj_chain()
        if chain_type == "VDJ":
            return self.get_vdj_chain()
        raise ValueError(f"{chain_type} not in ['VJ', 'VDJ']")

    def get_chain_attribute(self, attribute, chain_type):
        chain = self.get_chain(chain_type)
        if chain is None:
            return ""
        return getattr(chain, attribute)

    def __repr__(self):
        lines = [f"{self.cell_type.upper()} RECEPTOR".strip()]
        if self.organism:
            lines.append(f"in {self.organism}")
        for chain in self.chains.values():
            lines.append(" " + str(chain).replace('\n', '\n '))
        return "\n".join(lines)

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))


class TCellReceptor(ImmuneReceptor):
    """
       This class represents T cell receptor with alpha, beta, gamma and delta chains.
    """

    def __init__(self, receptor_chains, cell_type=None, organism=None):
        """
        :param receptor_chains: List containing the immune receptor chains (either alpha-beta or gamma-delta)
        :type receptor_chains: list(:class: `~epytope.Core.ImmuneReceptorChain.ImmuneReceptorChain`)
        :param str cell_type: cell type in which the IR was found (e.g. CD8 T cell)
        :param str organism: origin of the immune receptor
        """
        ImmuneReceptor.__init__(self, receptor_chains, "T cell", organism)

        self.cd4_cd8_type = "CD4" if "CD4" in cell_type.upper() else "CD8" if "CD8" in cell_type.upper() else ''
        self.alphabeta_gamadelta_type = self.assign_alphabeta_gammadelta_type()
        self.cell_type = " ".join([self.cd4_cd8_type, self.alphabeta_gamadelta_type, self.cell_type]).strip()

    def assign_alphabeta_gammadelta_type(self):
        """
        Determine whether the TCR is of alpha-beta, gamma-delta, or mixed type based on its chains.
        :return: str of ['alpha-beta', 'gamma-delta', 'conflicting']
        """
        alphabeta = ("TRA" in self.chains) or ("TRB" in self.chains)
        gammadelta = ("TRG" in self.chains) or ("TRD" in self.chains)
        if alphabeta and not gammadelta:
            return "alpha-beta"
        if not alphabeta and gammadelta:
            return "gamma-delta"
        return "conflicting"


class BCellReceptor(ImmuneReceptor):
    """
        This class represents B cell receptor with light (lambda or kappa) and heavy chains.
    """

    def __init__(self, receptor_chains, cell_type=None, organism=None):
        """
        :param receptor_chains: List containing the immune receptor chains (heavy-light)
        :type receptor_chains: list(:class: `~epytope.Core.ImmuneReceptorChain.ImmuneReceptorChain`)
        :param str cell_type: cell type in which the IR was found (e.g. kappa B cell)
        :param str organism: origin of the immune receptor
        """
        MetadataLogger.__init__(self)
        ImmuneReceptor.__init__(self, receptor_chains, "B cell", organism)
        self.kappa_lambda_type = "kappa" if "kappa" in cell_type.lower() else "lambda" \
            if "lambda" in cell_type.lower() else None
        self.cell_type = " ".join([self.kappa_lambda_type, self.cell_type])
