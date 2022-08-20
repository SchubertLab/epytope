# This code is part of the epytope distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.
__author__ = 'walzer', 'haegele', 'schubert', 'szolek'

from epytope.IO.FileReader import read_annovar_exonic,read_fasta,read_lines
from epytope.IO.MartsAdapter import MartsAdapter
from epytope.IO.RefSeqAdapter import RefSeqAdapter
from epytope.IO.UniProtAdapter import UniProtDB
from epytope.IO.EnsemblAdapter import EnsemblDB
from epytope.IO.ADBAdapter import EIdentifierTypes, EAdapterFields
