# This code is part of the epytope distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.
"""
.. module:: Reader
   :synopsis: Module handles reading of files. line reading, FASTA reading, annovar reading
.. moduleauthor:: brachvogel, schubert

"""

import os
import re
import warnings
import pandas as pd
import vcf
from Bio.SeqIO.FastaIO import SimpleFastaParser

from epytope.Core.Peptide import Peptide
from epytope.Core.Variant import Variant, VariationType, MutationSyntax


####################################
#       F A S T A  -  R E A D E R
####################################
def read_fasta(files, in_type=Peptide, id_position=1):
    """
    Generator function:

    Read a (couple of) peptide, protein or rna sequence from a FASTA file.
    User needs to specify the correct type of the underlying sequences. It can
    either be: Peptide, Protein or Transcript (for RNA).

    :param files: A (list) of file names to read in
    :in_type files: list(str) or str
    :param in_type: The type to read in
    :type in_type: :class:`~epytope.Core.Peptide.Peptide` or :class:`~epytope.Core.Transcript.Transcript`
                or :class:`~epytope.Core.Protein.Protein`
    :param int id_position: the position of the id specified counted by |
    :returns: a list of the specified sequence type derived from the FASTA file sequences.
    :rtype: (list(:attr:`in_type`))
    :raises ValueError: if a file is not readable
    """

    if isinstance(files, str):
            files = [files]
    else:
            if any(not os.path.exists(f) for f in files):
                raise ValueError("Specified Files do not exist")

    collect = set()
    # open all specified files:
    for name in files:
        with open(name, 'r') as handle:
            # iterate over all FASTA entries:
            for _id, seq in SimpleFastaParser(handle):
                # generate element:
                try:
                    _id = _id.split("|")[id_position]
                except IndexError:
                   _id = _id

                try:
                    collect.add(in_type(seq.strip().upper(), transcript_id=_id))
                except TypeError:
                    collect.add(in_type(seq.strip().upper()))
    return list(collect)


####################################
#       L I N E  -  R E A D E R
####################################
def read_lines(files, in_type=Peptide):
    """
    Generator function:

    Read a sequence directly from a line. User needs to manually specify the 
    correct type of the underlying data. It can either be:
    Peptide, Protein or Transcript, Allele.

    :param files: a list of strings of absolute file names that are to be read.
    :in_type files: list(str) or str
    :param in_type: Possible in_type are :class:`~epytope.Core.Peptide.Peptide`, :class:`~epytope.Core.Protein.Protein`,
                 :class:`~epytope.Core.Transcript.Transcript`, and :class:`~epytope.Core.Allele.Allele`.
    :type in_type: :class:`~epytope.Core.Peptide.Peptide` or :class:`~epytope.Core.Protein.Protein` or
                :class:`~epytope.Core.Transcript.Transcript` or :class:`~epytope.Core.Allele.Allele`
    :returns: A list of the specified objects
    :rtype: (list(:attr:`in_type`))
    :raises IOError: if a file is not readable
    """

    if isinstance(files, str):
            files = [files]
    else:
            if any(not os.path.exists(f) for f in files):
                raise IOError("Specified Files do not exist")

    collect = set()
    #alternative to using strings is like: cf = getattr(epytope.Core, "Protein"/"Peptide"/"Allele"/...all in core)
    for name in files:
        with open(name, 'r') as handle:
            # iterate over all lines:
            for line in handle:
                # generate element:
                collect.add(in_type(line.strip().upper()))

    return list(collect)


#####################################
#       A N N O V A R  -  R E A D E R
#####################################
def read_annovar_exonic(annovar_file, gene_filter=None, experimentalDesig=None):
    """
    Reads an gene-based ANNOVAR output file and generates :class:`~epytope.Core.Variant.Variant` objects containing
    all annotated :class:`~epytope.Core.Transcript.Transcript` ids an outputs a list :class:`~epytope.Core.Variant.Variant`.

    :param str annovar_file: The path ot the ANNOVAR file
    :param list(str) gene_filter: A list of gene names of interest (only variants associated with these genes
                                  are generated)
    :return: List of :class:`~epytope.Core.Variant.Variants fully annotated
    :rtype: list(:class:`~epytope.Core.Variant.Variant`)
    """

    vars = []
    gene_filter = gene_filter if gene_filter is not None else []

    #fgd3:nm_001083536:exon6:c.g823a:p.v275i,fgd3:nm_001286993:exon6:c.g823a:p.v275i,fgd3:nm_033086:exon6:c.g823a:p.v275i
    #RE = re.compile("\w+:(\w+):exon\d+:c.(\D*)(\d+)_*(\d*)(\D\w*):p.\w+:\D*->\D*:(\D).*?,")
    #RE = re.compile("\w+:(\w+):exon\d+:c.(\D*)(\d+)_*(\d*)(\D\w*):p.(\D*)(\d+)_*(\d*)(\D\w*):(\D).*?,")
    RE = re.compile("((\w+):(\w+):exon\d+:c.\D*(\d+)\D\w*:p.\D*(\d+)\D\w*)")
    type_mapper = {('synonymous', 'snv'): VariationType.SNP,
                   ('nonsynonymous', 'snv'): VariationType.SNP,
                   ('stoploss', 'snv'): VariationType.SNP,
                   ('stopgain', 'snv'): VariationType.SNP,
                   ('nonframeshift', 'deletion'): VariationType.DEL,
                   ('frameshift', 'deletion'): VariationType.FSDEL,
                   ('nonframeshift', 'insertion'): VariationType.INS,
                   ('frameshift', 'insertion'): VariationType.FSINS}
    with open(annovar_file, "r") as f:
        for line in f:
            mut_id, mut_type, line, chrom, genome_start, genome_stop, ref, alt, zygos = [x.strip().lower() for x in line.split("\t")[:9]]
            #print ref, alt

            #test if its a intersting snp

            gene = line.split(":")[0].strip().upper()

            if gene not in gene_filter and len(gene_filter):
                continue

            if gene == "UNKNOWN":
                warnings.warn("Skipping UNKWON gene")
                continue

           # print "Debug ", gene, type.split(),mut_id
            #print "Debug ", line, RE.findall(line), type, zygos
            coding = {}
            for nm_id_pos in RE.findall(line):
                mutation_string, geneID, nm_id, trans_pos, prot_start = nm_id_pos
                #print "Debug ",nm_id_pos

                nm_id = nm_id.upper()
                _,_, _, trans_coding, prot_coding = mutation_string.split(":")
                #internal transcript and protein position start at 0!
                coding[nm_id] = MutationSyntax(nm_id, int(trans_pos)-1, int(prot_start)-1, trans_coding, prot_coding,
                                               geneID=geneID.upper())

            ty = tuple(mut_type.split())

            vars.append(
                Variant(mut_id, type_mapper.get(ty, VariationType.UNKNOWN), chrom, int(genome_start), ref.upper(),
                        alt.upper(), coding, zygos == "hom", ty[0] == "synonymous",
                        experimentalDesign=experimentalDesig))
    return vars


#####################################
#       V C F  -  R E A D E R
#####################################

def read_vcf(vcf_file, gene_filter=None, experimentalDesig=None):
    """
    Reads an vcf v4.0 or 4.1 file and generates :class:`~epytope.Core.Variant.Variant` objects containing
    all annotated :class:`~epytope.Core.Transcript.Transcript` ids an outputs a list :class:`~epytope.Core.Variant.Variant`.
    Only the following variants are considered by the reader where synonymous labeled variants will not be integrated into any variant:
    filter_variants = ['missense_variant', 'frameshift_variant', 'stop_gained', 'missense_variant&splice_region_variant', "synonymous_variant", "inframe_deletion", "inframe_insertion"]

    :param str vcf_file: The path ot the vcf file
    :param list(str) gene_filter: A list of gene names of interest (only variants associated with these genes
                                  are generated)
    :return: List of :class:`~epytope.Core.Variant.Variants fully annotated
    :rtype: Tuple of (list(:class:`~epytope.Core.Variant.Variant`), list(transcript_ids)
    """
    vl = list()
    with open(vcf_file, 'rb') as tsvfile:
        vcf_reader = vcf.Reader(open(vcf_file, 'r'))
        vl = [r for r in vcf_reader]

    list_vars = []
    transcript_ids = []

    genotye_dict = {"het": False, "hom": True, "ref": True}

    for num, record in enumerate(vl):
        c = record.CHROM.strip('chr')  # chrom
        p = record.POS - 1  # vcf is 1-based & epytope 0-based
        variation_dbid = record.ID  # e.g. rs0123
        r = str(record.REF)  # reference nuc (seq)
        v_list = record.ALT  # list of variants
        q = record.QUAL  # ?
        f = record.FILTER  # empty if PASS, content otherwise
        # I guess we shouldn't expect that keyword to be there ?!
        #z = record.INFO['SOMATIC'] #if true somatic

        vt = VariationType.UNKNOWN
        if record.is_snp:
            vt = VariationType.SNP
        elif record.is_indel:
            if len(v_list)%3 == 0:  # no frameshift
                if record.is_deletion:
                    vt = VariationType.DEL
                else:
                    vt = VariationType.INS
            else:  # frameshift
                if record.is_deletion:
                    vt = VariationType.FSDEL
                else:
                    vt = VariationType.FSINS
        gene = None

        # WHICH VARIANTS TO FILTER ?
        filter_variants = ['missense_variant', 'frameshift_variant', 'stop_gained', 'missense_variant&splice_region_variant', "synonymous_variant", "inframe_deletion", "inframe_insertion"]

        for alt in v_list:
            isHomozygous = False
            if 'HOM' in record.INFO:
                #TODO set by AF & FILTER as soon as available
                isHomozygous = record.INFO['HOM'] == 1
            elif 'SGT' in record.INFO:
                zygosity = record.INFO['SGT'].split("->")[1]
                if zygosity in genotye_dict:
                    isHomozygous = genotye_dict[zygosity]
                else:
                    if zygosity[0] == zygosity[1]:
                        isHomozygous = True
                    else:
                        isHomozygous = False
            else:
                for sample in record.samples:
                    if 'GT' in sample.data:
                        isHomozygous = sample.data['GT'] == '1/1'

            if "ANN" in record.INFO and record.INFO['ANN']:
                isSynonymous = False
                coding = dict()
                for annraw in record.INFO['ANN']:  # for each ANN only add a new coding! see GSvar
                    annots = annraw.split('|')

                    obs, a_mut_type, impact, a_gene, a_gene_id, feature_type, transcript_id, exon, tot_exon, trans_coding, prot_coding, cdna, cds, aa, distance, warns = annots

                    if a_mut_type in filter_variants:
                        tpos = 0
                        ppos = 0

                        # get cds/protein positions and convert mutation syntax to epytope format
                        if trans_coding != '':
                            positions = re.findall(r'\d+', trans_coding)
                            ppos = int(positions[0]) - 1

                        if prot_coding != '':
                            positions = re.findall(r'\d+', prot_coding)
                            tpos = int(positions[0]) - 1

                        isSynonymous = (a_mut_type == "synonymous_variant")

                        #rather take gene_id than gene name
                        gene = a_gene_id

                        #REFSEQ specific ? Do have to split because of biomart ?
                        transcript_id = transcript_id.split(".")[0]

                        #TODO vcf are not REFSEQ only

                        #coding string not parsed anyway ? just use the one given by SnpEff
                        coding[transcript_id] = MutationSyntax(transcript_id, ppos, tpos, trans_coding, prot_coding)
                        transcript_ids.append(transcript_id)

                if coding and not isSynonymous:
                    if vt == VariationType.SNP:
                        pos, reference, alternative = p, str(r), str(alt)
                    elif vt == VariationType.DEL or vt == VariationType.FSDEL:
                        if alt != '-':
                            pos, reference, alternative = p + len(alt), r[len(alt):], '-'
                        else:
                            pos, reference, alternative = p, str(r), str(alt)
                    elif vt == VariationType.INS or vt == VariationType.FSINS:
                        if r != '-':
                            if alt != '-':
                                pos, reference, alternative = p + len(r), '-', str(alt)[len(r):]
                            else:
                                pos, reference, alternative = p + len(r), '-', str(alt)
                        else:
                            pos, reference, alternative = p, str(r), str(alt)

                    var = Variant("line" + str(num), vt, c, pos, reference, alternative, coding, isHomozygous, isSynonymous, experimentalDesign=experimentalDesig)
                    var.gene = gene
                    var.log_metadata("vardbid", variation_dbid)
                    list_vars.append(var)

            else:
                warnings.warn("Skipping unannotated variant", UserWarning)

    return list_vars, transcript_ids


def process_dataset_TCR(path: str = None, df: pd.DataFrame = None, source: str = None, score: int = 1) \
        -> pd.DataFrame:
    """
    can read and process four different datasets [vdjdb, McPAS, scirpy, IEDB] or a csv file with fixed column names
    dataset.columns = ["Receptor_ID", 'TRA', "TRA_nt", 'TRB', "TRB_nt", "TRAV", "TRAJ", "TRBV", "TRBJ",
    "T-Cell-Type", "Peptide", "MHC", "species", "Antigen.species", "Tissue"] If some values for one or more variables
    are unavailable, leave them as blank cells
    :param int score: An integer representing a confidence score between 0 and 3 (0: critical information missing,
    1: medium confidence, 2: high confidence, 3: very high confidence). By processing all entries with a confidence
    score >= the passed parameter score will be kept. Default value is 1
    :param str path: a string representing a path to the dataset(csv file), which will be processed. Default value is
     None, when the DataFrame object is given.
    :param `pd.DataFrame` df: a dataframe object representing the dataset, which will be processed. Default value
    None, when the path is given
    :param str source: the source of the dataset [vdjdb, McPAS, scirpy, IEDB]. If this parameter does not be passed, the
    dataset should be a csv file with the column names mentioned above
    :return: Returns a dataframe with the following header:
    ["Receptor_ID", 'TRA', "TRA_nt", 'TRB', "TRB_nt", "TRAV", "TRAJ", "TRBV", "TRBJ", "T-Cell-Type", "Peptide",
    "MHC", "species", "Antigen.species", "Tissue"]
    :rtype: `pd.DataFrame`
    """

    def invalid(seq: str) -> bool:
        """
        helper function to check if the passed sequence is an invalid protein sequence
        :param str seq: a String representing the protein sequence
        :return: Returns true if the passed sequence is not a protein sequence
        :rtype: bool
        """
        aas = set("ARNDCEQGHILKMFPSTWYV")
        if seq:
            return any([aa not in aas for aa in seq])
        return True

    def process(df: pd.DataFrame, source: str = None) -> pd.DataFrame:
        """
        helper function to check for invalid protein sequences in upper case.
        All rows with invalid cdr3 beta seqs will be removed, whereas invalid cdr3 alpha seqs will be replaced with an
        empty string
        :param df: a dataframe, which will be processed
        :param str source: the source of the dataset [vdjdb, McPAS, scirpy, IEDB].
        :return: returns the processed dataframe
        :rtype: `pd.DataFrame`
        """
        if "MHC" in df.columns:
            df.loc[:, "MHC"] = df["MHC"].apply(lambda x: re.search(r".*?(?=:)|.*", str(x)).group())
        df["TRA"] = df["TRA"].str.upper()
        df["TRB"] = df["TRB"].str.upper()
        if source != "scirpy" and "Peptide" in df.columns:
            df["Peptide"] = df["Peptide"].str.upper()
            df = df[df["Peptide"].apply(lambda x: not invalid(str(x)))]
        # keep only rows, where cdr3 beta sequences consist only of amino acid characters
        df = df[df["TRB"].apply(lambda x: not invalid(str(x)))]
        # if TCR alpha seq is not a protein seq leave the cell blank
        df.loc[:, "TRA"] = df["TRA"].apply(lambda x: x if not invalid(str(x)) else "")
        return df

    if path is None and df is None:
        raise FileNotFoundError("A path to a csv file or a dataframe should be passed")
    # reading and processing the vdjdb
    if source and source.lower() == "vdjdb":
        if df is None:
            df = pd.read_csv(path, sep='\t', low_memory=False)
            pd.options.mode.chained_assignment = None
        # remove duplicates that share the same cdr3.alpha, cdr3.beta and epitope seqs and keep only entries with score
        # equal greater than the passed score
        if score in range(4):
            df = df[df["vdjdb.score"] >= score].drop_duplicates(
                subset=["cdr3.alpha", "cdr3.beta", "antigen.epitope"],
                keep='first').reset_index(drop=True)

        # select only the columns mentioned in the description above
        df = df[["meta.clone.id", "cdr3.alpha", "cdr3.beta", "v.alpha", "j.alpha", "v.beta", "j.beta",
                 "meta.cell.subset", "antigen.epitope", "mhc.a", "species", "antigen.species", "meta.tissue"]]
        # rename the selected columns
        df.columns = ["Receptor_ID", 'TRA', 'TRB', "TRAV", "TRAJ", "TRBV", "TRBJ", "T-Cell-Type", "Peptide", "MHC",
                      "Species", "Antigen.species", "Tissue"]
        df = df.reindex(columns=df.columns.tolist() + ["TRA_nt", "TRB_nt"])
        # replace not available values with empty cells
        df = df.fillna('')
        df.loc[:, "T-Cell-Type"] = df["T-Cell-Type"].apply(lambda x: x[:3])
        return process(df)

    # reading and processing the McPAS
    elif source and source.upper() == "MCPAS":
        if df is None:
            df = pd.read_csv(path, sep=",", encoding='cp1252', low_memory=False)
        df = df[["CDR3.alpha.aa", "CDR3.beta.aa", "TRAV", "TRAJ", "TRBV", "TRBJ", "T.Cell.Type", "Epitope.peptide",
                 "MHC", "Species", "Pathology", "Tissue", "CDR3.alpha.nt", "CDR3.beta.nt"]]
        df.columns = ['TRA', 'TRB', "TRAV", "TRAJ", "TRBV", "TRBJ", "T-Cell-Type", "Peptide", "MHC", "Species",
                      "Antigen.species", "Tissue", "TRA_nt", "TRB_nt"]
        df.insert(0, "Receptor_ID", [i for i in range(len(df))])
        # replace not available values with empty cells
        df = df.fillna('')
        df.drop_duplicates(subset=["TRA", "TRB", "Peptide"], keep='first', inplace=True)
        return process(df)

    elif source and source.lower() == "scirpy":
        if df is None:
            df = pd.read_csv(path, low_memory=False)
        df.reset_index(inplace=True)
        # different versions of scirpy have different column names of scirpy datasets, so we have to deal with
        # that
        if "IR_VJ_1_cdr3" in df.columns:
            df = df[["IR_VJ_1_cdr3", "IR_VDJ_1_cdr3", "IR_VJ_1_v_gene", "IR_VJ_1_j_gene", "IR_VDJ_1_v_gene",
                     "IR_VDJ_1_j_gene", "IR_VJ_1_cdr3_nt", "IR_VDJ_1_cdr3_nt"]]
        else:
            df = df[["IR_VJ_1_junction_aa", "IR_VDJ_1_junction_aa", "IR_VJ_1_v_call", "IR_VJ_1_j_call",
                     "IR_VDJ_1_v_call", "IR_VDJ_1_j_call", "IR_VJ_1_junction", "IR_VDJ_1_junction"]]
        df = df.reindex(columns=df.columns.tolist() + ["T-Cell-Type", "Peptide", "MHC", "Species", "Antigen.species",
                                                       "Tissue"])
        df.columns = ['TRA', 'TRB', "TRAV", "TRAJ", "TRBV", "TRBJ", "TRA_nt", "TRB_nt", "T-Cell-Type", "Peptide", "MHC",
                      "Species", "Antigen.species", "Tissue"]
        df.insert(0, "Receptor_ID", [i for i in range(len(df))])
        df.replace("None", "", inplace=True)
        df.replace("nan", "", inplace=True)
        df.fillna("", inplace=True)
        df.drop_duplicates(subset=["TRA", "TRB"], keep='first', inplace=True)
        return process(df, source="scirpy")

    elif source and source.upper() == "IEDB":
        if df is None:
            df = pd.read_csv(path, sep=",", low_memory=False)
        df = df[["Receptor ID", 'Chain 1 CDR3 Calculated', 'Chain 2 CDR3 Calculated', 'Calculated Chain 1 V Gene',
                'Calculated Chain 1 J Gene', 'Calculated Chain 2 V Gene', 'Calculated Chain 2 J Gene', 'Description',
                 'MHC Allele Names', 'Organism']]
        df["Description"] = df["Description"].apply(lambda x: x.split()[0]
                                                        if x.split()[0].isupper() and not invalid(x.split()[0]) else "")
        df.insert(7, "T-Cell-Type", "")
        df.insert(10, "Species", "")
        df = df.reindex(columns=df.columns.tolist() + ["Tissue", "TRA_nt", "TRB_nt"])
        df.columns = ["Receptor_ID", 'TRA', 'TRB', "TRAV", "TRAJ", "TRBV", "TRBJ", "T-Cell-Type", "Peptide", "MHC",
                      "Species", "Antigen.species", "Tissue", "TRA_nt", "TRB_nt"]
        df[["TRA", "TRB", "TRAV", "TRAJ", "TRBV", "TRBJ"]] = \
            df[["TRA", "TRB", "TRAV", "TRAJ", "TRBV", "TRBJ"]].replace("nan", "")
        df.fillna("", inplace=True)
        df.drop_duplicates(subset=["TRA", "TRB", "Peptide"], keep='first', inplace=True)
        df = df[df["TRB"] != ""]
        return process(df)
    elif source and source.lower() == "dash":
        if df is None:
            if path is None or not os.path.isfile(path):
                raise FileNotFoundError(f"{path} is not a right path to a csv file")
            df = pd.read_csv(path, sep=",")
        df = df[['cdr3_a_aa', 'cdr3_b_aa', 'epitope', 'v_a_gene', 'j_a_gene', 'cdr3_a_nucseq', 'v_b_gene',
                 'j_b_gene', 'cdr3_b_nucseq']]
        df.columns = ["TRA", "TRB", "Peptide", "TRAV", "TRAJ", "TRA_nt", "TRBV", "TRBJ", "TRB_nt"]
        df.drop_duplicates(subset=["TRA", "TRB"], keep="first", inplace=True)
        df = df.reindex(columns=df.columns.tolist() + ["Receptor_ID", "T-Cell-Type", "MHC", "Species",
                                                       "Antigen.species", "Tissue"])
        df.fillna("", inplace=True)
        df.loc[:, "Receptor_ID"] = [i for i in range(len(df))]
        return df
    else:
        if df is None:
            df = pd.read_csv(path, sep=",")
        df.fillna("", inplace=True)
        return process(df)
