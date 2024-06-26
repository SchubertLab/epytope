# This code is part of the epytope distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.
"""
.. module:: IO.MartsAdapter
   :synopsis: BDB-Adapter for BioMart
.. moduleauthor:: walzer, schubert
"""

from imghdr import tests
import warnings
import logging
import pymysql.cursors
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import io
import pandas as pd
import re
import sys
import xml.etree.ElementTree as ElementTree
from bs4 import BeautifulSoup
from enum import Enum
from html.parser import HTMLParser
from epytope.IO.ADBAdapter import ADBAdapter, EAdapterFields, EIdentifierTypes


class EnsemblMartAttributes(Enum):
    ENSEMBL_HSAPIENS_DATASET = "hsapiens_gene_ensembl"
    ENSEMBL_GENE_CONFIG = "gene_ensembl_config"
    ENSEMBL_GENE_ID = "ensembl_gene_id"
    ENSEMBL_GENE_ID_VERSION = "ensembl_gene_id_version"
    ENSEMBL_TRANSCRIPT_ID = "ensembl_transcript_id"
    ENSEMBL_TRANSCRIPT_ID_VERSION = "ensembl_transcript_id_version"
    ENSEMBL_PEPTIDE_ID = "ensembl_peptide_id"
    ENSEMBL_PEPTIDE_ID_VERSION = "ensembl_peptide_id_version"
    EXTERNAL_GENE_NAME = "external_gene_name"
    REFSEQ_PEPTIDE = "refseq_peptide"
    REFSEQ_MRNA_PREDICTED = "refseq_mrna_predicted"
    REFSEQ_PEPTIDE_PREDICTED = "refseq_peptide_predicted"
    REFSEQ_MRNA = "refseq_mrna"


class MartsAdapter(ADBAdapter):
    def __init__(self, usr=None, host=None, pwd=None, db=None, biomart="https://www.ensembl.org/biomart/martservice", mart_properties=None):
        """
        Used to fetch sequences from given RefSeq id's either from BioMart if no credentials given else from a MySQLdb
co
        :param str usr: db user e.g. = 'ucsc_annot_query'
        :param str host: db host e.g. = "pride"
        :param str pwd: pw for user e.g. = 'an0q3ry'
        :param str db: db on host e.g. = "hg18_ucsc_annotation"
        :param str biomart: Ensembl page url e.g. = "https://www.ensembl.org/"
        :param dict mart_properties: details for mart query e.g. = ""header":"1""

        """
        self.ids_proxy = dict()
        self.gene_proxy = dict()
        self.sequence_proxy = dict()
        self.biomart_archive_url = "https://www.ensembl.org/info/website/archives/index.html?redirect=no"
        self.biomart_head = {"virtualSchemaName": "default", "formatter": "TSV",
                            "header": "1", "uniqueRows": "0", "datasetConfigVersion": "0.6"}

        if usr and host and pwd and db:
            self.connection = pymysql.connect(
                user=usr, host=host, password=pwd, db=db)
        else:
            self.connection = None

        self.biomart_url = biomart
        if not self.biomart_url.endswith("/biomart/martservice"):
            self.biomart_url += "/biomart/martservice"

        if mart_properties:
            self.biomart_head.update(mart_properties)

    def __create_biomart_header_xml(self, mart_properties):
        """
        Create the biomart header with given properties

        :param dict mart_properties: The fields and values for the header 

        :return: The biomart header as XML
        :rtype: ElementTree.Element
        """
        # XML header will be added before query is sent
        biomart_header = ElementTree.Element("Query")
        biomart_header.attrib.update(mart_properties)
        return biomart_header

    def __add_attribute(self, parent, attribute_name, attribute_value, name="Attribute"):
        """
        Adds an attribute to the ElementTree SubElement

        :param ElementTree parent: The parent element (which is getting modified)
        :param str attribute_name: The name of the attribute
        :param str attribute_value: The value of the attribute
        :param str name: The name of the XML field (default: Attribute)

        :return: The SubElement with attribute
        :rtype: ElementTree.SubElement
        """
        element = ElementTree.SubElement(parent, name)
        element.set(attribute_name, attribute_value)
        return element

    def __add_filter(self, parent, filter_id, id, filter_value, value, name="Filter"):
        """
        Adds a filter to the ElementTree SubElement

        :param ElementTree parent: The parent element (which is getting modified)
        :param str filter_id: The name of the filter name field
        :param str id: The id of the filter
        :param str filter_value: The name of the filter value field
        :param str value: The value of the filter
        :param str name: The name of the XML field (default: Attribute)

        :return: The SubElement with attribute
        :rtype: ElementTree.SubElement
        """
        element = ElementTree.SubElement(parent, name)
        element.set(filter_id, id)
        element.set(filter_value, value)
        return element

    def __search_for_resources(self, xml_root):
        """
        Perform search (GET) using the given XML as query value

        :param ElementTree Element xml_root: The XML of the query

        :return: The found resources as data frame
        :rtype: pandas.core.frame.DataFrame
        """
        f = io.BytesIO()
        et = ElementTree.ElementTree(xml_root)
        et.write(f, encoding="UTF-8", xml_declaration=True)
        response = self.__perform_request(self.biomart_url, {"query": f.getvalue().decode("utf8")})
        result = pd.read_csv(io.StringIO(response.content.decode('utf-8')), delimiter='\t')
        return result

    def __chunks(self, lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def __perform_request(self, url, params):
        """
        Perform a request (GET) on the given url and parameters. 
        Request will be retried three times on specified errors.

        :return: The response
        :rtype: request.response
        """
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            backoff_factor=3
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        request = requests.Session()
        request.mount("https://", adapter)
        response = request.get(url, params=params)
        request.close()
        return response

    def list_archives(self):
        """
        Fetches the available Ensembl archives from the ensembl website

        :return: The available Ensembl archives
        :rtype: pandas.core.frame.DataFrame
        """
        html_text = requests.get(self.biomart_archive_url).text
        soup = BeautifulSoup(html_text, 'html.parser')

        archives = []
        for archive in soup.find_all('a', string=re.compile(r'http[s]?://.*ensembl\\.org|[A-Z][a-z]{2} [0-9]{4}')):
            url = archive['href']
            version, date = archive.contents[0].split(':')
            archives.append([version, url, date])

        return pd.DataFrame(archives, columns=['name', 'url', 'date'])

    def get_marts(self):
        """
        Fetches the mart registry with information on available marts

        :return: The available marts for this server
        :rtype: pandas.core.frame.DataFrame
        """
        registry = self.__perform_request(self.biomart_url, {"type": "registry"})
        tree = ElementTree.fromstring(registry.content.decode('utf-8'))
        data = []
        for registry in tree:
            data.append([*registry.attrib.values()])
        df = pd.DataFrame.from_dict(data)
        df.columns = [*registry.attrib.keys()]
        return df

    def get_datasets(self, mart_name):
        """
        Fetches available datasets for a given mart name

        :param str mart_name: The name of the mart (e.g. ENSEMBL_MART_ENSEMBL)

        :return: The available datasets for this mart
        :rtype: pandas.core.frame.DataFrame
        """
        datasets = self.__perform_request(self.biomart_url, {"type": "datasets", "mart": mart_name})
        df = pd.read_csv(io.StringIO(
            datasets.content.decode('utf-8')), delimiter='\t', header=None)
        df_slice = df.iloc[:, [1, 2, 4, 7, 8]]
        df_slice.dropna(how='all', inplace=True)
        df_slice.columns = ['dataset_id', 'dataset_name',
                            'short_name', 'interface', 'date']
        return df_slice

    def get_dataset_attributes(self, dataset_name):
        """
        Fetches available attributes for a given dataset of a mart

        :param str dataset_name: The name of the dataset

        :return: The available attributes
        :rtype: pandas.core.frame.DataFrame
        """
        attributes = self.__perform_request(self.biomart_url, {"type": "attributes", "dataset": dataset_name})
        df = pd.read_csv(io.StringIO(
            attributes.content.decode('utf-8')), delimiter='\t')
        df_part = df[df.columns[:3]]
        df_part.columns = ['attribute_id', 'attribute_name', 'description']
        return df_part

    def get_dataset_filters(self, dataset_name):
        """
        Fetches available filters for a given dataset of a mart

        :param str dataset_name: The name of the dataset

        :return: The available filters
        :rtype: pandas.core.frame.DataFrame
        """
        filters = self.__perform_request(self.biomart_url, {"type": "filters", "dataset": dataset_name})
        df = pd.read_csv(io.StringIO(
            filters.content.decode('utf-8')), delimiter='\t')
        df_part = df[df.columns[:-2]]
        df_part.columns = ['filter_id', 'filter_name', 'values',
                        'description', 'type', 'filter_type', 'operator']
        return df_part

    def get_attribute_name_for_id(self, attributes, attribute_id):
        """
        Fetches the attribute name for an attribute id for a given dataset of a mart

        :param pandas.core.frame.DataFrame attributes: The available dataset attributes
        :param str attribute_id: The id of the attribute

        :return: The corresponding attribute name
        :rtype: str
        """
        return attributes.loc[attributes['attribute_id'] == attribute_id, 'attribute_name'].iloc[0]

    def get_attribute_id_for_name(self, attributes, attribute_name):
        """
        Fetches the attribute id for an attribute name for a given dataset of a mart

        :param str dataset_name: The name of the dataset
        :param pandas.core.frame.DataFrame attributes: The available dataset attributes
        :param str attribute_name: The name of the attribute

        :return: The corresponding attribute id
        :rtype: str
        """
        return attributes.loc[attributes['attribute_name'] == attribute_name, 'attribute_id'].iloc[0]

    def get_product_sequence(self, product_id, **kwargs):
        """
        Fetches product (i.e. protein) sequence for the given id

        :param str product_id: The id to be queried
        :keyword type: Assumes given ID from type found in :func:`~epytope.IO.ADBAdapter.EIdentifierTypes`, default is ensembl_peptide_id
        :type type: :func:`~epytope.IO.ADBAdapter.EIdentifierTypes`
        :keyword str _db: Can override MartsAdapter default db ("hsapiens_gene_ensembl")
        :keyword str _dataset: Specifies the query dbs dataset if default is not wanted ("gene_ensembl_config")

        :return: The requested sequence
        :rtype: str
        """
        _db = kwargs.get(
            "_db", EnsemblMartAttributes.ENSEMBL_HSAPIENS_DATASET.value)
        _dataset = kwargs.get(
            "_dataset", EnsemblMartAttributes.ENSEMBL_GENE_CONFIG.value)
        attributes = {"peptide": "",
                      EnsemblMartAttributes.EXTERNAL_GENE_NAME.value: ""}

        with_version = '.' in product_id
        query_filter = EnsemblMartAttributes.ENSEMBL_PEPTIDE_ID_VERSION.value if with_version else EnsemblMartAttributes.ENSEMBL_PEPTIDE_ID.value
        if "type" in kwargs:
            if kwargs["type"] == EIdentifierTypes.REFSEQ:
                query_filter = EnsemblMartAttributes.REFSEQ_PEPTIDE.value
            elif kwargs["type"] == EIdentifierTypes.PREDREFSEQ:
                query_filter = EnsemblMartAttributes.REFSEQ_PEPTIDE_PREDICTED.value
            elif kwargs["type"] == EIdentifierTypes.ENSEMBL:
                pass
            else:
                logging.warning(
                    "Could not infer the origin of product id " + str(product_id))
                return None

        if product_id in self.sequence_proxy:
            return self.sequence_proxy[product_id]

        dataset_attributes = self.get_dataset_attributes(_db)
        root = self.__create_biomart_header_xml(self.biomart_head)
        dataset = ElementTree.SubElement(root, "Dataset")
        dataset.attrib.update({"name": _db, "interface": "default"})
        self.__add_filter(dataset, "name", query_filter,
                          "value", str(product_id))
        for attribute in attributes:
            self.__add_attribute(dataset, "name", attribute)
            try:
                attribute_name = self.get_attribute_name_for_id(dataset_attributes, attribute)
                attributes.update({attribute: attribute_name})
            except:
                logging.error("Attribute {} not found for dataset {} on {}".format(
                    attribute, _db, self.biomart_url))
                sys.exit(1)

        result = self.__search_for_resources(root)

        if result.empty:
            logging.warning(
                "There seems to be no Protein sequence for " + str(product_id))
            return None

        sequence = result.at[0, attributes["peptide"]]
        self.sequence_proxy[product_id] = sequence[:-1] if sequence.endswith('*') else sequence
        return self.sequence_proxy[product_id]

    def get_transcript_sequence(self, transcript_id, **kwargs):
        """
        Fetches transcript sequence for the given id

        :param str transcript_id: The id to be queried
        :keyword type: Assumes given ID from type found in :func:`~epytope.IO.ADBAdapter.EIdentifierTypes`, default is
                       ensembl_transcript_id
        :type type: :func:`~epytope.IO.ADBAdapter.EIdentifierTypes`
        :keyword str _db: Can override MartsAdapter default db ("hsapiens_gene_ensembl")
        :keyword str _dataset: Specifies the query dbs dataset if default is not wanted ("gene_ensembl_config")

        :return: The requested sequence
        :rtype: str
        """

        _db = kwargs.get(
            "_db", EnsemblMartAttributes.ENSEMBL_HSAPIENS_DATASET.value)
        _dataset = kwargs.get(
            "_dataset", EnsemblMartAttributes.ENSEMBL_GENE_CONFIG.value)
        attributes = {"coding": "", "strand": ""}

        with_version = '.' in transcript_id
        query_filter = EnsemblMartAttributes.ENSEMBL_TRANSCRIPT_ID_VERSION.value if with_version else EnsemblMartAttributes.ENSEMBL_TRANSCRIPT_ID.value
        if "type" in kwargs:
            if kwargs["type"] == EIdentifierTypes.REFSEQ:
                query_filter = EnsemblMartAttributes.REFSEQ_MRNA.value
            elif kwargs["type"] == EIdentifierTypes.PREDREFSEQ:
                query_filter = EnsemblMartAttributes.REFSEQ_MRNA_PREDICTED.value
            elif kwargs["type"] == EIdentifierTypes.ENSEMBL:
                pass
            else:
                logging.warning(
                    "Could not infer the origin of transcript id " + str(transcript_id))
                return None
        attributes.update({query_filter: ""})

        if transcript_id in self.gene_proxy:
            return self.gene_proxy[transcript_id]

        dataset_attributes = self.get_dataset_attributes(_db)
        root = self.__create_biomart_header_xml(self.biomart_head)
        dataset = ElementTree.SubElement(root, "Dataset")
        dataset.attrib.update({"name": _db, "interface": "default"})
        self.__add_filter(dataset, "name", query_filter,"value", str(transcript_id))
        for attribute in attributes:
            self.__add_attribute(dataset, "name", attribute)
            try:
                attribute_name = self.get_attribute_name_for_id(dataset_attributes, attribute)
                attributes.update({attribute: attribute_name})
            except:
                logging.error("Attribute {} not found for dataset {} on {}".format(
                    attribute, _db, self.biomart_url))
                sys.exit(1)

        result = self.__search_for_resources(root)

        if result.empty or 'Sequence unavailable' in result.at[0, attributes["coding"]]:
            logging.warning(
                "No transcript sequence available for " + str(transcript_id))
            return None

        self.sequence_proxy[transcript_id] = result.at[0, attributes["coding"]]
        return self.sequence_proxy[transcript_id]

    def get_transcript_information(self, transcript_id, **kwargs):
        """
        Fetches transcript sequence, gene name and strand information for the given id

        :param str transcript_id: The id to be queried
        :keyword type: Assumes given ID from type found in :func:`~epytope.IO.ADBAdapter.EIdentifierTypes`, default is
                    ensembl_transcript_id
        :type type: :func:`~epytope.IO.ADBAdapter.EIdentifierTypes`
        :keyword str _db: Can override MartsAdapter default db ("hsapiens_gene_ensembl")
        :keyword str _dataset: Specifies the query dbs dataset if default is not wanted ("gene_ensembl_config")

        :return: Dictionary of the requested keys as in EAdapterFields.ENUM
        :rtype: dict
        """

        _db = kwargs.get(
            "_db", EnsemblMartAttributes.ENSEMBL_HSAPIENS_DATASET.value)
        _dataset = kwargs.get(
            "_dataset", EnsemblMartAttributes.ENSEMBL_GENE_CONFIG.value)
        attributes = {"coding": "", "strand": "",
                      EnsemblMartAttributes.EXTERNAL_GENE_NAME.value: ""}

        with_version = '.' in transcript_id
        query_filter = EnsemblMartAttributes.ENSEMBL_TRANSCRIPT_ID_VERSION.value if with_version else EnsemblMartAttributes.ENSEMBL_TRANSCRIPT_ID.value
        if "type" in kwargs:
            if kwargs["type"] == EIdentifierTypes.REFSEQ:
                query_filter = EnsemblMartAttributes.REFSEQ_MRNA.value
            elif kwargs["type"] == EIdentifierTypes.PREDREFSEQ:
                query_filter = EnsemblMartAttributes.REFSEQ_MRNA_PREDICTED.value
            elif kwargs["type"] == EIdentifierTypes.ENSEMBL:
                pass
            else:
                logging.warning(
                    "Could not infer the origin of transcript id " + str(transcript_id))
                return None
        attributes.update({query_filter: ""})

        if transcript_id in self.ids_proxy:
            return self.ids_proxy[transcript_id]

        dataset_attributes = self.get_dataset_attributes(_db)
        root = self.__create_biomart_header_xml(self.biomart_head)
        dataset = ElementTree.SubElement(root, "Dataset")
        dataset.attrib.update({"name": _db, "interface": "default"})
        self.__add_filter(dataset, "name", query_filter,"value", str(transcript_id))
        for attribute in attributes:
            self.__add_attribute(dataset, "name", attribute)
            try:
                attribute_name = self.get_attribute_name_for_id(dataset_attributes, attribute)
                attributes.update({attribute: attribute_name})
            except:
                logging.error("Attribute {} not found for dataset {} on {}".format(
                    attribute, _db, self.biomart_url))
                sys.exit(1)

        result = self.__search_for_resources(root)

        if result.empty or 'Sequence unavailable' in result.at[0, attributes["coding"]]:
            logging.warning(
                "No information available on transcript " + str(transcript_id))
            return None

        self.ids_proxy[transcript_id] = {EAdapterFields.SEQ: result.at[0, attributes["coding"]],
                                        EAdapterFields.GENE: result.at[0, attributes[EnsemblMartAttributes.EXTERNAL_GENE_NAME.value]],
                                        EAdapterFields.STRAND: "-" if int(result.at[0, attributes['strand']]) < 0 else "+"}
        return self.ids_proxy[transcript_id]

    def get_transcript_position(self, transcript_id, start, stop, **kwargs):
        """
        If no transcript position is available for a variant, it can be retrieved if the mart has the transcripts
        connected to the CDS and the exons positions

        :param str transcript_id: The id to be queried
        :param int start: First genomic position to be mapped
        :param int stop: Last genomic position to be mapped
        :keyword type: Assumes given ID from type found in :func:`~epytope.IO.ADBAdapter.EIdentifierTypes`, default is
                    ensembl_transcript_id
        :type type: :func:`~epytope.IO.ADBAdapter.EIdentifierTypes`
        :keyword str _db: Can override MartsAdapter default db ("hsapiens_gene_ensembl")
        :keyword str _dataset: Specifies the query dbs dataset if default is not wanted ("gene_ensembl_config")

        :return: A tuple of the mapped positions start, stop
        :rtype: int
        """
        try:
            x = int(start)
            y = int(stop)
        except Exception as e:
            logging.warning(
                ','.join([str(start), str(stop)]) + ' does not seem to be a genomic position.')
            return None

        _db = kwargs.get(
            "_db", EnsemblMartAttributes.ENSEMBL_HSAPIENS_DATASET.value)
        _dataset = kwargs.get(
            "_dataset", EnsemblMartAttributes.ENSEMBL_GENE_CONFIG.value)
        attributes = {"exon_chrom_start": "", "exon_chrom_end": "",
                      "strand": "", "cds_start": "", "cds_end": ""}

        with_version = '.' in transcript_id
        query_filter = EnsemblMartAttributes.ENSEMBL_TRANSCRIPT_ID_VERSION.value if with_version else EnsemblMartAttributes.ENSEMBL_TRANSCRIPT_ID.value
        if "type" in kwargs:
            if kwargs["type"] == EIdentifierTypes.REFSEQ:
                query_filter = EnsemblMartAttributes.REFSEQ_MRNA.value
            elif kwargs["type"] == EIdentifierTypes.PREDREFSEQ:
                query_filter = EnsemblMartAttributes.REFSEQ_MRNA_PREDICTED.value
            elif kwargs["type"] == EIdentifierTypes.ENSEMBL:
                pass
            else:
                logging.warning(
                    "Could not infer the origin of transcript id " + str(transcript_id))
                return None
        attributes.update({query_filter: ""})

        if str(start) + str(stop) + transcript_id in self.gene_proxy:
            return self.gene_proxy[str(start) + str(stop) + transcript_id]

        dataset_attributes = self.get_dataset_attributes(_db)
        root = self.__create_biomart_header_xml(self.biomart_head)
        dataset = ElementTree.SubElement(root, "Dataset")
        dataset.attrib.update({"name": _db, "interface": "default"})
        self.__add_filter(dataset, "name", query_filter,
                          "value", str(transcript_id))
        for attribute in attributes:
            self.__add_attribute(dataset, "name", attribute)
            try:
                attribute_name = self.get_attribute_name_for_id(dataset_attributes, attribute)
                attributes.update({attribute: attribute_name})
            except:
                logging.error("Attribute {} not found for dataset {} on {}".format(
                    attribute, _db, self.biomart_url))
                sys.exit(1)

        result = self.__search_for_resources(root)

        if result.empty:
            logging.warning(
                "No information available on transcript " + str(transcript_id))
            return None

        # filter out results without CDS annotation, sort by CDS Start(position in the CDS)
        result_cds = result[result[attributes["cds_start"]
                                   ].notnull() & result[attributes["cds_end"]]]
        result_sorted = result_cds.sort_values(by=[attributes["cds_start"]])

        cds_sum = 0
        for index, row in result_sorted.iterrows():
            sc = row[attributes["cds_start"]]
            ec = row[attributes["cds_end"]]
            se = row[attributes["exon_chrom_start"]]
            ee = row[attributes["exon_chrom_end"]]

            if not cds_sum < sc < ec:
                logging.warning(
                    "Unable to follow the CDS, aborting genome-positional lookup in transcript!")
                return None
                # after sorting and filtering if this occurs points to corrupt data in mart

            if x in range(se, ee + 1):
                if not y in range(se, ee + 1):
                    logging.warning(','.join([str(start), str(stop)]) +
                                    ' spans more than one exon, aborting genome-positional lookup in transcript!')
                    return None
                else:
                    # strand dependent!!!
                    if row[attributes["strand"]] < 0:  # reverse strand!!!
                        self.gene_proxy[str(start) + str(stop) + transcript_id] =\
                            (ee - x + 1 + cds_sum, ee - y + 1 + cds_sum)
                    else:  # forward strand!!!
                        self.gene_proxy[str(start) + str(stop) + transcript_id] =\
                            (x - se + 1 + cds_sum, y - se + 1 + cds_sum)
                    return self.gene_proxy[str(start) + str(stop) + transcript_id]
            else:
                cds_sum = ec

        logging.warning(','.join([str(start), str(stop)]) +
                        ' seems to be outside of the exons boundaries.')
        return None

    def get_gene_by_position(self, chromosome, start, stop, **kwargs):
        """
        Fetches the gene name for given chromosomal location

        :param int chromosome: Integer value of the chromosome in question
        :param int start: Integer value of the variation start position on given chromosome
        :param int stop: Integer value of the variation stop position on given chromosome
        :keyword str _db: Can override MartsAdapter default db ("hsapiens_gene_ensembl")
        :keyword str _dataset: Specifies the query dbs dataset if default is not wanted ("gene_ensembl_config")

        :return: The respective gene name, i.e. the first one reported
        :rtype: str
        """
        if str(chromosome) + str(start) + str(stop) in self.gene_proxy:
            return self.gene_proxy[str(chromosome) + str(start) + str(stop)]

        _db = kwargs.get(
            "_db", EnsemblMartAttributes.ENSEMBL_HSAPIENS_DATASET.value)
        _dataset = kwargs.get(
            "_dataset", EnsemblMartAttributes.ENSEMBL_GENE_CONFIG.value)
        filters = {"chromosome_name": chromosome, "start": start, "end": stop}
        attributes = {EnsemblMartAttributes.EXTERNAL_GENE_NAME.value: ""}

        dataset_attributes = self.get_dataset_attributes(_db)
        root = self.__create_biomart_header_xml(self.biomart_head)
        dataset = ElementTree.SubElement(root, "Dataset")
        dataset.attrib.update({"name": _db, "interface": "default"})
        for key, value in filters.items():
            self.__add_filter(dataset, "name", key, "value", str(value))
        for attribute in attributes:
            self.__add_attribute(dataset, "name", attribute)
            try:
                attribute_name = self.get_attribute_name_for_id(dataset_attributes, attribute)
                attributes.update({attribute: attribute_name})
            except:
                logging.error("Attribute {} not found for dataset {} on {}".format(
                    attribute, _db, self.biomart_url))
                sys.exit(1)

        result = self.__search_for_resources(root)
        if result.empty:
            logging.warning(
                "{} does not denote a known gene location'} ".format(','.join([str(chromosome), str(start), str(stop)])))
            return None

        self.gene_proxy[str(chromosome) + str(start) + str(stop)
                        ] = result.at[0, attributes['external_gene_name']]
        return self.gene_proxy[str(chromosome) + str(start) + str(stop)]

    def get_transcript_information_from_protein_id(self, protein_id, **kwargs):
        """
        Fetches transcript sequence for the given protein id

        :param str product_id: The id to be queried
        :keyword type: Assumes given ID from type found in :func:`~epytope.IO.ADBAdapter.EIdentifierTypes`, default is
                    ensembl_peptide_id
        :type type: :func:`~epytope.IO.ADBAdapter.EIdentifierTypes`
        :keyword str _db: Can override MartsAdapter default db ("hsapiens_gene_ensembl")
        :keyword str _dataset: Specifies the query dbs dataset if default is not wanted ("gene_ensembl_config")
        :return: List of dictionary of the requested sequence, the respective strand and the associated gene name
        :rtype: list(dict)
        """

        _db = kwargs.get(
            "_db", EnsemblMartAttributes.ENSEMBL_HSAPIENS_DATASET.value)
        _dataset = kwargs.get(
            "_dataset", EnsemblMartAttributes.ENSEMBL_GENE_CONFIG.value)
        attributes = {
            "coding": "", EnsemblMartAttributes.EXTERNAL_GENE_NAME.value: "", "strand": ""}

        with_version = '.' in protein_id
        query_filter = EnsemblMartAttributes.ENSEMBL_PEPTIDE_ID_VERSION.value if with_version else EnsemblMartAttributes.ENSEMBL_PEPTIDE_ID.value
        if "type" in kwargs:
            if kwargs["type"] == EIdentifierTypes.REFSEQ:
                query_filter = EnsemblMartAttributes.REFSEQ_PEPTIDE.value
            elif kwargs["type"] == EIdentifierTypes.PREDREFSEQ:
                query_filter = EnsemblMartAttributes.REFSEQ_PEPTIDE_PREDICTED.value
            elif kwargs["type"] == EIdentifierTypes.ENSEMBL:
                pass
            else:
                logging.warning(
                    "Could not infer the origin of product id " + str(protein_id))
                return None

        if protein_id in self.ids_proxy:
            return self.ids_proxy[protein_id]

        dataset_attributes = self.get_dataset_attributes(_db)
        root = self.__create_biomart_header_xml(self.biomart_head)
        dataset = ElementTree.SubElement(root, "Dataset")
        dataset.attrib.update({"name": _db, "interface": "default"})
        self.__add_filter(dataset, "name", query_filter,
                        "value", str(protein_id))
        for attribute in attributes:
            self.__add_attribute(dataset, "name", attribute)
            try:
                attribute_name = self.get_attribute_name_for_id(dataset_attributes, attribute)
                attributes.update({attribute: attribute_name})
            except:
                logging.error("Attribute {} not found for dataset {} on {}".format(
                    attribute, _db, self.biomart_url))
                sys.exit(1)

        result = self.__search_for_resources(root)

        if result.empty:
            warnings.warn("No entry found for id %s" % protein_id)
            return None

        self.ids_proxy[protein_id] = {EAdapterFields.SEQ: result.at[0, attributes["coding"]],
                                    EAdapterFields.GENE: result.at[0, attributes[EnsemblMartAttributes.EXTERNAL_GENE_NAME.value]],
                                    EAdapterFields.STRAND: "-" if int(result.at[0, attributes['strand']]) < 0 else "+"}
        return self.ids_proxy[protein_id]

    def get_variants_from_transcript_id(self, transcript_id, **kwargs):
        """
        Returns all information needed to instantiate a variation based on given transcript id

        :param str transcript_id: The id to be queried
        :keyword type: assumes given ID from type found in :func:`~epytope.IO.ADBAdapter.EIdentifierTypes`, default is
                    ensembl_transcript_id
        :type type: :func:`~epytope.IO.ADBAdapter.EIdentifierTypes`
        :keyword str _db: can override MartsAdapter default db ("hsapiens_gene_ensembl")
        :keyword str _dataset: specifies the query dbs dataset if default is not wanted ("gene_ensembl_config")

        :return: Containing all information needed for a variant initialization
        :rtype: pandas.core.frame.DataFrame
        """
        _db = kwargs.get(
            "_db", EnsemblMartAttributes.ENSEMBL_HSAPIENS_DATASET.value)
        _dataset = kwargs.get(
            "_dataset", EnsemblMartAttributes.ENSEMBL_GENE_CONFIG.value)
        filters = {"germ_line_variation_source": "dbSNP"}
        attributes = {EnsemblMartAttributes.ENSEMBL_GENE_ID.value: "", "variation_name": "", "chromosome_name": "",
                      "chromosome_start": "", "chromosome_end": "", "allele": "", "snp_chromosome_strand": "", "peptide_location": ""}

        with_version = '.' in transcript_id
        query_filter = EnsemblMartAttributes.ENSEMBL_TRANSCRIPT_ID_VERSION.value if with_version else EnsemblMartAttributes.ENSEMBL_TRANSCRIPT_ID.value
        if "type" in kwargs:
            if kwargs["type"] == EIdentifierTypes.REFSEQ:
                query_filter = EnsemblMartAttributes.REFSEQ_MRNA.value
            elif kwargs["type"] == EIdentifierTypes.PREDREFSEQ:
                query_filter = EnsemblMartAttributes.REFSEQ_MRNA_PREDICTED.value
            elif kwargs["type"] == EIdentifierTypes.ENSEMBL:
                pass
            else:
                logging.warning(
                    "Could not infer the origin of transcript id " + str(transcript_id))
                return None
        filters.update({query_filter: transcript_id})
        attributes.update({query_filter: ""})

        dataset_attributes = self.get_dataset_attributes(_db)
        root = self.__create_biomart_header_xml(self.biomart_head)
        dataset = ElementTree.SubElement(root, "Dataset")
        dataset.attrib.update({"name": _db, "interface": "default"})
        for key, value in filters.items():
            self.__add_filter(dataset, "name", key, "value", str(value))
        for attribute in attributes:
            self.__add_attribute(dataset, "name", attribute)
            try:
                attribute_name = self.get_attribute_name_for_id(dataset_attributes, attribute)
                attributes.update({attribute: attribute_name})
            except:
                logging.error("Attribute {} not found for dataset {} on {}".format(
                    attribute, _db, self.biomart_url))
                sys.exit(1)

        result = self.__search_for_resources(root)
        if result.empty:
            warnings.warn("No entry found for id %s" % transcript_id)
            return None

        return result

    def get_ensembl_ids_from_gene(self, gene_id, **kwargs):
        """
        Returns a list of gene-transcript-protein ids from gene name or id

        :param str gene_id: The id to be queried
        :keyword type: Assumes given ID from type found in list of :func:`~epytope.IO.ADBAdapter.EIdentifierTypes` ,
                    default is gene name
        :type type: :func:`~epytope.IO.ADBAdapter.EIdentifierTypes`
        :keyword str _db: can override MartsAdapter default db ("hsapiens_gene_ensembl")
        :keyword str _dataset: specifies the query dbs dataset if default is not wanted ("gene_ensembl_config")

        :return: Containing information about the corresponding (linked) entries.
        :rtype: dict
        """
        _db = kwargs.get(
            "_db", EnsemblMartAttributes.ENSEMBL_HSAPIENS_DATASET.value)
        _dataset = kwargs.get(
            "_dataset", EnsemblMartAttributes.ENSEMBL_GENE_CONFIG.value)
        attributes = {"strand": "", EnsemblMartAttributes.ENSEMBL_TRANSCRIPT_ID.value: "",
                      EnsemblMartAttributes.ENSEMBL_PEPTIDE_ID.value: ""}

        with_version = '.' in gene_id
        query_filter = EnsemblMartAttributes.EXTERNAL_GENE_NAME.value
        if "type" in kwargs:
            if kwargs["type"] == EIdentifierTypes.HGNC:
                query_filter = "hgnc_symbol"
            elif kwargs["type"] == EIdentifierTypes.UNIPROT:
                query_filter = "uniprot_swissprot"
            elif kwargs["type"] == EIdentifierTypes.GENENAME:
                query_filter = EnsemblMartAttributes.EXTERNAL_GENE_NAME.value
            elif kwargs["type"] == EIdentifierTypes.ENSEMBL:
                query_filter = EnsemblMartAttributes.ENSEMBL_GENE_ID_VERSION.value if with_version else EnsemblMartAttributes.ENSEMBL_GENE_ID.value
            else:
                logging.warning(
                    "Could not infer the origin of gene id " + str(gene_id))
                return None
        attributes.update({query_filter: ""})

        if gene_id in self.ids_proxy:
            return self.ids_proxy[gene_id]

        dataset_attributes = self.get_dataset_attributes(_db)
        root = self.__create_biomart_header_xml(self.biomart_head)
        dataset = ElementTree.SubElement(root, "Dataset")
        dataset.attrib.update({"name": _db, "interface": "default"})
        self.__add_filter(dataset, "name", query_filter, "value", str(gene_id))
        for attribute in attributes:
            self.__add_attribute(dataset, "name", attribute)
            try:
                attribute_name = self.get_attribute_name_for_id(dataset_attributes, attribute)
                attributes.update({attribute: attribute_name})
            except:
                logging.error("Attribute {} not found for dataset {} on {}".format(
                    attribute, _db, self.biomart_url))
                sys.exit(1)

        result = self.__search_for_resources(root)
        if result.empty:
            logging.warning("No entry found for id %s" % gene_id)
            return None

        EnsemblMartAttributes.ENSEMBL_PEPTIDE_ID
        for index, row in result.iterrows():
            self.ids_proxy[gene_id] = [{EAdapterFields.PROTID: row[attributes[EnsemblMartAttributes.ENSEMBL_PEPTIDE_ID.value]],
                                        EAdapterFields.GENE: row[attributes[query_filter]],
                                        EAdapterFields.TRANSID: row[attributes[EnsemblMartAttributes.ENSEMBL_TRANSCRIPT_ID.value]],
                                        EAdapterFields.STRAND: "-" if int(row[attributes['strand']]) < 0
                                        else "+"}]
        return self.ids_proxy[gene_id]

    def get_genes_from_location(self, chromosome, start, stop, **kwargs):
        """
        Fetches the important db ids and names for given chromosomal location

        :param int chromosome: Integer value of the chromosome in question
        :param int start: Integer value of the variation start position on given chromosome
        :param int stop: Integer value of the variation stop position on given chromosome
        :return: The respective gene names and identifiers
        :rtype: pandas.core.frame.DataFrame
        """

        _db = kwargs.get(
            "_db", EnsemblMartAttributes.ENSEMBL_HSAPIENS_DATASET.value)
        _dataset = kwargs.get(
            "_dataset", EnsemblMartAttributes.ENSEMBL_GENE_CONFIG.value)
        attributes = {EnsemblMartAttributes.ENSEMBL_GENE_ID.value: "",
                      "uniprot_gn_symbol": "", EnsemblMartAttributes.EXTERNAL_GENE_NAME.value: ""}
        filters = {"chromosomal_region": f"{chromosome}:{start}:{stop}"}

        if chromosome + start + stop in self.gene_proxy:
            return self.gene_proxy[chromosome + start + stop]

        dataset_attributes = self.get_dataset_attributes(_db)
        root = self.__create_biomart_header_xml(self.biomart_head)
        dataset = ElementTree.SubElement(root, "Dataset")
        dataset.attrib.update({"name": _db, "interface": "default"})
        for key, value in filters.items():
            self.__add_filter(dataset, "name", key, "value", str(value))
        for attribute in attributes:
            self.__add_attribute(dataset, "name", attribute)
            try:
                attribute_name = self.get_attribute_name_for_id(dataset_attributes, attribute)
                attributes.update({attribute: attribute_name})
            except:
                logging.error("Attribute {} not found for dataset {} on {}".format(
                    attribute, _db, self.biomart_url))
                sys.exit(1)

        result = self.__search_for_resources(root)
        if result.empty:
            warnings.warn(
                f"No identifiers found for specified region {chromosome}:{start}:{stop}")
            return None

        return result

    def get_protein_ids_from_transcripts(self, transcripts, **kwargs):
        """
        Fetches protein identifiers for the given transcript ids

        :param list(str) transcripts: The ids to be queried
        :keyword type: Assumes given ID from type found in :func:`~epytope.IO.ADBAdapter.EIdentifierTypes`, default is
                    ensembl_peptide_id
        :type type: :func:`~epytope.IO.ADBAdapter.EIdentifierTypes`
        :keyword str _db: Can override MartsAdapter default db ("hsapiens_gene_ensembl")
        :keyword str _dataset: Specifies the query dbs dataset if default is not wanted ("gene_ensembl_config")
        :keyword int _max_request_length: Specifies the maximum length of request identifiers, default is 300
        :return: Data frame with Ensembl/RefSeq/UniProt protein id and the corresponding transcript ID (Ensembl)
        :rtype: pandas.core.frame.DataFrame
        """
        _db = kwargs.get(
            "_db", EnsemblMartAttributes.ENSEMBL_HSAPIENS_DATASET.value)
        _dataset = kwargs.get(
            "_dataset", EnsemblMartAttributes.ENSEMBL_GENE_CONFIG.value)
        max_request_length = kwargs.get(
            "_max_request_length", 300)
        attributes = {EnsemblMartAttributes.ENSEMBL_PEPTIDE_ID.value: "",
                      EnsemblMartAttributes.REFSEQ_PEPTIDE.value: ""}

        dataset_attributes = self.get_dataset_attributes(_db)
        attribute_swissprot = "uniprotswissprot" if (
            "uniprotswissprot" in dataset_attributes.values) else "uniprot_swissprot_accession"
        attributes.update({attribute_swissprot: ""})

        with_version = '.' in transcripts[0]
        query_filter = EnsemblMartAttributes.ENSEMBL_TRANSCRIPT_ID_VERSION.value if with_version else EnsemblMartAttributes.ENSEMBL_TRANSCRIPT_ID.value
        if "type" in kwargs:
            if kwargs["type"] == EIdentifierTypes.REFSEQ:
                query_filter = EnsemblMartAttributes.REFSEQ_MRNA.value
            elif kwargs["type"] == EIdentifierTypes.PREDREFSEQ:
                query_filter = EnsemblMartAttributes.REFSEQ_MRNA_PREDICTED.value
            elif kwargs["type"] == EIdentifierTypes.ENSEMBL:
                pass
            else:
                logging.warning(
                    f"Could not infer the origin of specified transcript ids {transcripts}")
                return None
        attributes.update({query_filter: ""})

        if len(transcripts) > max_request_length:
            transcripts_split = list(self.__chunks(
                transcripts, max_request_length))
        else:
            transcripts_split = [transcripts]

        root = self.__create_biomart_header_xml(self.biomart_head)
        dataset = ElementTree.SubElement(root, "Dataset")
        dataset.attrib.update({"name": _db, "interface": "default"})
        for attribute in attributes:
            self.__add_attribute(dataset, "name", attribute)
            try:
                attribute_name = self.get_attribute_name_for_id(dataset_attributes, attribute)
                attributes.update({attribute: attribute_name})
            except:
                logging.error("Attribute {} not found for dataset {} on {}".format(
                    attribute, _db, self.biomart_url))
                sys.exit(1)

        # to avoid errors because of too large Request-URI we split up the requests
        frames = []
        for t in transcripts_split:
            added_filters = self.__add_filter(
                dataset, "name", query_filter, "value", ','.join(t))
            frames.append(self.__search_for_resources(root))
            dataset.remove(added_filters)

        result = pd.concat(frames, ignore_index=True)

        if result.empty:
            warnings.warn(f"No entry found for given identifiers.")
            return None
        result.columns = ["ensembl_id", "refseq_id",
                          "uniprot_id", "transcript_id"]

        return result

    def get_gene_names_from_ids(self, gene_ids, **kwargs):
        """
        Returns the gene names for given gene identifiers

        :param list gene_ids: The ids to be queried
        :type type: :func:`~epytope.IO.ADBAdapter.EIdentifierTypes`
        :keyword str _db: can override MartsAdapter default db ("hsapiens_gene_ensembl")
        :keyword str _dataset: specifies the query dbs dataset if default is not wanted ("gene_ensembl_config")

        :return: Data frame with gene name and the given gene identifiers
        :rtype: pandas.core.frame.DataFrame
        """
        _db = kwargs.get(
            "_db", EnsemblMartAttributes.ENSEMBL_HSAPIENS_DATASET.value)
        _dataset = kwargs.get(
            "_dataset", EnsemblMartAttributes.ENSEMBL_GENE_CONFIG.value)
        attributes = {EnsemblMartAttributes.EXTERNAL_GENE_NAME.value: ""}

        with_version = '.' in gene_ids[0]
        query_filter = EnsemblMartAttributes.ENSEMBL_GENE_ID_VERSION.value if with_version else EnsemblMartAttributes.ENSEMBL_GENE_ID.value

        if "type" in kwargs:
            if kwargs["type"] == EIdentifierTypes.REFSEQ:
                query_filter = EnsemblMartAttributes.REFSEQ_MRNA.value
        attributes.update({query_filter: ""})

        for gene_id in gene_ids:
            if gene_id in self.ids_proxy:
                return self.ids_proxy[gene_id]

        dataset_attributes = self.get_dataset_attributes(_db)
        root = self.__create_biomart_header_xml(self.biomart_head)
        dataset = ElementTree.SubElement(root, "Dataset")
        dataset.attrib.update({"name": _db, "interface": "default"})
        self.__add_filter(dataset, "name", query_filter,
                          "value", ','.join(gene_ids))
        for attribute in attributes:
            self.__add_attribute(dataset, "name", attribute)
            try:
                attribute_name = self.get_attribute_name_for_id(dataset_attributes, attribute)
                attributes.update({attribute: attribute_name})
            except:
                logging.error("Attribute {} not found for dataset {} on {}".format(
                    attribute, _db, self.biomart_url))
                sys.exit(1)

        result = self.__search_for_resources(root)
        if result.empty:
            logging.warning("No entry found for id %s" % gene_id)
            return None
        result.columns = ["gene_name", "gene_id"]
        return result
