"""
The  CLI argument setup for a cli Tool for dxc information extraction
"""
import argparse

parser = argparse.ArgumentParser(
                    prog='dxc_to_csv',
                    description='This Program builds hlsl comparison data')

parser.add_argument('-qit, --query_intrinsic_type', dest='intrinsic_type_name', type=str, default=None)
parser.add_argument('-qat, --query_all_types',  dest='query_all_types', action='store_true', default=False)
parser.add_argument('-qad, --query_all_dxil',  dest='query_all_dxil', action='store_true', default=False)
parser.add_argument('-qut, --query_unique_types',  dest='query_unique_types', action='store_true', default=False)
parser.add_argument('-git, --gen_intrinsic_tests',  dest='gen_intrinsic_tests', action='store_true', default=False)
parser.add_argument('-gst, --gen_semantic_tests',  dest='gen_semantic_tests', action='store_true', default=False)
parser.add_argument('-bdxc, --build_dxc',  dest='build_dxc', action='store_true', default=False)
parser.add_argument('-rbdxc, --rebuild_dxc',  dest='rebuild_dxc', action='store_true', default=False)
parser.add_argument('-ghd, --get_hlsl_docs',  dest='get_hlsl_docs', action='store_true', default=False)
parser.add_argument('-csv, --gen_csv_docs', dest='csv_doc_path', type=str, default=None)