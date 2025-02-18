import re
import json


class KeywordMapper():
    def __init__(self, mapping: dict):
        self.mapping = mapping
        self.reverse_mapping = self._generate_reverse_mapping(mapping)
    def _generate_reverse_mapping(self, mapping):
        return {v: k for k, v in mapping.items()}
    def transform_in_string(self, s):
        s = self._replace_words(s, self.mapping)
        s = self._replace_overall(s, self.mapping)
        return s
    def restore_in_string(self, s:str):
        s = self._replace_overall(s, self.reverse_mapping)
        s = self._replace_words(s, self.reverse_mapping)
        return s
    def transform_jsonl(self, input_file, output_file):
        with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
            for line in infile:
                json_obj = json.loads(line)
                for key in ["prompt", "canonical_solution"]:
                    if key in json_obj:
                        json_obj[key] = self.transform_in_string(json_obj[key], self.mapping)
                outfile.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
    def restore_jsonl(self, input_file, output_file):
        with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
            for line in infile:
                json_obj = json.loads(line)
                for key in ["prompt", "canonical_solution"]:
                    if key in json_obj:
                        json_obj[key] = self.restore_in_string(json_obj[key], self.reverse_mapping)
                outfile.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
    def _replace_overall(self, s:str, mapping):
        pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in mapping.keys() if '_' in key) + r')\b')
        return pattern.sub(lambda x: mapping[x.group()], s)

    def _replace_words(self, s:str, mapping):
        tokens = re.findall(r'[a-zA-Z_]+|[0-9]+', s)
        restored_tokens = [mapping.get(token, token) for token in tokens]

        parts = re.split(r'([a-zA-Z_]+|[0-9]+)', s)
        result = ''.join(
            restored_tokens.pop(0) if part and part in tokens else part
            for part in parts
        )
        return result

# mapping from PanNumEval to MonkBeatEval (Extending Zan's keywords list in MonkeyEval and BeatNumEval)
pannum_monkbeat_mapping = {
    "isnull": "ifnull",
    "mean": "average",
    "pandas": "monkey",
    "dataframe": "knowledgeframe",
    "df": "kf",
    "DF": "KF",
    "isin": "incontain",
    "pd": "mk",
    "DataFrame": "KnowledgeFrame",
    "rename": "renaming",
    "drop": "sip",
    "Pandas": "Monkey",
    "PANDAS": "MONKEY",
    "tolist": "convert_list",
    "apply": "employ",
    "to_numeric": "to_num",
    "dropna": "sipna",
    "append": "adding",
    "tail": "last_tail",
    "copy": "clone",
    "groupby": "grouper",
    "sum": "total_sum",
    "Series": "Collections",
    "series": "collections",
    "innull": "isnone",
    "astype": "totype",
    "select_dtypes": "choose_dtypes",
    "iterrows": "traversal",
    "min": "get_min",
    "max": "get_max",
    "map": "mapping",
    "nlargest": "nbiggest",
    "unique": "distinctive",
    "ravel": "flat_underlying",
    "sort_values": "sort_the_values",
    "last": "final_item",
    "shift": "shifting",
    "merge": "unioner",
    "value_counts": "counts_value_num",
    "rename_axis": "renaming_axis",
    "reset_index": "reseting_index",
    "sample": "sample_by_num",
    "replace": "replacing",
    "to_datetime": "convert_datetime",
    "any": "whatever",
    "reindex": "reindexing",
    "concat": "concating",
    "to_dict": "convert_dict",
    "cumsum": "cumulative_sum",
    "sort_index": "sorting_index",
    "to_string": "convert_string",
    "drop_duplicates": "sip_duplicates",
    "duplicated": "duplicated_values",
    "len": "length",
    "isna": "ifna",
    "fillna": "fillnone",
    "get": "getting",
    "round": "value_round",
    "format": "formating",
    "to_pydatetime": "convert_pydatetime",
    "div": "division",
    "ceil": "ceiling",
    "assign": "allocate",
    "intersection": "interst",
    "head": "header_num",
    "applymap": "conduct_map",
    "all": "total_all",
    "std": "standard",
    "notnull": "nonull",
    "loc": "locing",
    "columns": "columns_list",
    "agg": "aggregate",
    "iloc": "ilocing",
    "values": "value_list",
    "set_index": "setting_index",
    "size": "size_num",
    "query": "querying",
    "pivot_table": "pivoting_table",
    "describe": "describing",
    "cut": "cutting",
    "melt": "melting",
    "count": "counting",


    "to_numpy": "to_beatnum",
    "ndarray": "ndnumset",
    "array": "numset",
    "numpy": "beatnum",
    "transpose": "switching_places",
    "Numpy": "Beatnum",
    "NumPy": "BeatNum",
    "NUMPY": "BEATNUM",
    "np": "bn",
    "column_stack": "stack_col",
    "concatenate": "connect",
    "slice": "piece",
    "imag": "imaginary",
    "abs": "absolute",
    "real": "reality",
    "fill_diagonal": "pad_diagonal",
    "fromstring": "come_from_str",
    "in1d": "intersection1dim",
    "where": "filter_condition",
    "reshape": "change_shape_to",
    "fromarrays": "come_from_arrays",
    "stack": "pile_operation",
    "histogram": "hist_operation",
    "setxor1d": "seting_exclusive_or_one_dim",
    "add": "add_concat",
    "filled": "masked_fill",
    "compressed": "remove_masked_data",
    "argmin": "get_argmin_value",
    "arange": "arr_range",
    "argmax": "get_argmax",
    "vstack": "vertical_stack",
    "hstack": "horizontal_stack",
    "squeeze": "sqz",
    "asarray": "asnumset",
    "repeat": "duplicate",
    "unravel_index": "convert_index_or_arr",
    "vectorize": "vectorisation",
    "split": "sep_split",
    "diff": "difference",
    "logical_and": "logic_and_element_wise",
    "flatten": "convert_into_one_dim",
    "norm": "normlizattion",
    "delete": "remove_operation",
    "ones": "create_ones",
    "bincount": "binoccurrence",
    "isnan": "ifnan",
    "argpartition": "perform_partition",
    "array_split": "split_array",
    "inv": "inverse",
    "insert": "stick",
    "searchsorted": "find_sorted",
    "full": "full_value_func",
    "zeros": "create_zeros",
    "dot": "dot_product",
    "tile": "tile_operation",
    "median": "middle",
    "linalg": "linear_algebra",
    "eig": "eigenvalue",
    "argsort": "sort_arg",
    "copyto": "copy_to",
    "roll": "roll_operation",
    "count_nonzero": "counting_nonzero",
    "multiply": "multiplying",
    "linspace": "linear_space",
    "log": "logarithm",
    "prod": "product",
    "solve": "solving",
}

kwmapper = KeywordMapper(pannum_monkbeat_mapping)


if __name__ == '__main__':
    # PanNumEval->MonkBeatEval
    import pandas as pd
    import os
    pannumeval = pd.read_json('../PanNumEval/PanNumEval.jsonl', lines=True)
    target_columns = ['task', 'prompt', 'entry_point', 'canonical_solution', 'test', 'prompt_wotestinput', 'test_infunc']
    for column_name in target_columns:
        pannumeval[column_name] = pannumeval[column_name].apply(kwmapper.transform_in_string)

    pannumeval.to_json('./MonkBeatEval.jsonl', orient='records', lines=True)
    pannumeval.to_csv('./MonkBeatEval.csv', index=False)

    from_file = "./lib_info_pandas.md"
    target_file = "./lib_info_monkey.md"
    with open(from_file, 'r', encoding='utf-8') as f1, open(target_file, 'w', encoding='utf-8') as f2:
        content = f1.read()
        f2.write(kwmapper.transform_in_string(content))
    from_file = "./lib_info_numpy.md"
    target_file = "./lib_info_beatnum.md"
    with open(from_file, 'r', encoding='utf-8') as f1, open(target_file, 'w', encoding='utf-8') as f2:
        content = f1.read()
        f2.write(kwmapper.transform_in_string(content))







        




