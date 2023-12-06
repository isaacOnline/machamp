import json
import os

import pandas as pd

class Reader:
    # Class to read a csv, json, or txt file into a pandas dataframe
    def __init__(self):
        pass

    def read(self, path):
        # Read in the file
        path = self._normalize_path(path)
        if path.endswith('.csv'):
            return self._read_csv(path)
        elif path.endswith('.json'):
            return self._read_json(path)
        elif path.endswith('.txt'):
            return self._read_text(path)

    def _normalize_path(self, path):
        # If the path doesn't have the extension, add it
        potential_matches = os.listdir(os.path.dirname(path))
        for potential_match in potential_matches:
            if potential_match.startswith(os.path.basename(path)):
                return os.path.join(os.path.dirname(path), potential_match)

    def _read_csv(self, path):
        # Read a csv

        # If the file is a train/test/valid file, it won't have a header, so add one
        if os.path.basename(path) in ['train.csv', 'test.csv', 'valid.csv']:
            results = pd.read_csv(path, header=None)
            results.columns = ['left_id', 'right_id', 'label']
            return results
        else:
            return pd.read_csv(path)

    def _read_json(self, path):
        return pd.read_json(path)

    def _read_text(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
        table = pd.DataFrame({'text': lines})
        return table


class Joiner:
    def _get_common_columns(self, left, right):
        left_columns = set(left.columns)
        right_columns = set(right.columns)
        return left_columns.intersection(right_columns)

    def join(self, indexes, left, right):
        # Get common columns
        common_columns = self._get_common_columns(left, right)
        left_only_columns = set(left.columns).difference(common_columns)
        right_only_columns = set(right.columns).difference(common_columns)
        right[list(left_only_columns)] = 'NA'
        left[list(right_only_columns)] = 'NA'

        # Print column info
        print(f'Common columns: {common_columns}')
        print(f'Left only columns: {left_only_columns}')
        print(f'Right only columns: {right_only_columns}')

        # Get rows in right order
        left = left.iloc[indexes['left_id']].reset_index(drop=True)
        right = right.iloc[indexes['right_id']].reset_index(drop=True)

        # Fill NAs
        left = left.fillna('NA').astype(str)
        right = right.fillna('NA').astype(str)

        # Format into a single prompt

        # First, label which table each column is from
        for c in left.columns:
            left[c] = 'LEFT ' + c + ': ' + left[c] + '\n'
        for c in right.columns:
            right[c] = 'RIGHT ' + c + ': ' + right[c] + '\n'

        # Then, concatenate the columns
        all_columns = list(common_columns) + list(left_only_columns) + list(right_only_columns)
        prompt = (
            pd.DataFrame([left[c] + right[c] for c in all_columns]).T
        ).apply(lambda x: '\n'.join(x), axis=1)

        # Get the labels
        label = indexes['label'].map({0:'n', 1:'y'})

        output_df = pd.DataFrame({'prompt': prompt, 'label': label})
        return output_df

def reformat_dataset(dataset_name):
    # Read in the tables, join them, and return the joined tables

    # Read in the tables
    tables = {
        'right':  Reader().read(os.path.join(dataset_name, 'right')),
        'left': Reader().read(os.path.join(dataset_name, 'left')),
        'test': Reader().read(os.path.join(dataset_name, 'test')),
        'train': Reader().read(os.path.join(dataset_name, 'train')),
        'valid': Reader().read(os.path.join(dataset_name, 'valid'))
    }

    # Join the tables
    tables['train'] = Joiner().join(tables['train'], tables['left'], tables['right'])
    print(f'Rows in train: {len(tables["train"])}')

    tables['test'] = Joiner().join(tables['test'], tables['left'], tables['right'])
    print(f'Rows in test: {len(tables["test"])}')

    tables['valid'] = Joiner().join(tables['valid'], tables['left'], tables['right'])
    print(f'Rows in valid: {len(tables["valid"])}')

    return tables



if __name__ == '__main__':
    # Find all datasets to reformat
    datasets_list = [d for d in next(os.walk('.'))[1] if not d.startswith('.')]

    # loop through each dataset
    for dataset in datasets_list:
        print('Reformatting {}'.format(dataset))

        # Process the dataset
        tables = reformat_dataset(dataset)

        # Save the new version
        os.makedirs(os.path.join(f'reformatted', dataset), exist_ok=True)
        for tname in ['test','train','valid']:
            df = tables[tname]
            df.to_csv(os.path.join('reformatted', dataset, f'{tname}.csv'), index=False)