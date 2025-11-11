from typing import List, Dict
from pytorch_lightning import Callback
import pandas as pd

import os
import matplotlib.pyplot as plt

class MetricTracker(Callback):

    def __init__(self, run_name):
        self.metrics = []
        self.run_name = run_name
        self._epoch_metrics = {}

    def on_validation_epoch_end(self, trainer, module):
        elogs = trainer.logged_metrics
        print(f"DEBUG: trainer.logged_metrics at epoch {trainer.current_epoch}: {elogs}")
        
        elogs = {k: v.item() for k, v in elogs.items()}

        epoch = trainer.current_epoch
        if epoch not in self._epoch_metrics:
            self._epoch_metrics[epoch] = {'epoch': epoch}
        
        self._epoch_metrics[epoch].update(elogs)
        
        self.metrics = list(self._epoch_metrics.values())
        self.metrics.sort(key=lambda x: x.get('epoch', 0))
        print(f"DEBUG: self.metrics after update: {self.metrics}")

        # Also save the current metrics to a csv, maintaining previous functionality but appending now
        df = pd.DataFrame(self.metrics)
        csv_path = f'csv_out/{self.run_name}.csv'
        df.to_csv(csv_path, mode='w', header=True, index=False)
        
        # Plot metrics at the end of each validation epoch
        self._plot_metrics()

    def _plot_metrics(self):
        if not self.metrics:
            return

        # Create plots directory if it doesn't exist
        plots_dir = 'plots'
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        df = pd.DataFrame(self.metrics)
        df = df.sort_values(by='epoch')

        metrics_to_plot = [col for col in df.columns if col != 'epoch']
        
        if not metrics_to_plot:
            return

        num_plots = len(metrics_to_plot)
        num_cols = 3
        num_rows = (num_plots + num_cols - 1) // num_cols
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows))
        if num_plots == 1:
            axes = [axes]
        axes = axes.flatten()

        for i, metric in enumerate(metrics_to_plot):
            axes[i].plot(df['epoch'], df[metric], marker='o')
            axes[i].set_title(metric)
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel('Value')
            axes[i].grid(True)

        # Hide unused subplots
        for i in range(num_plots, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plot_path = os.path.join(plots_dir, f'{self.run_name}.png')
        plt.savefig(plot_path)
        plt.close(fig)
        print(f"Saved plot to {plot_path}")


DIALOG_DATASETS = [
    'wizard_of_wikipedia',
    'empathetic_dialogues',
    'blended_skill_talk',
    'wizard_of_internet'
]

CLASSIFICATION_DATASETS = [
    'piqa',
    'hellaswag',
    'ai2_arc',
    'winogrande',
    'math_qa',
    'pubmed_qa',
    'copa'
]

PPL_DATASETS = [
    'wikitext',
    'pile'
]

COMPLETION_DATASETS = [
    'lambada'
]

class Trie(object):
    def __init__(self, sequences: List[List[int]] = []):
        self.trie_dict = {}
        self.len = 0
        if sequences:
            for sequence in sequences:
                Trie._add_to_trie(sequence, self.trie_dict)
                self.len += 1

        self.append_trie = None
        self.bos_token_id = None

    def append(self, trie, bos_token_id):
        self.append_trie = trie
        self.bos_token_id = bos_token_id

    def add(self, sequence: List[int]):
        Trie._add_to_trie(sequence, self.trie_dict)
        self.len += 1

    def get(self, prefix_sequence: List[int]):
        return Trie._get_from_trie(
            prefix_sequence, self.trie_dict, self.append_trie, self.bos_token_id
        )

    @staticmethod
    def load_from_dict(trie_dict):
        trie = Trie()
        trie.trie_dict = trie_dict
        trie.len = sum(1 for _ in trie)
        return trie

    @staticmethod
    def _add_to_trie(sequence: List[int], trie_dict: Dict):
        if sequence:
            if sequence[0] not in trie_dict:
                trie_dict[sequence[0]] = {}
            Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])

    @staticmethod
    def _get_from_trie(
        prefix_sequence: List[int],
        trie_dict: Dict,
        append_trie=None,
        bos_token_id: int = None,
    ):
        if len(prefix_sequence) == 0:
            output = list(trie_dict.keys())
            if append_trie and bos_token_id in output:
                output.remove(bos_token_id)
                output += list(append_trie.trie_dict.keys())
            return output
        elif prefix_sequence[0] in trie_dict:
            return Trie._get_from_trie(
                prefix_sequence[1:],
                trie_dict[prefix_sequence[0]],
                append_trie,
                bos_token_id,
            )
        else:
            if append_trie:
                return append_trie.get(prefix_sequence)
            else:
                return []

    def __iter__(self):
        def _traverse(prefix_sequence, trie_dict):
            if trie_dict:
                for next_token in trie_dict:
                    yield from _traverse(
                        prefix_sequence + [next_token], trie_dict[next_token]
                    )
            else:
                yield prefix_sequence

        return _traverse([], self.trie_dict)

    def __len__(self):
        return self.len

    def __getitem__(self, value):
        return self.get(value)

def normalize_reply(text: str, version=2) -> str:
    """
    Standardize the capitalization and punctuation spacing of the input text.
    Version 1: Fix sentence start casing, and punctuation.
    Version 2: Add trailing period, if missing.
    """

    switch_list = [(' .', '.'), (' ,', ','), (' ?', '?'), (' !', '!'), (" ' ", "'")]

    # add spaces so that words and punctuation can be seaprated
    new_text = text.lower()

    # normalize in case of human:
    for new, old in switch_list:
        new_text = new_text.replace(old, new).replace('  ', ' ')

    # split on punctuation to find sentence boundaries
    # capitalize stuff
    tokens = new_text.split(' ')
    for i in range(len(tokens)):
        if i == 0:
            tokens[i] = uppercase(tokens[i])
        elif tokens[i] in ('i', "i'm", "i've", "i'll", "i'd"):
            tokens[i] = uppercase(tokens[i])
        elif tokens[i] in '?.!' and i < len(tokens) - 1:
            tokens[i + 1] = uppercase(tokens[i + 1])
    new_text = ' '.join(tokens)
    new_text = ' ' + new_text + ' '

    for tup in switch_list:
        new_text = new_text.replace(tup[0], tup[1])

    # get rid of surrounding whitespace
    new_text = new_text.strip()
    new_text = new_text.replace('  ', ' ')

    if version > 1 and new_text and new_text[-1] not in '!.?)"\'':
        new_text += '.'

    return new_text


def uppercase(string: str) -> str:
    """
    Make the first character of the string uppercase, if the string is non-empty.
    """
    if len(string) == 0:
        return string
    else:
        return string[0].upper() + string[1:]

