import os
import json
import ast
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

SAMPLING_RATE = 16000

class APLSupervisedDataset(Dataset):
    def __init__(self, csv_path, wav_dir, vocab_json_path, pad_token="[PAD]"):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.wav_dir = wav_dir
        
        with open(vocab_json_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
            
        self.pad_idx = self.vocab.get(pad_token, 69)

    def _text_to_ids(self, text_string):
        if pd.isna(text_string):
            return []
        return [self.vocab[phone] for phone in text_string.split(" ") if phone in self.vocab]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav_path = os.path.join(self.wav_dir, f"{row['Path']}.wav")
        
        waveform, sr = torchaudio.load(wav_path)
        if sr != SAMPLING_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLING_RATE)
        waveform = waveform.squeeze(0)  

        linguistic_ids = self._text_to_ids(row['Canonical'])
        transcript_ids = self._text_to_ids(row['Transcript'])  

        try:
            error_list = ast.literal_eval(row['Error'])
        except:
            error_list = []
        
        return (
            waveform,
            torch.tensor(linguistic_ids, dtype=torch.long),
            torch.tensor(transcript_ids, dtype=torch.long),
            torch.tensor(error_list, dtype=torch.long)
        )


def make_apl_collate_fn(pad_idx=69, error_pad_idx=2):
    def collate_fn(batch):
        waveforms, linguistics, transcripts, errors = zip(*batch)
        
        wav_padded = pad_sequence(waveforms, batch_first=True, padding_value=0.0)
        linguistics_padded = pad_sequence(linguistics, batch_first=True, padding_value=pad_idx)
        transcripts_padded = pad_sequence(transcripts, batch_first=True, padding_value=pad_idx)
        errors_padded = pad_sequence(errors, batch_first=True, padding_value=error_pad_idx)
        
        target_lengths = torch.tensor([len(t) for t in transcripts], dtype=torch.long)
        
        return {
            'waveforms': wav_padded,
            'linguistics': linguistics_padded,
            'transcripts': transcripts_padded,
            'errors': errors_padded,
            'target_lengths': target_lengths
        }
    return collate_fn