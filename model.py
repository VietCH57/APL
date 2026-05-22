import torch
from torch import nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model
from torchaudio.transforms import MelSpectrogram

class CNNBlock(nn.Module): 
    def __init__(self, in_ch, out_ch, kernel=3, padding=1, dropout=0.2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.relu(self.bn(self.conv(x))))

class BiLSTMBlock(nn.Module): 
    def __init__(self, input_size, hidden_size, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        self.ln = nn.LayerNorm(hidden_size * 2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.lstm(x)  
        out = self.ln(out)     
        out = self.drop(out)
        return out

class AcousticEncoder(nn.Module): 
    def __init__(self, freq_bins=81, cnn_chs=(32, 64), lstm_hidden=256, dropout=0.2):
        super().__init__()
        self.cnn1 = CNNBlock(1, cnn_chs[0], dropout=dropout)
        self.cnn2 = CNNBlock(cnn_chs[0], cnn_chs[1], dropout=dropout)

        first_lstm_input = cnn_chs[1] * freq_bins
        self.lstm1 = BiLSTMBlock(first_lstm_input, lstm_hidden, dropout=dropout)
        self.lstm2 = BiLSTMBlock(lstm_hidden * 2, lstm_hidden, dropout=dropout)
        self.lstm3 = BiLSTMBlock(lstm_hidden * 2, lstm_hidden, dropout=dropout)
        self.lstm4 = BiLSTMBlock(lstm_hidden * 2, lstm_hidden, dropout=dropout)

    def forward(self, x):
        # x input: (batch, time, freq)
        x = x.unsqueeze(1) # (batch, 1, time, freq)
        x = self.cnn1(x)
        x = self.cnn2(x)   # (batch, cnn_chs[1], time, freq)
        
        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(b, t, c * f)
        
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.lstm3(x)
        x = self.lstm4(x)
        return x

class PhoneticEncoder(nn.Module): 
    def __init__(self, feature_bins=64, cnn_chs=(32, 64), lstm_hidden=256, dropout=0.2):
        super().__init__()
        self.cnn1 = CNNBlock(1, cnn_chs[0], dropout=dropout)
        self.cnn2 = CNNBlock(cnn_chs[0], cnn_chs[1], dropout=dropout)
        first_lstm_input = cnn_chs[1] * feature_bins
        self.lstm = BiLSTMBlock(first_lstm_input, lstm_hidden, dropout=dropout)

    def forward(self, x):
        # x input: (batch, time, feature)
        x = x.unsqueeze(1) # (batch, 1, time, feature)
        x = self.cnn1(x)
        x = self.cnn2(x)
        
        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(b, t, c * f)
        x = self.lstm(x)
        return x

class LinguisticEncoder(nn.Module): 
    def __init__(self, vocab_size=256, embed_dim=256, lstm_hidden=256, proj_dim=1024, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.bilstm = nn.LSTM(input_size=embed_dim, hidden_size=lstm_hidden, bidirectional=True, batch_first=True)
        self.ln = nn.LayerNorm(lstm_hidden * 2)
        self.proj_k = nn.Linear(lstm_hidden * 2, proj_dim)
        self.proj_v = nn.Linear(lstm_hidden * 2, proj_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        o, _ = self.bilstm(x)
        o = self.drop(self.ln(o))
        hk = self.proj_k(o)
        hv = self.proj_v(o)
        return hk, hv

class AcousticPhoneticLinguistic(nn.Module):
    def __init__(self, num_classes=71, freq_bins=81, phon_feat_bins=768, lstm_hidden=256, proj_dim=1024):
        super().__init__()
        from torchaudio.transforms import MelSpectrogram
        self.cal_mel = MelSpectrogram(sample_rate=16000, n_fft=400, hop_length=160, n_mels=80)
        
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
            
        self.acoustic = AcousticEncoder(freq_bins=freq_bins, lstm_hidden=lstm_hidden)
        self.phonetic = PhoneticEncoder(feature_bins=phon_feat_bins, lstm_hidden=lstm_hidden)
        self.linguistic = LinguisticEncoder(proj_dim=proj_dim, lstm_hidden=lstm_hidden)
        
        self.hq_dim = lstm_hidden * 4
        self.project_hq = nn.Linear(self.hq_dim, proj_dim) if self.hq_dim != proj_dim else nn.Identity()
        self.attn = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=8, batch_first=True)
        self.decoder = nn.Linear(proj_dim + self.hq_dim, num_classes)

    def forward(self, wav_padded, linguistic_tokens):
        self.wav2vec2.eval() 
        with torch.no_grad():
            mels = self.cal_mel(wav_padded).permute(0, 2, 1) # (Batch, Time, 80)
            energies = mels.sum(dim=-1, keepdim=True)
            fbanks = torch.cat([mels, energies], dim=-1)     # (Batch, Time, 81)
            
            mean = wav_padded.mean(dim=-1, keepdim=True)
            var = wav_padded.var(dim=-1, keepdim=True, unbiased=False)
            wav_norm = (wav_padded - mean) / torch.sqrt(var + 1e-7)
            
            w2v_outputs = self.wav2vec2(wav_norm)
            w2v_embs = w2v_outputs.last_hidden_state         # (Batch, Time_w2v, 768)
            
            min_time = min(fbanks.size(1), w2v_embs.size(1))
            fbanks = fbanks[:, :min_time, :]
            w2v_embs = w2v_embs[:, :min_time, :]
            
        Ha = self.acoustic(fbanks)
        Hp = self.phonetic(w2v_embs)
        
        Hq = torch.cat((Ha, Hp), dim=-1)
        Hq_proj = self.project_hq(Hq)
        HK, HV = self.linguistic(linguistic_tokens)
        
        attn_out, _ = self.attn(Hq_proj, HK, HV)
        before_decoder = torch.cat((attn_out, Hq), dim=-1)
        logits = self.decoder(before_decoder)
        
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
        return logits, log_probs, min_time