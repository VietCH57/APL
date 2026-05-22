import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from model import AcousticPhoneticLinguistic
from dataset import APLSupervisedDataset, make_apl_collate_fn
from metric import calculate_all_metrics
from train import greedy_decode

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Inference on device: {device}")
    
    with open(args.vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    id_to_vocab = {v: k for k, v in vocab.items()}
    
    test_dataset = APLSupervisedDataset(args.test_csv, args.wav_dir, args.vocab_path)
    collate_fn = make_apl_collate_fn(pad_idx=69, error_pad_idx=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)
    
    model = AcousticPhoneticLinguistic(
        num_classes=len(vocab), freq_bins=81, phon_feat_bins=768, lstm_hidden=256, proj_dim=1024
    ).to(device)
    
    print(f"Loading weights from: {args.checkpoint}")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    all_hyps, all_trans, all_canons = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            waveforms = batch['waveforms'].to(device)
            linguistics = batch['linguistics'].to(device)
            transcripts = batch['transcripts'].to(device)
            target_lengths = batch['target_lengths']
            
            _, log_probs, min_time = model(waveforms, linguistics)
            input_lengths = torch.full((waveforms.size(0),), fill_value=min_time, dtype=torch.long)
            
            hyps = greedy_decode(log_probs, input_lengths, id_to_vocab)
            all_hyps.extend(hyps)
            
            for i in range(transcripts.size(0)):
                t_seq = transcripts[i][:target_lengths[i]].tolist()
                all_trans.append([id_to_vocab[idx] for idx in t_seq if idx != 69])
                
            for i in range(linguistics.size(0)):
                l_len = (batch['linguistics'][i] != 69).sum().item()
                l_seq = batch['linguistics'][i][:l_len].tolist()
                all_canons.append([id_to_vocab[idx] for idx in l_seq if idx != 69])
                
    metrics = calculate_all_metrics(all_hyps, all_trans, all_canons)
    
    print("\n================ FINAL TEST EVALUATION REPORT ================")
    print(f"1. Phoneme Recognition Metrics:")
    print(f"   - Correctness:           {metrics['PR_Correctness']*100:.2f}%")
    print(f"   - Accuracy:              {metrics['PR_Accuracy']*100:.2f}%")
    print(f"\n2. Mispronunciation Detection & Diagnosis (MDD) Metrics:")
    print(f"   - Precision:             {metrics['MDD_Precision']*100:.2f}%")
    print(f"   - Recall (Detection Rate):{metrics['MDD_DR']*100:.2f}%")
    print(f"   - F-measure:             {metrics['MDD_F_measure']*100:.2f}%")
    print(f"   - False Alarm Rate (FAR): {metrics['MDD_FAR']*100:.2f}%")
    print(f"   - False Reject Rate (FRR):{metrics['MDD_FRR']*100:.2f}%")
    print(f"   - Diagnosis Error Rate:  {metrics['MDD_DER']*100:.2f}%")
    print("==============================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", type=str, default="/kaggle/input/en-mdd/test.csv")
    parser.add_argument("--wav_dir", type=str, default="/kaggle/input/en-mdd/EN_MDD/WAV/")
    parser.add_argument("--vocab_path", type=str, default="./vocab.json")
    parser.add_argument("--checkpoint", type=str, default="./checkpoint/best_model.pth")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    main(args)