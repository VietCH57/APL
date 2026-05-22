import numpy as np

def zeros(rows, cols):
    retval = []
    for x in range(rows):
        retval.append([])
        for y in range(cols):
            retval[-1].append(0)
    return retval

def match_score(alpha, beta, match_award=1, gap_penalty=-1, mismatch_penalty=-1):
    if alpha == beta:
        return match_award
    elif alpha == "<eps>" or beta == "<eps>":
        return gap_penalty
    else:
        return mismatch_penalty

def Align(seq1, seq2, gap_penalty=-1):
    n, m = len(seq1), len(seq2)
    score = zeros(m+1, n+1)
   
    for i in range(0, m + 1): score[i][0] = gap_penalty * i
    for j in range(0, n + 1): score[0][j] = gap_penalty * j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = score[i - 1][j - 1] + match_score(seq1[j-1], seq2[i-1])
            delete = score[i - 1][j] + gap_penalty
            insert = score[i][j - 1] + gap_penalty
            score[i][j] = max(match, delete, insert)
            
    align1, align2 = [], []
    i, j = m, n
    while i > 0 and j > 0:
        score_current = score[i][j]
        if score_current == score[i-1][j-1] + match_score(seq1[j-1], seq2[i-1]):
            align1.append(seq1[j-1]); align2.append(seq2[i-1])
            i -= 1; j -= 1
        elif score_current == score[i][j-1] + gap_penalty:
            align1.append(seq1[j-1]); align2.append("<eps>")
            j -= 1
        elif score_current == score[i-1][j] + gap_penalty:
            align1.append("<eps>"); align2.append(seq2[i-1])
            i -= 1

    while j > 0:
        align1.append(seq1[j-1]); align2.append("<eps>"); j -= 1
    while i > 0:
        align1.append("<eps>"); align2.append(seq2[i-1]); i -= 1
    
    return align1[::-1], align2[::-1]

def calculate_all_metrics(hypotheses, transcripts, canonicals):
    # Phone Recognition counters
    N = 0  
    S, D, I = 0, 0, 0
    
    # Mispronunciation Detection and Diagnosis counters
    TA, FR, FA, TR = 0, 0, 0, 0
    CD, DE = 0, 0

    for hyp, trans, canon in zip(hypotheses, transcripts, canonicals):
        # Align hypothesis with transcript for PR Metrics
        a_hyp, a_trans = Align(hyp, trans)
        for h, t in zip(a_hyp, a_trans):
            if t != "<eps>": 
                N += 1
                if h == t:
                    pass # Correct 
                elif h != "<eps>": 
                    S += 1  # Substitution
                else: 
                    D += 1  # Deletion
            else:
                if h != "<eps>": 
                    I += 1  # Insertion

        # Allign hypothesis and transcript with canonical for MDD Metrics
        a_hyp_c, a_canon_h = Align(hyp, canon)
        a_trans_c, a_canon_t = Align(trans, canon)
        
        for idx in range(len(canon)):
            h_phone, t_phone = "<eps>", "<eps>"
            
            curr_c_h, curr_c_t = 0, 0
            for h, c in zip(a_hyp_c, a_canon_h):
                if c != "<eps>":
                    if curr_c_h == idx: h_phone = h; break
                    curr_c_h += 1
            for t, c in zip(a_trans_c, a_canon_t):
                if c != "<eps>":
                    if curr_c_t == idx: t_phone = t; break
                    curr_c_t += 1

            c_phone = canon[idx]
            is_correct_pron = (t_phone == c_phone)  # Correct Pronunciation
            is_detected_corr = (h_phone == c_phone) # Correctly Detected as Correct

            if is_correct_pron:
                if is_detected_corr: TA += 1  # True Acceptance
                else: FR += 1                # False Rejection
            else:
                if is_detected_corr: FA += 1  # False Acceptance
                else:
                    TR += 1                  # True Rejection
                    if h_phone == t_phone: CD += 1  # Correct Diagnosis
                    else: DE += 1                  # Diagnosis Error

    
    # 1. Phone Recognition Metrics
    correctness = (N - S - D) / max(N, 1)
    accuracy = (N - S - D - I) / max(N, 1)
    
    # 2. Mispronunciation Detection and Diagnosis Metrics
    frr = FR / max(TA + FR, 1)
    far = FA / max(FA + TR, 1)
    
    detection_accuracy = (TR + TA) / max(TR + TA + FR + FA, 1)
    precision = TR / max(TR + FR, 1)
    recall = TR / max(TR + FA, 1)  
    
    f_measure = 2 * (precision * recall) / max(precision + recall, 1e-6)
    diagnosis_error_rate = DE / max(DE + CD, 1)

    return {
        "PR_Correctness": correctness,
        "PR_Accuracy": accuracy,
        "MDD_Detection_Accuracy": detection_accuracy,
        "MDD_Precision": precision,
        "MDD_Recall": recall,
        "MDD_F_measure": f_measure,
        "MDD_FAR": far,
        "MDD_FRR": frr,
        "MDD_DER": diagnosis_error_rate
    }