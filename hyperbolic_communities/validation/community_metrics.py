from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
def dyn_nmi(y_true, y_pred): return normalized_mutual_info_score(y_true, y_pred)
def dyn_ari(y_true, y_pred): return adjusted_rand_score(y_true, y_pred)
