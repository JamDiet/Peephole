import torch

def compare_features(vec1: torch.Tensor,
                     vec2: torch.Tensor,
                     threshold: float=.9):
    '''
    Determine if 2 feature embeddings represent the same object
    based on cosine similarity.

    Parameters
    ----------
    vec1 : torch.Tensor
        First feature vector.
    vec2 : torch.Tensor
        Second feature vector.
    threshold : float, optional
        Returns True if cosine similarity is greater than or
        equal to this number.
    
    Returns
    -------
    bool
        True if vectors are similar enough, False otherwise.
    '''
    vec1_norm = torch.linalg.norm(vec1)
    vec2_norm = torch.linalg.norm(vec2)
    cosine_similarity = torch.dot(vec1, vec2) / (vec1_norm * vec2_norm)
    
    if cosine_similarity >= threshold:
        return True
    else:
        return False