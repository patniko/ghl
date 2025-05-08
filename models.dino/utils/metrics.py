import torch

def compute_metrics(student_output, teacher_output):
    # Cosine similarity between student and teacher projections
    sim = torch.nn.functional.cosine_similarity(student_output, teacher_output)
    return {'cosine_sim': sim.mean().item()}