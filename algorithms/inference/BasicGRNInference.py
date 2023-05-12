from algorithms.inference.AbstractGRNInferrence import AbstractGRNInferrence

class BasicGRNInference(AbstractGRNInferrence):
    def __init__(self, data) -> None:
        super().__init__(data)

    def _get_top_k_edges():
        return 0
    

    def _infer_cluster_specific_GRNS(self, cluster_labels) -> None:
        return super()._infer_cluster_specific_GRNS(cluster_labels)
    
    def _write_results(self):
        return super()._write_results()
    
    def _check_GRN_convergence(self, tolerance):
        return super()._check_GRN_convergence(tolerance)
    



