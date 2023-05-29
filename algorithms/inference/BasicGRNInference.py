from algorithms.inference.AbstractGRNInferrence import AbstractGRNInferrence
import anndata as ad
import numpy as np
import itertools
import scipy.sparse as scs
import os.path as op


class BasicGRNInference(AbstractGRNInferrence):

    """
    This Basic GRN inferrence method generates random GRN per cluster by
    1. Sampling a number of Genes for each module.
    2. Sampling a number of Edges for the modules.

    The GRN is represented as a list of edges.
    """

    def __init__(self, data: ad.AnnData) -> None:
        super().__init__(data)

    def _get_top_k_edges():
        return 0

    def _infer_cluster_specific_GRNS(self) -> None:
        if not "GRNs" in self.data.uns.keys():
            self.data.uns["GRNs"] = {}

        i = self.data.uns["current_iteration"]
        self.data.uns["GRNs"]["iteration_" + str(i)] = {}
        for lab in range(self.data.uns["algorithm.n_clusters"]):
            self.data.uns["GRNs"][f"iteration_{i!r}"][f"cluster_{lab!r}"] = f"iteration{i!r}_cluster{lab!r}"
            index_of = np.random.choice(len(self.data.var.index), size=100, replace=False)
            binomials = np.random.binomial(n=1, p=0.2, size=100 * 99)

            # initialize the sparse gene module as dok matrix and insert elements
            varp = scs.dok_array((self.data.shape[1], self.data.shape[1]))
            for elem, bin in zip(itertools.product(index_of, index_of), binomials.astype(bool)):
                if bin > 0:
                    print(elem)
                    varp[elem] = 1
            # transform to csr and insert fo the current iteration.
            varp = scs.csr_matrix(varp)
            self.data.varp[f"iteration{i!r}_cluster{lab!r}"] = varp

        if i > 1:
            im2 = i - 2
            for GRN in self.data.uns["GRNs"][f"iteration_{im2!r}"]:
                del self.data.varp[self.data.uns["GRNs"][f"iteration_{im2!r}"][GRN]]
            del self.data.uns["GRNs"][f"iteration_{im2!r}"]

    def _write_results(self) -> None:

        """
        Write the last GNRs to file in a sparse matrix format. One file per GRN.
        """
        try:
            i = self.data.uns["current_iteration"] - 1
            for GRN in self.data.uns["GRNs"][f"iteration_{i!r}"].values():
                print(GRN)
                filename = op.join(self.data.uns["GNR_dir"], f"{GRN}.npz")
                scs.save_npz(file=filename, matrix=self.data.varp[GRN])
        except KeyError:
            print("Results not initialized")

    def _check_GRN_convergence(self, consistency) -> bool:
        """
        This method checks if the GRNs have converged by checking the edges. For each cluster, first, the overlap between the edges in the
        new and the old GRN are computed. For convergence it is required that the consitent edges make up a certain percentage of
        the old and the new GRN respectively. For example, the number of consitent edges is 75, and the consitency parameter is 0.75.
        A GRN would have converged, if the total number of edges in the old and the new GRNs are both smaller or equal
        100, as 75% of the genes are identical. If the GRNs for all clusters have converged, the function returns True.

        Arguments:
        consistency: the required fraction of edges that have to be consitent between two GRNs for one cluster.
        """

        i = self.data.uns["current_iteration"]
        number_of_converged_clusters = 0
        if i >= 1:
            im1 = i - 1
            for GRN_old, GRN_new in zip(self.data.uns["GRNs"][f"iteration_{im1!r}"]):
                # multiply the matrices, which is basically an element wise and operation
                number_of_consistent_edges = int(
                    np.sum(
                        self.data.varp[self.data.uns["GRNs"][f"iteration_{im1!r}"][GRN_old]].multiply(
                            self.data.varp[self.data.uns["GRNs"][f"iteration_{i!r}"][GRN_new]]
                        )
                    )
                )

                if (
                    int(np.sum(self.data.varp[self.data.uns["GRNs"][f"iteration_{im1!r}"][GRN_old]])) * consistency
                    >= number_of_consistent_edges
                    and int(np.sum(self.data.varp[self.data.uns["GRNs"][f"iteration_{i!r}"][GRN_new]])) * consistency
                    >= number_of_consistent_edges
                ):
                    number_of_converged_clusters = number_of_converged_clusters + 1

        # Check of thr GRNs for all clusters have converged.
        if number_of_converged_clusters == self.data.uns["algorithm.n_clusters"]:
            return True
        else:
            return False
