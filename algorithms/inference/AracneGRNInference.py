from algorithms.inference.AbstractGRNInferrence import AbstractGRNInferrence
import anndata as ad
import numpy as np
import itertools
import scipy.sparse as scs
import os.path as op
import os
import shutil
import pandas as pd
import subprocess

class AracneGRNInference(AbstractGRNInferrence):
    def __init__(self, data: ad.AnnData) -> None:
        super().__init__(data)

    def _get_top_k_edges(self):
        i  = self.data.uns['current_iterations']
        for GRN in self.data.uns["GRNs"][f"iteration_{i!r}"]:
            dok = self.data.varp[GRN]
            

    def _infer_cluster_specific_GRNS(self) -> None:
        """
        Aracne is a commandline tool available online. This wrapper proceeds as follows.

        1. Write out the cluster specific gene expression matrices ().
        2. Run arachne for each sub data set
        3. Find unique GRNs for each submodule
        4. 
        """
        self.before_run()
        # Check if GRNs key is available
        if not "GRNs" in self.data.uns.keys():
            self.data.uns["GRNs"] = {}

        # Get the current global iteration
        i = self.data.uns["current_iteration"]
        self.data.uns["GRNs"]["iteration_" + str(i)] = {}

        
        for lab in np.unique(self.data.obs['current_clustering']):
            # set the current cluster key
            self.data.uns["GRNs"][f"iteration_{i!r}"][f"cluster_{lab!r}"] = f"iteration{i!r}_cluster{lab!r}"
           
            cluster_output_dir = os.path.join(self.data.uns['GRN_dir'], 'temp', 'cluster_'+str(lab))
            os.makedirs(cluster_output_dir)
            #1. Save the data files
            expression_path = os.path.join(cluster_output_dir,  'data_cluster_'+str(lab)+'.tsv')
            expression_data = pd.DataFrame(self.data.X[self.data.obs['current_clustering'] == lab].toarray().T)
            expression_data.index = self.data.var.index
            expression_data.columns = self.data.obs[self.data.obs['current_clustering'] == lab].index
            expression_data.to_csv(expression_path, sep='\t', header=True, index=True)
            

            #2. Filter the TF file, so only genes contained in the expression data are included.
            ktf_path = os.path.join(self.data.uns['input.TF_file'])
            regulators = pd.read_csv(ktf_path, delimiter='\t', header=0)
            regulators = list(np.unique(regulators['source']))
            regulators = set(regulators).intersection(set(expression_data.index))
            regulator_path = os.path.join(cluster_output_dir,  f'regulators.txt')
            pd.DataFrame(regulators).to_csv(regulator_path, sep='\t', index=False, header=False)

            #3. Run Aracne-AP
            # The command requires to execute two separate steps.
            # hardcoded by Aracne-AP
            network_output_file = os.path.join(cluster_output_dir, 'nobootstrap_network.txt')
            exe = os.path.join(self.data.uns['external.directory'], 'ARACNe-AP', 'dist', 'aracne.jar')
            thresholdCommand = f'java -Xmx5G -jar {exe} -e {expression_path}  -o {cluster_output_dir} --tfs {regulator_path} --seed {self.data.uns["seed"]} --pvalue {self.data.uns["pvalue"]} --calculateThreshold'
            subprocess.run(thresholdCommand, shell=True)
            command = f'java -Xmx5G -jar {exe} -e {expression_path}  -o {cluster_output_dir} --tfs {regulator_path} --pvalue {self.data.uns["pvalue"]} --nobootstrap'
            subprocess.run(command, shell=True)

            # get results
            network = pd.read_csv(network_output_file, sep='\t', index_col=False)
            network = network.sort_values(by=['MI', 'Regulator', 'Target'], axis=0, ascending=False)
            
            network = network.rename({'Regulator': 'source', 'Target': 'target', 'MI': 'score'}, axis='columns')
            network['type'] = 'directed'

            # initialize the sparse gene module as coo matrix 
            varp = scs.dok_array((self.data.shape[1], self.data.shape[1]))
            for s,t, m in zip(network['source'], network['target'], network['score']):
                index_of_s = np.where(self.data.var.index == s)[0][0]
                index_of_t = np.where(self.data.var.index == t)[0][0]
                varp[index_of_s, index_of_t] = m
            
            # transform to csr and insert fo the current iteration.
            varp = scs.csr_matrix(varp)
            self.data.varp[f"iteration{i!r}_cluster{lab!r}"] = varp

        if i > 1:
            im2 = i - 2
            for GRN in self.data.uns["GRNs"][f"iteration_{im2!r}"]:
                del self.data.varp[self.data.uns["GRNs"][f"iteration_{im2!r}"][GRN]]
            del self.data.uns["GRNs"][f"iteration_{im2!r}"]

    def _write_results(self) -> None:
        pass

    def _check_GRN_convergence(self, consistency) -> bool:
        pass

    def before_run(self)-> bool:
        '''
        1. Clean up of the previous iteration. This is done in the preflight so the final results are preserved
        and we only need to copy at the end.
        2. Run some preflight checks, on whether the random seed has ben set and the files are available.
        '''
        if not 'seed' in self.data.uns.keys():
            self.data.uns['seed'] = 11

        if not 'pvalue' in self.data.uns.keys():
            self.data.uns['pvalue'] = 1e-4

        # remove temporary directory
        if  op.exists(op.join(self.data.uns['GRN_dir'], 'temp')):
            shutil.rmtree(path = op.join(self.data.uns['GRN_dir'], 'temp'), ignore_errors=True)


        if not op.exists(op.join(self.data.uns['GRN_dir'], 'temp')):
            os.mkdir(os.path.join(self.data.uns['GRN_dir'], 'temp'))
        
        print(os.path.join(self.data.uns['external.directory'], 'ARACNe-AP'))
        if not os.path.exists(os.path.join(self.data.uns['external.directory'], 'ARACNe-AP')):
            raise FileNotFoundError('Please download algorithms directory from https://github.com/bionetslab/grn-confounders to use predefined wrappers.')


    def after_run(self)-> bool:
        '''
        Run some clean up activities to remove the temporary files if required.
        '''
        pass


