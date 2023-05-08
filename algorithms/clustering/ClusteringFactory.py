from algorithms.Strategy import ClusteringUpdateStategy
from algorithms.clustering.BasicClustering import BasicClustering

class ClusteringFactory:
    
    def create_clustering_wrapper(self, type: ClusteringUpdateStategy):
        if type == ClusteringUpdateStategy.BASIC:
            return BasicClustering()
        

