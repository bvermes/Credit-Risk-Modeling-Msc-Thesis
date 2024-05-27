from lgd.lgd_builder_engine import LGDBuilderEngine
from pd.pd_builder_engine import PDBuilderEngine
from deployer_engine import DeployerEngine
from setup import pd_pre_behav_config, pd_post_behav_config, lgd_pre_behav_config, lgd_post_behav_config, param_grids
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



if __name__ == "__main__":

    replace_with_most_recent_model = True
    

    pd_pre_behav_BuilderEngine = PDBuilderEngine(config=pd_pre_behav_config, param_grids=param_grids)
    pd_pre_behav_BuilderEngine.run()
    
    pd_post_behav_BuilderEngine = PDBuilderEngine(config=pd_post_behav_config, param_grids=param_grids)
    pd_post_behav_BuilderEngine.run()
    
    lgdBuilderEngine = LGDBuilderEngine(config=lgd_pre_behav_config, param_grids=param_grids)
    lgdBuilderEngine.run()
    
    lgdBuilderEngine = LGDBuilderEngine(config=lgd_post_behav_config, param_grids=param_grids)
    lgdBuilderEngine.run()
    # 
    #source = lgdBuilderEngine.output_path + '../'
    source = "results/2024_05_16_21_13_06_all_models/"
    deployerEngine = DeployerEngine(source=source)
    deployerEngine.run()
    
    
