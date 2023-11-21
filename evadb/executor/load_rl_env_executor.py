import pandas as pd
import gymnasium as gym

from evadb.database import EvaDBDatabase
from evadb.executor.abstract_executor import AbstractExecutor
from evadb.executor.executor_utils import ExecutorError
from evadb.models.storage.batch import Batch
from evadb.plan_nodes.load_data_plan import LoadDataPlan
from evadb.storage.storage_engine import StorageEngine
from evadb.utils.logging_manager import logger


class LoadRLEnvExecutor(AbstractExecutor):
    def __init__(self, db: EvaDBDatabase, node: LoadDataPlan):
        super().__init__(db, node)

    def exec(self, *args, **kwargs):
        """
        Create a RL environment using gym and persist the meta-data
        using storage engine
        """

        # Check table existence
        table_info = self.node.table_info
        database_name = table_info.database_name
        table_name = table_info.table_name
        table_obj = self.catalog().get_table_catalog_entry(
            table_name,
            database_name,
        )
        if table_obj is not None:
            error = f"{table_name} already existes."
            logger.error(error)
            raise ExecutorError(error)
        else:
            table_obj = self.catalog().create_and_insert_multimedia_table_catalog_entry(table_name, self.node.file_options["file_format"])
        
        env_name = str(self.node.file_path)

        try:
            env = gym.make(env_name)
        except:
            err_msg = "Env {} not found in Gym".format(env_name)
            raise Exception(err_msg)
        else:
            storage_engine = StorageEngine.factory(self.db, table_obj)
            storage_engine.create(table_obj)

            env_meta_data = {
                "name": env_name,
                "action_dim": gym.spaces.utils.flatdim(env.action_space),
                "observation_dim": gym.spaces.utils.flatdim(env.observation_space),
            }
            storage_engine.write(
                table_obj,
                Batch(pd.DataFrame([env_meta_data])),
            )

            df_yield_result = Batch(
                pd.DataFrame(
                    [env_meta_data]
                )
            )

            yield df_yield_result
