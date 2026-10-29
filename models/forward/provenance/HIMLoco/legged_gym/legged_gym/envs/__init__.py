# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .base.legged_robot import LeggedRobot
from .a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from legged_gym.envs.go1.go1_config import Go1RoughCfg, Go1RoughCfgPPO
from legged_gym.envs.aliengo.aliengo_config import AlienGoRoughCfg, AlienGoRoughCfgPPO
from legged_gym.envs.lz_q1l.lz_q1l_config import LZQ1LFlatCfg, LZQ1LFlatCfgPPO, LZQ1LFlatV2Cfg, LZQ1LFlatV2CfgPPO, LZQ1LFlatV3Cfg, LZQ1LFlatV3CfgPPO, LZQ1LFlatV4Cfg, LZQ1LFlatV4CfgPPO, LZQ1LFlatV5Cfg, LZQ1LFlatV5CfgPPO
from legged_gym.envs.lz_q1l.lz_q1l import LZQ1LRobot
from legged_gym.envs.lz_q1l.lz_q1l_config import LZQ1LFlatV6Cfg, LZQ1LFlatV6CfgPPO
from legged_gym.envs.lz_q1l.lz_q1l_config import LZQ1LFlatV7Cfg, LZQ1LFlatV7CfgPPO
from legged_gym.envs.lz_q1l.lz_q1l_config import LZQ1LFlatV8Cfg, LZQ1LFlatV8CfgPPO
from legged_gym.envs.lz_q1l.lz_q1l_2_config import LZQ1LFlatV1OneMsCfg, LZQ1LFlatV1OneMsCfgPPO
from legged_gym.envs.lz_q1l.lz_q1l_2 import LZQ1LV1SupportRobot, LZQ1LV2UnifiedRobot, LZQ1LV2GaitRobot, LZQ1LV2GaitHeightRobot
from legged_gym.envs.lz_q1l.lz_q1l_2_config import LZQ1LFlatV1SupportCfg, LZQ1LFlatV1SupportCfgPPO
from legged_gym.envs.lz_q1l.lz_q1l_2_config import LZQ1LFlatV2UnifiedCfg, LZQ1LFlatV2UnifiedCfgPPO
from legged_gym.envs.lz_q1l.lz_q1l_2_config import LZQ1LFlatV2PostureCfg, LZQ1LFlatV2PostureCfgPPO
from legged_gym.envs.lz_q1l.lz_q1l_2_config import LZQ1LFlatV2GaitCfg, LZQ1LFlatV2GaitCfgPPO
from legged_gym.envs.lz_q1l.lz_q1l_2_config import LZQ1LFlatV2GaitHeightCfg, LZQ1LFlatV2GaitHeightCfgPPO
from legged_gym.envs.lz_q1l.lz_q1l_2_config import LZQ1LFlatV2UprightCfg, LZQ1LFlatV2UprightCfgPPO
from legged_gym.envs.lz_q1l.lz_q1l_2 import LZQ1LV2StandFrontRobot
from legged_gym.envs.lz_q1l.lz_q1l_2_config import LZQ1LFlatV2StandFrontCfg, LZQ1LFlatV2StandFrontCfgPPO

import os

from legged_gym.utils.task_registry import task_registry

task_registry.register( "a1", LeggedRobot, A1RoughCfg(), A1RoughCfgPPO() )
task_registry.register( "go1", LeggedRobot, Go1RoughCfg(), Go1RoughCfgPPO() )
task_registry.register( "aliengo", LeggedRobot, AlienGoRoughCfg(), AlienGoRoughCfgPPO() )
task_registry.register( "lz_q1l_flat", LeggedRobot, LZQ1LFlatCfg(), LZQ1LFlatCfgPPO() )
task_registry.register( "lz_q1l_flat_v1_1ms", LeggedRobot, LZQ1LFlatV1OneMsCfg(), LZQ1LFlatV1OneMsCfgPPO() )
task_registry.register( "lz_q1l_flat_v1_support", LZQ1LV1SupportRobot, LZQ1LFlatV1SupportCfg(), LZQ1LFlatV1SupportCfgPPO() )
task_registry.register( "lz_q1l_flat_v2_unified", LZQ1LV2UnifiedRobot, LZQ1LFlatV2UnifiedCfg(), LZQ1LFlatV2UnifiedCfgPPO() )
task_registry.register( "lz_q1l_flat_v2_posture", LZQ1LV2UnifiedRobot, LZQ1LFlatV2PostureCfg(), LZQ1LFlatV2PostureCfgPPO() )
task_registry.register( "lz_q1l_flat_v2_gait", LZQ1LV2GaitRobot, LZQ1LFlatV2GaitCfg(), LZQ1LFlatV2GaitCfgPPO() )
task_registry.register( "lz_q1l_flat_v2_gait_height", LZQ1LV2GaitHeightRobot, LZQ1LFlatV2GaitHeightCfg(), LZQ1LFlatV2GaitHeightCfgPPO() )
task_registry.register( "lz_q1l_flat_v2_upright", LZQ1LV2GaitHeightRobot, LZQ1LFlatV2UprightCfg(), LZQ1LFlatV2UprightCfgPPO() )
task_registry.register( "lz_q1l_flat_v2_stand_front", LZQ1LV2StandFrontRobot, LZQ1LFlatV2StandFrontCfg(), LZQ1LFlatV2StandFrontCfgPPO() )
task_registry.register( "lz_q1l_flat_v2", LZQ1LRobot, LZQ1LFlatV2Cfg(), LZQ1LFlatV2CfgPPO() )
task_registry.register( "lz_q1l_flat_v3", LZQ1LRobot, LZQ1LFlatV3Cfg(), LZQ1LFlatV3CfgPPO() )
task_registry.register( "lz_q1l_flat_v4", LZQ1LRobot, LZQ1LFlatV4Cfg(), LZQ1LFlatV4CfgPPO() )
task_registry.register( "lz_q1l_flat_v5", LZQ1LRobot, LZQ1LFlatV5Cfg(), LZQ1LFlatV5CfgPPO() )
task_registry.register( "lz_q1l_flat_v6", LZQ1LRobot, LZQ1LFlatV6Cfg(), LZQ1LFlatV6CfgPPO() )
task_registry.register( "lz_q1l_flat_v7", LZQ1LRobot, LZQ1LFlatV7Cfg(), LZQ1LFlatV7CfgPPO() )
task_registry.register( "lz_q1l_flat_v8", LZQ1LRobot, LZQ1LFlatV8Cfg(), LZQ1LFlatV8CfgPPO() )

from legged_gym.envs.lz_q1l.lz_q1l_2 import LZQ1LV2BalanceRollRobot
from legged_gym.envs.lz_q1l.lz_q1l_2_config import LZQ1LFlatV2BalanceRollCfg, LZQ1LFlatV2BalanceRollCfgPPO
task_registry.register("lz_q1l_flat_v2_balance_roll", LZQ1LV2BalanceRollRobot, LZQ1LFlatV2BalanceRollCfg(), LZQ1LFlatV2BalanceRollCfgPPO())

from legged_gym.envs.lz_q1l.lz_q1l_2_config import LZQ1LFlatV2StandLoadCfg, LZQ1LFlatV2StandLoadCfgPPO
task_registry.register("lz_q1l_flat_v2_stand_load", LZQ1LV2GaitHeightRobot, LZQ1LFlatV2StandLoadCfg(), LZQ1LFlatV2StandLoadCfgPPO())

from legged_gym.envs.lz_q1l.lz_q1l_2 import LZQ1LV2SideLoadRobot
from legged_gym.envs.lz_q1l.lz_q1l_2_config import LZQ1LFlatV2SideLoadCfg, LZQ1LFlatV2SideLoadCfgPPO
task_registry.register("lz_q1l_flat_v2_side_load", LZQ1LV2SideLoadRobot, LZQ1LFlatV2SideLoadCfg(), LZQ1LFlatV2SideLoadCfgPPO())

from legged_gym.envs.lz_q1l.lz_q1l_2_config import LZQ1LFlatV2SideLoadStrongCfg, LZQ1LFlatV2SideLoadStrongCfgPPO
task_registry.register("lz_q1l_flat_v2_side_load_strong", LZQ1LV2SideLoadRobot, LZQ1LFlatV2SideLoadStrongCfg(), LZQ1LFlatV2SideLoadStrongCfgPPO())

from legged_gym.envs.lz_q1l.lz_q1l_2_config import LZQ1LFlatV3RearTeacherCfg, LZQ1LFlatV3RearTeacherCfgPPO
task_registry.register("lz_q1l_flat_v3_rear_teacher", LZQ1LV2GaitRobot, LZQ1LFlatV3RearTeacherCfg(), LZQ1LFlatV3RearTeacherCfgPPO())

from legged_gym.envs.lz_q1l.lz_q1l_2_config import LZQ1LFlatV3RightRearOnlyCfg, LZQ1LFlatV3RightRearOnlyCfgPPO
task_registry.register("lz_q1l_flat_v3_right_rear_only", LZQ1LV2GaitRobot, LZQ1LFlatV3RightRearOnlyCfg(), LZQ1LFlatV3RightRearOnlyCfgPPO())

from legged_gym.envs.lz_q1l.lz_q1l_2_config import LZQ1LFlatV3RightRearSplitCfgPPO
task_registry.register("lz_q1l_flat_v3_right_rear_split", LZQ1LV2GaitRobot, LZQ1LFlatV3RightRearOnlyCfg(), LZQ1LFlatV3RightRearSplitCfgPPO())
