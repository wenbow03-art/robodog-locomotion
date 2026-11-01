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

from legged_gym.envs.lz_q1l.lz_q1l_2 import LZQ1LStandWalkRobot
from legged_gym.envs.lz_q1l.lz_q1l_2_config import LZQ1LStandWalkCfg, LZQ1LStandWalkCfgPPO
task_registry.register('lz_q1l_flat_stand_walk', LZQ1LStandWalkRobot, LZQ1LStandWalkCfg(), LZQ1LStandWalkCfgPPO())

from legged_gym.envs.lz_q1l.lz_q1l_2_config import LZQ1LLegacyLateralCfg, LZQ1LLegacyLateralCfgPPO
task_registry.register('lz_q1l_flat_baseline_v8_lateral', LZQ1LStandWalkRobot, LZQ1LLegacyLateralCfg(), LZQ1LLegacyLateralCfgPPO())

from legged_gym.envs.lz_q1l.lz_q1l_2 import LZQ1LLateralCycleRobot
from legged_gym.envs.lz_q1l.lz_q1l_2_config import LZQ1LLateralCycleCfg, LZQ1LLateralCycleCfgPPO
task_registry.register('lz_q1l_flat_lateral_cycle', LZQ1LLateralCycleRobot, LZQ1LLateralCycleCfg(), LZQ1LLateralCycleCfgPPO())

from legged_gym.envs.lz_q1l.lz_q1l_2_config import LZQ1LV8FourMsCfg, LZQ1LV8FourMsCfgPPO
task_registry.register('lz_q1l_flat_v8_4ms', LZQ1LRobot, LZQ1LV8FourMsCfg(), LZQ1LV8FourMsCfgPPO())

from legged_gym.envs.lz_q1l.lz_q1l_2 import LZQ1LNativeSideRobot
from legged_gym.envs.lz_q1l.lz_q1l_2_config import LZQ1LNativeSideCfg, LZQ1LNativeSideCfgPPO
task_registry.register('lz_q1l_flat_native_side_anchor', LZQ1LNativeSideRobot, LZQ1LNativeSideCfg(), LZQ1LNativeSideCfgPPO())

from legged_gym.envs.lz_q1l.lz_q1l_2 import LZQ1LV4LateralRobot
from legged_gym.envs.lz_q1l.lz_q1l_2_config import LZQ1LFlatV4LateralCfg, LZQ1LFlatV4LateralCfgPPO
task_registry.register('lz_q1l_flat_v4_lateral', LZQ1LV4LateralRobot, LZQ1LFlatV4LateralCfg(), LZQ1LFlatV4LateralCfgPPO())

from legged_gym.envs.lz_q1l.lz_q1l_2_config import LZQ1LNativeSideSpeedCfg, LZQ1LNativeSideSpeedCfgPPO
task_registry.register('lz_q1l_flat_native_side_speed', LZQ1LNativeSideRobot, LZQ1LNativeSideSpeedCfg(), LZQ1LNativeSideSpeedCfgPPO())

from legged_gym.envs.lz_q1l.lz_q1l_2_config import LZQ1LNativeSideHeightCfg, LZQ1LNativeSideHeightCfgPPO
task_registry.register('lz_q1l_flat_native_side_height', LZQ1LNativeSideRobot, LZQ1LNativeSideHeightCfg(), LZQ1LNativeSideHeightCfgPPO())

from legged_gym.envs.lz_q1l.lz_q1l_2_config import LZQ1LNativeSideHeightOneMsCfg, LZQ1LNativeSideHeightOneMsCfgPPO
task_registry.register('lz_q1l_flat_native_side_height1ms', LZQ1LNativeSideRobot, LZQ1LNativeSideHeightOneMsCfg(), LZQ1LNativeSideHeightOneMsCfgPPO())

from legged_gym.envs.lz_q1l.lz_q1l_2 import LZQ1LSideCoordinationRobot
from legged_gym.envs.lz_q1l.lz_q1l_2_config import LZQ1LSideCoordinationCfg, LZQ1LSideCoordinationCfgPPO
task_registry.register('lz_q1l_flat_side_coordination', LZQ1LSideCoordinationRobot, LZQ1LSideCoordinationCfg(), LZQ1LSideCoordinationCfgPPO())

from legged_gym.envs.lz_q1l.lz_q1l_2 import LZQ1LFrontPoseRobot
from legged_gym.envs.lz_q1l.lz_q1l_2_config import LZQ1LFrontPoseCfg, LZQ1LFrontPoseCfgPPO
task_registry.register('lz_q1l_flat_front_pose', LZQ1LFrontPoseRobot, LZQ1LFrontPoseCfg(), LZQ1LFrontPoseCfgPPO())

from legged_gym.envs.lz_q1l.lz_q1l_2 import LZQ1LSoftSlipRobot
from legged_gym.envs.lz_q1l.lz_q1l_2_config import LZQ1LSoftSlipCfg, LZQ1LSoftSlipCfgPPO
task_registry.register('lz_q1l_flat_soft_slip', LZQ1LSoftSlipRobot, LZQ1LSoftSlipCfg(), LZQ1LSoftSlipCfgPPO())

from legged_gym.envs.lz_q1l.lz_q1l_2 import LZQ1LStanceWidthRobot
from legged_gym.envs.lz_q1l.lz_q1l_2_config import LZQ1LStanceWidthCfg, LZQ1LStanceWidthStandingCfgPPO, LZQ1LStanceWidthForwardCfgPPO
task_registry.register('lz_q1l_flat_stance_width_standing', LZQ1LStanceWidthRobot, LZQ1LStanceWidthCfg(), LZQ1LStanceWidthStandingCfgPPO())
task_registry.register('lz_q1l_flat_stance_width_forward', LZQ1LStanceWidthRobot, LZQ1LStanceWidthCfg(), LZQ1LStanceWidthForwardCfgPPO())

from legged_gym.envs.lz_q1l.lz_q1l_2_config import LZQ1LForwardSpeedCfg, LZQ1LForwardSpeedCfgPPO
task_registry.register('lz_q1l_flat_forward_speed', LZQ1LStanceWidthRobot, LZQ1LForwardSpeedCfg(), LZQ1LForwardSpeedCfgPPO())

from legged_gym.envs.lz_q1l.lz_q1l_2_config import LZQ1LForwardSpeedTwoCfg, LZQ1LForwardSpeedTwoCfgPPO
task_registry.register('lz_q1l_flat_forward_speed2', LZQ1LStanceWidthRobot, LZQ1LForwardSpeedTwoCfg(), LZQ1LForwardSpeedTwoCfgPPO())

from legged_gym.envs.lz_q1l.lz_q1l_2 import LZQ1LForwardYawRobot
from legged_gym.envs.lz_q1l.lz_q1l_2_config import LZQ1LForwardYawCfg, LZQ1LForwardYawCfgPPO
task_registry.register('lz_q1l_flat_forward_yaw', LZQ1LForwardYawRobot, LZQ1LForwardYawCfg(), LZQ1LForwardYawCfgPPO())

from legged_gym.envs.lz_q1l.lz_q1l_2 import LZQ1LForwardAccelerationRobot
from legged_gym.envs.lz_q1l.lz_q1l_2_config import LZQ1LForwardAccelerationCfg, LZQ1LForwardAccelerationCfgPPO
task_registry.register('lz_q1l_flat_forward_acceleration', LZQ1LForwardAccelerationRobot, LZQ1LForwardAccelerationCfg(), LZQ1LForwardAccelerationCfgPPO())

from legged_gym.envs.lz_q1l.lz_q1l_2_config import LZQ1LForwardMainActorCfg, LZQ1LForwardMainActorCfgPPO
task_registry.register('lz_q1l_flat_forward_main_actor', LZQ1LForwardAccelerationRobot, LZQ1LForwardMainActorCfg(), LZQ1LForwardMainActorCfgPPO())

from legged_gym.envs.lz_q1l.lz_q1l_2_config import LZQ1LContinuousRHCfgPPO
task_registry.register('lz_q1l_flat_continuous_rh', LZQ1LForwardAccelerationRobot, LZQ1LForwardAccelerationCfg(), LZQ1LContinuousRHCfgPPO())
