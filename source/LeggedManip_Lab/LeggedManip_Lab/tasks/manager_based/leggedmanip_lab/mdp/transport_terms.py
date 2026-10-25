"""Transport actor/critic observations, tutorial costs, and wrench-based terrain curriculum."""

import torch
from isaaclab.utils.math import quat_apply_inverse

from .transport_math import payload_height_cost, tracking_terms


def term(env):
    return env.action_manager.get_term("transport")


def student_observation(env):
    action = term(env)
    action.update_history()
    return action.student_history.flatten(1)


def teacher_observation(env):
    action = term(env)
    action.update_history()
    return torch.cat((action.clean_history.flatten(1), action.reference()), dim=-1)


def policy_observation(env):
    return teacher_observation(env) if env.cfg.observation_mode == "teacher" else student_observation(env)


def critic_observation(env):
    action = term(env)
    teacher = teacher_observation(env)
    data = action.robot.data
    root, quaternion = data.root_pos_w, data.root_quat_w
    vectors = (data.body_pos_w[:, action.ee_id] - root, data.body_lin_vel_w[:, action.ee_id],
               action.ee_acceleration, action.payload.data.root_pos_w - root,
               action.payload.data.root_lin_vel_w, action.payload_acceleration)
    return torch.cat((teacher, *(quat_apply_inverse(quaternion, value) for value in vectors)), dim=-1)


def force_tracking_cost(env):
    action = term(env)
    return tracking_terms(action.reference(), action.robot.data.root_lin_vel_b,
                          action.robot.data.root_ang_vel_b[:, 2], env.cfg.literal_tracking_rewards)[0]


def torque_tracking_cost(env):
    action = term(env)
    return tracking_terms(action.reference(), action.robot.data.root_lin_vel_b,
                          action.robot.data.root_ang_vel_b[:, 2], env.cfg.literal_tracking_rewards)[1]


def payload_height(env):
    action = term(env)
    relative = quat_apply_inverse(action.robot.data.root_quat_w,
                                  action.payload.data.root_pos_w - action.robot.data.root_pos_w)
    return payload_height_cost(relative[:, 2])


def joint_torque(env):
    action = term(env)
    return action.robot.data.applied_torque[:, action.obs_ids].square().sum(-1)


def joint_dof(env):
    action = term(env)
    data = action.robot.data
    return (data.joint_vel[:, action.obs_ids].square() + 0.00025 * data.joint_acc[:, action.obs_ids].square()).sum(-1)


def action_cost(env):
    action = term(env)
    return ((action.processed_actions - action.previous_hl).square() + 0.5 * action.processed_actions.square()).sum(-1)


def arm_posture(env):
    action = term(env)
    data = action.robot.data
    return torch.linalg.vector_norm(data.joint_pos[:, action.arm_ids] - data.default_joint_pos[:, action.arm_ids], dim=-1)


def termination_cost(env):
    return env.termination_manager.terminated.float()


def terrain_progress(env, env_ids):
    """Use the episode wrench's speed reference; never query a velocity CommandTerm."""
    action = term(env)
    terrain = env.scene.terrain
    distance = torch.linalg.vector_norm(action.robot.data.root_pos_w[env_ids, :2] - action.start_position[env_ids, :2], dim=-1)
    required = torch.linalg.vector_norm(action.wrench_target[env_ids, :2], dim=-1) * (0.9 * env.max_episode_length_s)
    up = distance > terrain.cfg.terrain_generator.size[0] / 2
    down = (distance < 0.5 * required) & ~up & (required > 0.05)
    terrain.update_env_origins(env_ids, up, down)
    return terrain.terrain_levels.float().mean()
