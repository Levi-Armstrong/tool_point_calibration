#include <tool_point_calibration/tool_point_calibration.h>

#include <array>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

/**
 * @brief The ToolPointEstimator struct
 */
struct ToolPointEstimator
{
  ToolPointEstimator(const Eigen::Affine3d& robot_pose)
  {
      ceres::RotationMatrixToAngleAxis(robot_pose.rotation().data(), angle_axis_.data());
      translation_[0] = robot_pose.translation()(0);
      translation_[1] = robot_pose.translation()(1);
      translation_[2] = robot_pose.translation()(2);
  }

  template <typename T>
  bool operator()(const T* const tool, const T* const point, T* residuals) const
  {
    // Convert "angle_axis_" to type T
    std::array<T, 3> angle_axis; // NOLINT
    angle_axis[0] = T(angle_axis_[0]);
    angle_axis[1] = T(angle_axis_[1]);
    angle_axis[2] = T(angle_axis_[2]);

    // Rotate the estimated 'tool' offset and put the result into 'rotated_pt'
    std::array<T, 3> rotated_pt; // NOLINT
    ceres::AngleAxisRotatePoint(angle_axis.data(), tool, rotated_pt.data());

    // Convert "translation_" to type T and calculate the tool point in the world frame
    std::array<T, 3> pos; // NOLINT
    pos[0] = rotated_pt[0] + T(translation_[0]);
    pos[1] = rotated_pt[1] + T(translation_[1]);
    pos[2] = rotated_pt[2] + T(translation_[2]);

    // Compute the error between the current estimate of the tool points
    residuals[0] = point[0] - pos[0];
    residuals[1] = point[1] - pos[1];
    residuals[2] = point[2] - pos[2];
    return true;
  }

  std::array<double, 3> angle_axis_{0, 0, 0};
  std::array<double, 3> translation_{0, 0, 0};
};


tool_point_calibration::TcpCalibrationResult
tool_point_calibration::calibrateTcp(const tool_point_calibration::Affine3dVector &tool_poses,
                                     const Eigen::Vector3d &tcp_guess,
                                     const Eigen::Vector3d &touch_pt_guess)
{
    Eigen::Vector3d internal_tcp_guess = tcp_guess;
    Eigen::Vector3d internal_touch_guess = touch_pt_guess;

    ceres::Problem problem;
    for (const auto& tool_pose : tool_poses)
    {
      // NOLINTNEXTLINE
      auto* cost_fn =
          new ceres::AutoDiffCostFunction<ToolPointEstimator, 3, 3, 3>(
            new ToolPointEstimator( tool_pose )
          );

      problem.AddResidualBlock(cost_fn, nullptr, internal_tcp_guess.data(), internal_touch_guess.data());
    }

    ceres::Solver::Options options;
    ceres::Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);

    TcpCalibrationResult result;
    result.tcp_offset = internal_tcp_guess;
    result.touch_point = internal_touch_guess;
    result.average_residual = summary.final_cost / summary.num_parameter_blocks;
    result.converged = summary.termination_type == ceres::CONVERGENCE;

    return result;
}
