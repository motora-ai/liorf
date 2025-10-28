#include "utility.h" // Assumed to provide ParamServer, imuConverter(), ROS_TIME(), etc.

// GTSAM Includes
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

// ROS/TF/PCL Includes (for wrapper classes)
#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>

// Standard Library Includes
#include <mutex>
#include <deque>
#include <string>
#include <memory>  // For std::unique_ptr
#include <optional> // For std::optional (requires C++17)

using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)


// =======================================================================
//   ROS-Agnostic Data Structures
// =======================================================================

/**
 * @brief Plain data structure for IMU data. No ROS dependencies.
 */
struct CoreImuData
{
    double timestamp;
    gtsam::Vector3 linear_acceleration;
    gtsam::Vector3 angular_velocity;

    // We can keep the original message for the IMU handler to use in its twist calculation
    sensor_msgs::Imu originalImuMsg; 
};

/**
 * @brief Plain data structure for odometry correction. No ROS dependencies.
 */
struct CoreOdomCorrectionData
{
    double timestamp;
    gtsam::Pose3 pose;
    bool degenerate;
};

/**
 * @brief Plain data structure for publishing calculated odometry.
 */
struct CoreCalculatedOdometry
{
    double timestamp;
    gtsam::Pose3 lidarPose;
    gtsam::Vector3 velocity;
    gtsam::Vector3 angular_velocity_biased; // Gyro + bias
    ros::Time originalStamp; // Need this for header
};

/**
 * @brief Core odometry data for TransformFusion.
 * We include the original ROS message as a "template" for the
 * ROS wrapper to fill and publish, but the core logic only uses
 * timestamp and pose.
 */
struct CoreOdomData
{
    double timestamp;
    Eigen::Affine3f pose;
    nav_msgs::Odometry originalOdomMsg;
};


// =======================================================================
//   TransformFusion :: Core Logic
// =======================================================================

class TransformFusionCore
{
public:
    Eigen::Affine3f lidarOdomAffine_;
    double lidarOdomTime_ = -1;
    std::deque<CoreOdomData> imuOdomQueue_;
    Eigen::Affine3f lidar2Baselink_;
    bool useLidar2Baselink_ = false;

    /**
     * @brief Constructor is ROS-agnostic.
     * @param lidar2Baselink The transform to apply, if any.
     * @param useTransform Whether to apply the transform.
     */
    TransformFusionCore(const Eigen::Affine3f& lidar2Baselink, bool useTransform)
        : lidar2Baselink_(lidar2Baselink), useLidar2Baselink_(useTransform)
    {
        lidarOdomAffine_ = Eigen::Affine3f::Identity();
    }

    /**
     * @brief Updates the latest known lidar odometry.
     */
    void processLidarOdometry(double timestamp, const Eigen::Affine3f& pose)
    {
        lidarOdomTime_ = timestamp;
        lidarOdomAffine_ = pose;
    }

    /**
     * @brief Processes an incremental IMU odometry message and fuses it.
     * @param imuOdom The core IMU odometry data packet.
     * @return A pair containing the fused odometry message (for publishing)
     * and the final fused pose (for TF broadcasting).
     */
    std::optional<std::pair<nav_msgs::Odometry, Eigen::Affine3f>>
    processImuOdometry(const CoreOdomData& imuOdom)
    {
        imuOdomQueue_.push_back(imuOdom);

        // get latest odometry (at current IMU stamp)
        if (lidarOdomTime_ == -1)
            return std::nullopt; // Haven't received lidar odom yet

        while (!imuOdomQueue_.empty())
        {
            if (imuOdomQueue_.front().timestamp <= lidarOdomTime_)
                imuOdomQueue_.pop_front();
            else
                break;
        }

        if (imuOdomQueue_.empty())
            return std::nullopt; // Should not happen if check above is correct, but good safety

        Eigen::Affine3f imuOdomAffineFront = imuOdomQueue_.front().pose;
        Eigen::Affine3f imuOdomAffineBack = imuOdomQueue_.back().pose;
        Eigen::Affine3f imuOdomAffineIncre = imuOdomAffineFront.inverse() * imuOdomAffineBack;
        Eigen::Affine3f imuOdomAffineLast = lidarOdomAffine_ * imuOdomAffineIncre;

        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(imuOdomAffineLast, x, y, z, roll, pitch, yaw);
        
        // Prepare the odometry message to return
        nav_msgs::Odometry fusedOdometry = imuOdom.originalOdomMsg;
        fusedOdometry.pose.pose.position.x = x;
        fusedOdometry.pose.pose.position.y = y;
        fusedOdometry.pose.pose.position.z = z;
        fusedOdometry.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);

        return std::make_pair(fusedOdometry, imuOdomAffineLast);
    }
};


// =======================================================================
//   TransformFusion :: ROS Wrapper
// =======================================================================

class TransformFusionROS : public ParamServer
{
public:
    std::mutex mtx_;
    std::unique_ptr<TransformFusionCore> core_;

    ros::Subscriber subImuOdometry_;
    ros::Subscriber subLaserOdometry_;

    ros::Publisher pubImuOdometry_;
    ros::Publisher pubImuPath_;

    tf::TransformListener tfListener_;
    tf::TransformBroadcaster tfOdom2BaseLink_;
    tf::TransformBroadcaster tfMap2Odom_;
    tf::Transform map_to_odom_;
    tf::StampedTransform lidar2Baselink_;

    nav_msgs::Path imuPath_;
    double last_path_time_ = -1;
    double lidarOdomTime_cache_ = -1; // Cache for path pruning

    TransformFusionROS()
    {
        Eigen::Affine3f lidar2BaselinkAffine = Eigen::Affine3f::Identity();
        bool useTransform = (lidarFrame != baselinkFrame);

        if(useTransform)
        {
            try
            {
                tfListener_.waitForTransform(lidarFrame, baselinkFrame, ros::Time(0), ros::Duration(3.0));
                tfListener_.lookupTransform(lidarFrame, baselinkFrame, ros::Time(0), lidar2Baselink_);
                
                // Convert TF transform to Eigen::Affine3f for the core
                lidar2BaselinkAffine = tf2affine(lidar2Baselink_);
            }
            catch (tf::TransformException &ex)
            {
                ROS_ERROR("%s",ex.what());
            }
        }
        
        // Initialize the core logic
        core_ = std::make_unique<TransformFusionCore>(lidar2BaselinkAffine, useTransform);

        // Initialize ROS interface
        subLaserOdometry_ = nh.subscribe<nav_msgs::Odometry>("liorf/mapping/odometry", 5, &TransformFusionROS::lidarOdometryHandler, this, ros::TransportHints().tcpNoDelay());
        subImuOdometry_   = nh.subscribe<nav_msgs::Odometry>(odomTopic+"_incremental", 2000, &TransformFusionROS::imuOdometryHandler, this, ros::TransportHints().tcpNoDelay());

        pubImuOdometry_ = nh.advertise<nav_msgs::Odometry>(odomTopic, 2000);
        pubImuPath_     = nh.advertise<nav_msgs::Path>("liorf/imu/path", 1);
        
        map_to_odom_ = tf::Transform(tf::createQuaternionFromRPY(0, 0, 0), tf::Vector3(0, 0, 0));
    }

    /**
     * @brief Helper to convert nav_msgs::Odometry to Eigen::Affine3f
     */
    Eigen::Affine3f odom2affine(const nav_msgs::Odometry& odom)
    {
        double x, y, z, roll, pitch, yaw;
        x = odom.pose.pose.position.x;
        y = odom.pose.pose.position.y;
        z = odom.pose.pose.position.z;
        tf::Quaternion orientation;
        tf::quaternionMsgToTF(odom.pose.pose.orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        return pcl::getTransformation(x, y, z, roll, pitch, yaw);
    }

    /**
     * @brief Helper to convert tf::Transform to Eigen::Affine3f
     */
    Eigen::Affine3f tf2affine(const tf::Transform& trans)
    {
        Eigen::Affine3f T = Eigen::Affine3f::Identity();
        tf::Vector3 RPY;
        tf::Matrix3x3(trans.getRotation()).getRPY(RPY.m_floats[0], RPY.m_floats[1], RPY.m_floats[2]);
        T = pcl::getTransformation(
            trans.getOrigin().x(), trans.getOrigin().y(), trans.getOrigin().z(),
            RPY.m_floats[0], RPY.m_floats[1], RPY.m_floats[2]);
        return T;
    }

    void lidarOdometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg)
    {
        std::lock_guard<std::mutex> lock(mtx_);
        
        double timestamp = odomMsg->header.stamp.toSec();
        Eigen::Affine3f pose = odom2affine(*odomMsg);
        
        lidarOdomTime_cache_ = timestamp; // Cache for path pruning
        core_->processLidarOdometry(timestamp, pose);
    }

    void imuOdometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg)
    {
        // 1. Broadcast static map->odom TF (ROS-specific)
        tfMap2Odom_.sendTransform(tf::StampedTransform(map_to_odom_, odomMsg->header.stamp, mapFrame, odometryFrame));

        std::lock_guard<std::mutex> lock(mtx_);

        // 2. Prepare data for core logic
        CoreOdomData coreData;
        coreData.timestamp = odomMsg->header.stamp.toSec();
        coreData.pose = odom2affine(*odomMsg);
        coreData.originalOdomMsg = *odomMsg;

        // 3. Call core logic
        auto result = core_->processImuOdometry(coreData);
        
        if (!result)
            return; // Core logic isn't ready or has no data

        // 4. Unpack results from core
        nav_msgs::Odometry fusedOdometry = result->first;
        Eigen::Affine3f fusedAffine = result->second;

        // 5. Publish fused odometry (ROS-specific)
        pubImuOdometry_.publish(fusedOdometry);

        // 6. Publish odom->baselink TF (ROS-specific)
        tf::Transform tCur;
        tf::poseMsgToTF(fusedOdometry.pose.pose, tCur);
        if(lidarFrame != baselinkFrame)
            tCur = tCur * lidar2Baselink_;
        tf::StampedTransform odom_2_baselink(tCur, odomMsg->header.stamp, odometryFrame, baselinkFrame);
        tfOdom2BaseLink_.sendTransform(odom_2_baselink);

        // 7. Publish IMU Path (ROS-specific)
        double imuTime = fusedOdometry.header.stamp.toSec();
        if (imuTime - last_path_time_ > 0.1)
        {
            last_path_time_ = imuTime;
            geometry_msgs::PoseStamped pose_stamped;
            pose_stamped.header = fusedOdometry.header;
            pose_stamped.pose = fusedOdometry.pose.pose;
            imuPath_.poses.push_back(pose_stamped);
            
            // Prune path based on the cached lidar time
            while(!imuPath_.poses.empty() && imuPath_.poses.front().header.stamp.toSec() < lidarOdomTime_cache_ - 1.0)
                imuPath_.poses.erase(imuPath_.poses.begin());

            if (pubImuPath_.getNumSubscribers() != 0)
            {
                imuPath_.header = fusedOdometry.header;
                pubImuPath_.publish(imuPath_);
            }
        }
    }
};


// =======================================================================
//   IMUPreintegration :: Core Logic
// =======================================================================

class IMUPreintegrationCore
{
public:
    // --- State Variables ---
    bool systemInitialized = false;
    bool doneFirstOpt = false;
    int key = 1;

    // --- GTSAM Objects ---
    gtsam::ISAM2 optimizer;
    gtsam::NonlinearFactorGraph graphFactors;
    gtsam::Values graphValues;

    std::unique_ptr<gtsam::PreintegratedImuMeasurements> imuIntegratorOpt_;
    std::unique_ptr<gtsam::PreintegratedImuMeasurements> imuIntegratorImu_;

    // --- Noise Models ---
    gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise;
    gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise;
    gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise;
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise;
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise2;
    gtsam::Vector noiseModelBetweenBias;

    // --- Data Queues (ROS-Agnostic) ---
    std::deque<CoreImuData> imuQueOpt;
    std::deque<CoreImuData> imuQueImu;

    // --- State Estimates ---
    gtsam::Pose3 prevPose_;
    gtsam::Vector3 prevVel_;
    gtsam::NavState prevState_;
    gtsam::imuBias::ConstantBias prevBias_;

    gtsam::NavState prevStateOdom;
    gtsam::imuBias::ConstantBias prevBiasOdom;

    // --- Timestamps ---
    double lastImuT_imu = -1;
    double lastImuT_opt = -1;
    const double delta_t = 0; // From original code
    double imuRate_;

    // --- Transforms ---
    gtsam::Pose3 imu2Lidar;
    gtsam::Pose3 lidar2Imu;

public:
    IMUPreintegrationCore(double imuGravity, double imuAccNoise, double imuGyrNoise,
                          double imuAccBiasN, double imuGyrBiasN, double imuRate,
                          const Eigen::Vector3d& extTrans)
        : imuRate_(imuRate)
    {
        // Setup IMU-Lidar transforms
        imu2Lidar = gtsam::Pose3(gtsam::Rot3::RzRyRx(extRot.z(), extRot.y(), extRot.x()), 
                                gtsam::Point3(extTrans.x(), extTrans.y(), extTrans.z()));
        lidar2Imu = imu2Lidar.inverse();

        // Setup IMU preintegration parameters
        boost::shared_ptr<gtsam::PreintegrationParams> p = gtsam::PreintegrationParams::MakeSharedU(imuGravity);
        p->accelerometerCovariance  = gtsam::Matrix33::Identity(3,3) * pow(imuAccNoise, 2);
        p->gyroscopeCovariance      = gtsam::Matrix33::Identity(3,3) * pow(imuGyrNoise, 2);
        p->integrationCovariance    = gtsam::Matrix33::Identity(3,3) * pow(1e-4, 2);
        gtsam::imuBias::ConstantBias prior_imu_bias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished());

        // Setup noise models
        priorPoseNoise  = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished());
        priorVelNoise   = gtsam::noiseModel::Isotropic::Sigma(3, 1e4);
        priorBiasNoise  = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3);
        correctionNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished());
        correctionNoise2 = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1, 1, 1, 1, 1, 1).finished());
        noiseModelBetweenBias = (gtsam::Vector(6) << imuAccBiasN, imuAccBiasN, imuAccBiasN, imuGyrBiasN, imuGyrBiasN, imuGyrBiasN).finished();
        
        // Use std::make_unique to avoid raw 'new'
        imuIntegratorImu_ = std::make_unique<gtsam::PreintegratedImuMeasurements>(p, prior_imu_bias);
        imuIntegratorOpt_ = std::make_unique<gtsam::PreintegratedImuMeasurements>(p, prior_imu_bias);

        // Initialize state
        prevPose_ = gtsam::Pose3(gtsam::Rot3::RzRyRx(0,0,0), gtsam::Point3(0,0,0));
        prevVel_ = gtsam::Vector3(0,0,0);
        prevState_ = gtsam::NavState(prevPose_, prevVel_);
        prevBias_ = gtsam::imuBias::ConstantBias();
        prevStateOdom = prevState_;
        prevBiasOdom = prevBias_;
    }

    void resetOptimization()
    {
        gtsam::ISAM2Params optParameters;
        optParameters.relinearizeThreshold = 0.1;
        optParameters.relinearizeSkip = 1;
        optimizer = gtsam::ISAM2(optParameters);

        graphFactors.resize(0);
        graphValues.clear();
    }

    void resetParams()
    {
        lastImuT_imu = -1;
        doneFirstOpt = false;
        systemInitialized = false;
    }

    /**
     * @brief Detects optimization failure.
     * @return true if failure detected, false otherwise.
     */
    bool failureDetection(const gtsam::Vector3& velCur, const gtsam::imuBias::ConstantBias& biasCur)
    {
        Eigen::Vector3f vel(velCur.x(), velCur.y(), velCur.z());
        if (vel.norm() > 30)
            return true; // Large velocity

        Eigen::Vector3f ba(biasCur.accelerometer().x(), biasCur.accelerometer().y(), biasCur.accelerometer().z());
        Eigen::Vector3f bg(biasCur.gyroscope().x(), biasCur.gyroscope().y(), biasCur.gyroscope().z());
        if (ba.norm() > 1.0 || bg.norm() > 1.0)
            return true; // Large bias

        return false;
    }

    /**
     * @brief Handles odometry correction data. Main optimization loop.
     * @param odomCorrection The ROS-agnostic odometry correction data.
     * @return true if optimization was successful, false if a reset occurred.
     */
    bool processOdometry(const CoreOdomCorrectionData& odomCorrection)
    {
        double currentCorrectionTime = odomCorrection.timestamp;

        if (imuQueOpt.empty())
            return true; // Not a failure, just nothing to do

        gtsam::Pose3 lidarPose = odomCorrection.pose;

        // 0. initialize system
        if (systemInitialized == false)
        {
            resetOptimization();

            // pop old IMU message
            while (!imuQueOpt.empty())
            {
                if (imuQueOpt.front().timestamp < currentCorrectionTime - delta_t)
                {
                    lastImuT_opt = imuQueOpt.front().timestamp;
                    imuQueOpt.pop_front();
                }
                else
                    break;
            }
            // initial pose
            prevPose_ = lidarPose.compose(lidar2Imu);
            graphFactors.add(gtsam::PriorFactor<gtsam::Pose3>(X(0), prevPose_, priorPoseNoise));
            // initial velocity
            prevVel_ = gtsam::Vector3(0, 0, 0);
            graphFactors.add(gtsam::PriorFactor<gtsam::Vector3>(V(0), prevVel_, priorVelNoise));
            // initial bias
            prevBias_ = gtsam::imuBias::ConstantBias();
            graphFactors.add(gtsam::PriorFactor<gtsam::imuBias::ConstantBias>(B(0), prevBias_, priorBiasNoise));
            // add values
            graphValues.insert(X(0), prevPose_);
            graphValues.insert(V(0), prevVel_);
            graphValues.insert(B(0), prevBias_);
            // optimize once
            optimizer.update(graphFactors, graphValues);
            graphFactors.resize(0);
            graphValues.clear();

            imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);
            imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
            
            key = 1;
            systemInitialized = true;
            return true;
        }

        // reset graph for speed
        if (key == 100)
        {
            gtsam::noiseModel::Gaussian::shared_ptr updatedPoseNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(X(key-1)));
            gtsam::noiseModel::Gaussian::shared_ptr updatedVelNoise  = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(V(key-1)));
            gtsam::noiseModel::Gaussian::shared_ptr updatedBiasNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(B(key-1)));
            
            resetOptimization();
            
            graphFactors.add(gtsam::PriorFactor<gtsam::Pose3>(X(0), prevPose_, updatedPoseNoise));
            graphFactors.add(gtsam::PriorFactor<gtsam::Vector3>(V(0), prevVel_, updatedVelNoise));
            graphFactors.add(gtsam::PriorFactor<gtsam::imuBias::ConstantBias>(B(0), prevBias_, updatedBiasNoise));

            graphValues.insert(X(0), prevPose_);
            graphValues.insert(V(0), prevVel_);
            graphValues.insert(B(0), prevBias_);
            
            optimizer.update(graphFactors, graphValues);
            graphFactors.resize(0);
            graphValues.clear();

            key = 1;
        }

        // 1. integrate imu data and optimize
        while (!imuQueOpt.empty())
        {
            CoreImuData* thisImu = &imuQueOpt.front();
            double imuTime = thisImu->timestamp;
            if (imuTime < currentCorrectionTime - delta_t)
            {
                double dt = (lastImuT_opt < 0) ? (1.0 / imuRate_) : (imuTime - lastImuT_opt);
                imuIntegratorOpt_->integrateMeasurement(
                            thisImu->linear_acceleration,
                            thisImu->angular_velocity, dt);
                
                lastImuT_opt = imuTime;
                imuQueOpt.pop_front();
            }
            else
                break;
        }
        // add imu factor to graph
        const gtsam::PreintegratedImuMeasurements& preint_imu = *imuIntegratorOpt_;
        graphFactors.add(gtsam::ImuFactor(X(key - 1), V(key - 1), X(key), V(key), B(key - 1), preint_imu));
        // add imu bias between factor
        graphFactors.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(B(key - 1), B(key), gtsam::imuBias::ConstantBias(),
                            gtsam::noiseModel::Diagonal::Sigmas(sqrt(imuIntegratorOpt_->deltaTij()) * noiseModelBetweenBias)));
        // add pose factor
        gtsam::Pose3 curPose = lidarPose.compose(lidar2Imu);
        graphFactors.add(gtsam::PriorFactor<gtsam::Pose3>(X(key), curPose, odomCorrection.degenerate ? correctionNoise2 : correctionNoise));
        // insert predicted values
        gtsam::NavState propState_ = imuIntegratorOpt_->predict(prevState_, prevBias_);
        graphValues.insert(X(key), propState_.pose());
        graphValues.insert(V(key), propState_.v());
        graphValues.insert(B(key), prevBias_);
        
        // optimize
        optimizer.update(graphFactors, graphValues);
        optimizer.update();
        graphFactors.resize(0);
        graphValues.clear();
        
        // Overwrite the beginning of the preintegration for the next step.
        gtsam::Values result = optimizer.calculateEstimate();
        prevPose_   = result.at<gtsam::Pose3>(X(key));
        prevVel_    = result.at<gtsam::Vector3>(V(key));
        prevState_  = gtsam::NavState(prevPose_, prevVel_);
        prevBias_   = result.at<gtsam::imuBias::ConstantBias>(B(key));
        
        // Reset the optimization preintegration object.
        imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
        
        // check optimization
        if (failureDetection(prevVel_, prevBias_))
        {
            resetParams();
            return false; // Failure!
        }

        // 2. after optimization, re-propagate imu odometry preintegration
        prevStateOdom = prevState_;
        prevBiasOdom  = prevBias_;
        
        double lastImuQT = -1;
        while (!imuQueImu.empty() && imuQueImu.front().timestamp < currentCorrectionTime - delta_t)
        {
            lastImuQT = imuQueImu.front().timestamp;
            imuQueImu.pop_front();
        }
        
        // repropagate
        if (!imuQueImu.empty())
        {
            imuIntegratorImu_->resetIntegrationAndSetBias(prevBiasOdom);
            for (size_t i = 0; i < imuQueImu.size(); ++i)
            {
                CoreImuData* thisImu = &imuQueImu[i];
                double imuTime = thisImu->timestamp;
                double dt = (lastImuQT < 0) ? (1.0 / imuRate_) :(imuTime - lastImuQT);

                imuIntegratorImu_->integrateMeasurement(
                    thisImu->linear_acceleration,
                    thisImu->angular_velocity, dt);
                lastImuQT = imuTime;
            }
        }

        ++key;
        doneFirstOpt = true;
        return true; // Success
    }

    /**
     * @brief Adds a new IMU measurement and propagates the IMU-only odometry.
     * @param imuMsg The ROS-agnostic IMU data.
     * @return An optional CoreCalculatedOdometry. If it contains a value, it should be published.
     */
    std::optional<CoreCalculatedOdometry> addImuMeasurement(const CoreImuData& imuMsg, const ros::Time& imuStamp)
    {
        imuQueOpt.push_back(imuMsg);
        imuQueImu.push_back(imuMsg);

        if (doneFirstOpt == false)
            return std::nullopt; // Nothing to publish yet

        double imuTime = imuMsg.timestamp;
        double dt = (lastImuT_imu < 0) ? (1.0 / imuRate_) : (imuTime - lastImuT_imu);
        lastImuT_imu = imuTime;

        // integrate this single imu message
        imuIntegratorImu_->integrateMeasurement(
            imuMsg.linear_acceleration,
            imuMsg.angular_velocity, dt);

        // predict odometry
        gtsam::NavState currentState = imuIntegratorImu_->predict(prevStateOdom, prevBiasOdom);

        // transform imu pose to lidar
        gtsam::Pose3 imuPose = gtsam::Pose3(currentState.quaternion(), currentState.position());
        gtsam::Pose3 lidarPose = imuPose.compose(imu2Lidar);

        // Create the odometry data structure to return
        CoreCalculatedOdometry odom;
        odom.timestamp = imuTime;
        odom.originalStamp = imuStamp; // Pass through the ROS timestamp
        odom.lidarPose = lidarPose;
        odom.velocity = currentState.velocity();
        odom.angular_velocity_biased = gtsam::Vector3(
            imuMsg.angular_velocity.x() + prevBiasOdom.gyroscope().x(),
            imuMsg.angular_velocity.y() + prevBiasOdom.gyroscope().y(),
            imuMsg.angular_velocity.z() + prevBiasOdom.gyroscope().z()
        );

        return odom;
    }
};


// =======================================================================
//   IMUPreintegration :: ROS Wrapper
// =======================================================================

class IMUPreintegrationROS : public ParamServer
{
public:
    std::mutex mtx_;

    // --- ROS Interface ---
    ros::Subscriber subImu_;
    ros::Subscriber subOdometry_;
    ros::Publisher pubImuOdometry_;

    // --- Core Logic ---
    std::unique_ptr<IMUPreintegrationCore> core_;

    IMUPreintegrationROS()
    {
        // 1. Initialize the core logic object
        core_ = std::make_unique<IMUPreintegrationCore>(
            imuGravity, imuAccNoise, imuGyrNoise,
            imuAccBiasN, imuGyrBiasN, imuRate,
            extTrans // extTrans and extRot are from ParamServer
        );

        // 2. Initialize ROS subscribers and publishers
        subImu_        = nh.subscribe<sensor_msgs::Imu>(imuTopic, 2000, 
                            &IMUPreintegrationROS::imuHandler, this, 
                            ros::TransportHints().tcpNoDelay());
        
        subOdometry_   = nh.subscribe<nav_msgs::Odometry>("liorf/mapping/odometry_incremental", 5,   
                            &IMUPreintegrationROS::odometryHandler, this, 
                            ros::TransportHints().tcpNoDelay());

        pubImuOdometry_ = nh.advertise<nav_msgs::Odometry>(odomTopic + "_incremental", 2000);
    }

    // --- ROS Callback Handlers ---

    void odometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg)
    {
        std::lock_guard<std::mutex> lock(mtx_);
        
        // 1. Convert ROS message to core data type
        CoreOdomCorrectionData coreOdom;
        coreOdom.timestamp = odomMsg->header.stamp.toSec();
        coreOdom.degenerate = (int)odomMsg->pose.covariance[0] == 1;
        coreOdom.pose = gtsam::Pose3(
            gtsam::Rot3::Quaternion(
                odomMsg->pose.pose.orientation.w,
                odomMsg->pose.pose.orientation.x,
                odomMsg->pose.pose.orientation.y,
                odomMsg->pose.pose.orientation.z),
            gtsam::Point3(
                odomMsg->pose.pose.position.x,
                odomMsg->pose.pose.position.y,
                odomMsg->pose.pose.position.z)
        );

        // 2. Call the core logic function
        bool success = core_->processOdometry(coreOdom);

        // 3. Handle results (logging)
        if (!success)
        {
            ROS_WARN("IMU preintegration reset due to large velocity or bias.");
        }
    }

    void imuHandler(const sensor_msgs::Imu::ConstPtr& imu_raw)
    {
        std::lock_guard<std::mutex> lock(mtx_);

        // 1. Convert ROS message to core data type
        sensor_msgs::Imu thisImu = imuConverter(*imu_raw); // From utility.h/ParamServer
        CoreImuData coreImu;
        coreImu.timestamp = thisImu.header.stamp.toSec();
        coreImu.linear_acceleration = gtsam::Vector3(
            thisImu.linear_acceleration.x,
            thisImu.linear_acceleration.y,
            thisImu.linear_acceleration.z);
        coreImu.angular_velocity = gtsam::Vector3(
            thisImu.angular_velocity.x,
            thisImu.angular_velocity.y,
            thisImu.angular_velocity.z);
        coreImu.originalImuMsg = thisImu; // Save for twist info

        // 2. Call the core logic function
        std::optional<CoreCalculatedOdometry> result = core_->addImuMeasurement(coreImu, thisImu.header.stamp);

        // 3. Handle results (publishing)
        if (result)
        {
            nav_msgs::Odometry odometry;
            odometry.header.stamp = result->originalStamp;
            odometry.header.frame_id = odometryFrame;
            odometry.child_frame_id = "odom_imu";

            const auto& lidarPose = result->lidarPose;
            odometry.pose.pose.position.x = lidarPose.translation().x();
            odometry.pose.pose.position.y = lidarPose.translation().y();
            odometry.pose.pose.position.z = lidarPose.translation().z();
            odometry.pose.pose.orientation.x = lidarPose.rotation().toQuaternion().x();
            odometry.pose.pose.orientation.y = lidarPose.rotation().toQuaternion().y();
            odometry.pose.pose.orientation.z = lidarPose.rotation().toQuaternion().z();
            odometry.pose.pose.orientation.w = lidarPose.rotation().toQuaternion().w();
            
            const auto& velocity = result->velocity;
            odometry.twist.twist.linear.x = velocity.x();
            odometry.twist.twist.linear.y = velocity.y();
            odometry.twist.twist.linear.z = velocity.z();

            const auto& angVel = result->angular_velocity_biased;
            odometry.twist.twist.angular.x = angVel.x();
            odometry.twist.twist.angular.y = angVel.y();
            odometry.twist.twist.angular.z = angVel.z();
            
            pubImuOdometry_.publish(odometry);
        }
    }
};


// =======================================================================
//   Main Function
// =======================================================================

int main(int argc, char** argv)
{
    ros::init(argc, argv, "roboat_loam");
    
    // Instantiate the ROS wrappers.
    // They will automatically read params and set up their core logic.
    IMUPreintegrationROS ImuP;
    TransformFusionROS TF;

    ROS_INFO("\033[1;32m----> IMU Preintegration Started.\033[0m");
    
    ros::MultiThreadedSpinner spinner(4);
    spinner.spin();
    
    return 0;
}