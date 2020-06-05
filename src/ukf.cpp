#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.6;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.8;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
  is_initialized_ = false;
  n_x_ = 5;
  n_aug_ = 7;
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  lambda_ = 3 - n_aug_;
  weights_ = VectorXd(2*n_aug_+1);

  double weight_0 = lambda_/(lambda_+n_aug_);
  weights_(0) = weight_0;
  for (int i=1; i<2*n_aug_+1; i++) {  //2n+1 weights
    double weight = 0.5/(n_aug_+lambda_);
    weights_(i) = weight;
  }
  
   R_radar_ = MatrixXd(3, 3);
   R_radar_ << std_radr_*std_radr_, 0, 0,
               0, std_radphi_*std_radphi_, 0,
               0, 0, std_radrd_*std_radrd_;

   R_lidar_ = MatrixXd(2,2);
   R_lidar_ << std_laspx_*std_laspx_, 0,
               0, std_laspy_*std_laspy_;

  
  P_ << 1,0,0,0,0,
       0,1,0,0,0,
       0,0,1,0,0,
       0,0,0,1,0,
       0,0,0,0,1;
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
  
  if (!is_initialized_) {
  is_initialized_ = true;
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {

    float rho = meas_package.raw_measurements_[0]; 
    float phi = meas_package.raw_measurements_[1];  
    float rho_dot = meas_package.raw_measurements_[2]; 

    float x = rho*cos(phi);
    float y = rho*sin(phi);
    float vx = rho_dot*cos(phi);
    float vy = rho_dot*sin(phi);
    float v = sqrt(vx*vx + vy*vy);

    x_ << x,y,v,0,0;  
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    float x0 = meas_package.raw_measurements_[0]; 
    float y0 = meas_package.raw_measurements_[1]; 
    x_ << x0, y0, 0, 0, 0;


  }
   

  time_us_ = meas_package.timestamp_;
  
  return;
  }
  
  double deltat = (meas_package.timestamp_ - time_us_) / 1000000.0; 
  time_us_ = meas_package.timestamp_;

  Prediction(deltat);

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  } 
  if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(meas_package);
  }
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
   
   VectorXd x_aug = VectorXd(n_aug_);
   MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
   MatrixXd Xsig_aug = MatrixXd(n_aug_, 2*n_aug_+1);
   
   x_aug.head(5) = x_;
   x_aug(5) = 0;
   x_aug(6) = 0;
   // create augmented covariance matrix
   P_aug.fill(0.0);
   P_aug.topLeftCorner(5,5) = P_;
   P_aug(5,5) = std_a_*std_a_;
   P_aug(6,6) = std_yawdd_*std_yawdd_;

   // create square root matrix
   MatrixXd L = P_aug.llt().matrixL();
   // create augmented sigma points
   Xsig_aug.col(0) = x_aug;
   for(int i = 0; i < n_aug_; i++){
     Xsig_aug.col(i+1) = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
     Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
   }

   // sigma point prediction

   for(int i = 0; i<2*n_aug_+1; i++){
     double p_x = Xsig_aug(0, i);
     double p_y = Xsig_aug(1, i);
     double v = Xsig_aug(2, i);
     double yaw = Xsig_aug(3, i);
     double yawd = Xsig_aug(4,i);
     double nu_a = Xsig_aug(5,i);
     double nu_yawd = Xsig_aug(6,i);
     // predicted state values
     double px_p, py_p, v_p, yaw_p, yawd_p;

     // avoid division by zero
     if(fabs(yawd) > 0.001){
       px_p = p_x + v/yawd * (sin(yaw + yawd*delta_t) - sin(yaw));
       py_p = p_y + v/yawd * (-cos(yaw + yawd*delta_t) + cos(yaw));
     } else{
       px_p = p_x + v * delta_t * cos(yaw);
       py_p = p_y + v * delta_t * sin(yaw);
     }
     v_p = v;
     yaw_p = yaw + yawd * delta_t;
     yawd_p = yawd;

     // add noise
     px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
     py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
     v_p = v_p + nu_a * delta_t;
     yaw_p = yaw_p + 0.5 * nu_yawd * delta_t * delta_t;
     yawd_p = yawd_p + nu_yawd * delta_t;

     // write predicted sigma point into right column
     Xsig_pred_(0, i) = px_p;
     Xsig_pred_(1, i) = py_p;
     Xsig_pred_(2, i) = v_p;
     Xsig_pred_(3, i) = yaw_p;
     Xsig_pred_(4, i) = yawd_p;
   }

   // Predict state mean
   x_.fill(0.0);
   for(int i = 0; i < 2*n_aug_+1; i++){
     x_ = x_ + weights_(i)*Xsig_pred_.col(i);
   }

   // Predict state covairance
   P_.fill(0.0);
   for(int i = 0; i < 2*n_aug_+1; i++){
     VectorXd x_diff = Xsig_pred_.col(i) - x_;
     // angle normalization
     while(x_diff(3) > M_PI) x_diff(3) -= 2.*M_PI;
     while(x_diff(3) < -M_PI) x_diff(3) += 2.*M_PI;
     P_ = P_ + weights_(i)*x_diff*x_diff.transpose();
   }
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
    // *************** Predict lidar measurement *************************
  
  VectorXd z_ = meas_package.raw_measurements_;

   int n_z_ = 3;
   MatrixXd Zsig = MatrixXd(n_z_, 2*n_aug_+1);

   for(int i = 0; i < 2 * n_aug_ + 1; i++){


     Zsig(0, i) = Xsig_pred_(0, i);                      // x
     Zsig(1, i) = Xsig_pred_(1, i);                      // y
    
   }

   // calculate mean predicted measurement
   VectorXd z_pred_ = VectorXd(n_z_);
   z_pred_.fill(0.0);
   for(int i = 0; i < 2*n_aug_+1; i++){
     z_pred_ = z_pred_ + weights_(i)*Zsig.col(i);
   }
   // calculate covariance of predicted measurement
   MatrixXd S = MatrixXd(n_z_, n_z_);
   S.fill(0.0);
   for(int i = 0; i < 2*n_aug_+1; i++){
     VectorXd z_diff = Zsig.col(i) - z_pred_;

     while(z_diff(1) > M_PI) z_diff(1) -= 2.*M_PI;
     while(z_diff(1) < -M_PI) z_diff(1) += 2.*M_PI;

     S = S + weights_(i) * z_diff * z_diff.transpose();
   }

   // add measurement noise covariance matrix
   S = S + R_lidar_;
     
     // *************** UKF update *************************
  
   MatrixXd Tc = MatrixXd(n_x_, n_z_);
   Tc.fill(0.0);
   for(int i = 0; i < 2*n_aug_+1; i++){
     VectorXd x_diff = Xsig_pred_.col(i) - x_;
     while(x_diff(3) > M_PI) x_diff(3) -= 2.*M_PI;
     while(x_diff(3) < -M_PI) x_diff(3) += 2.*M_PI;

     VectorXd z_diff = Zsig.col(i) - z_pred_;
     while(z_diff(1) > M_PI) z_diff(1) -= 2.*M_PI;
     while(z_diff(1) < -M_PI) z_diff(1) += 2.*M_PI;

     Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
   }

   // calculate Kalman gain K
   MatrixXd K = Tc * S.inverse();

   // update state mean and covariance
   // residual
   VectorXd z_diff = z_ - z_pred_;
   while(z_diff(1) > M_PI) z_diff(1) -= 2.*M_PI;
   while(z_diff(1) < -M_PI) z_diff(1) += 2.*M_PI;

   x_ = x_ + K*z_diff;

   P_ = P_ - K*S*K.transpose();

  
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
    
  // *************** Predict radar measurement *************************
  
  VectorXd z_ = meas_package.raw_measurements_;

   int n_z_ = 3;
   MatrixXd Zsig = MatrixXd(n_z_, 2*n_aug_+1);

   for(int i = 0; i < 2 * n_aug_ + 1; i++){
     double p_x = Xsig_pred_(0, i);
     double p_y = Xsig_pred_(1, i);
     double v = Xsig_pred_(2, i);
     double yaw = Xsig_pred_(3, i);
     double yawd = Xsig_pred_(4, i);

     double vx = cos(yaw)*v;
     double vy = sin(yaw)*v;

     Zsig(0, i) = sqrt(p_x*p_x + p_y*p_y);                      // r
     Zsig(1, i) = atan2(p_y, p_x);                              // phi
     Zsig(2, i) = (p_x*vx + p_y*vy)/(sqrt(p_x*p_x + p_y*p_y));  // r_dot
   }

   // calculate mean predicted measurement
   VectorXd z_pred_ = VectorXd(n_z_);
   z_pred_.fill(0.0);
   for(int i = 0; i < 2*n_aug_+1; i++){
     z_pred_ = z_pred_ + weights_(i)*Zsig.col(i);
   }
   // calculate covariance of predicted measurement
   MatrixXd S = MatrixXd(n_z_, n_z_);
   S.fill(0.0);
   for(int i = 0; i < 2*n_aug_+1; i++){
     VectorXd z_diff = Zsig.col(i) - z_pred_;

     while(z_diff(1) > M_PI) z_diff(1) -= 2.*M_PI;
     while(z_diff(1) < -M_PI) z_diff(1) += 2.*M_PI;

     S = S + weights_(i) * z_diff * z_diff.transpose();
   }

   // add measurement noise covariance matrix
   S = S + R_radar_;
     
     // *************** UKF update *************************
  
     MatrixXd Tc = MatrixXd(n_x_, n_z_);
   Tc.fill(0.0);
   for(int i = 0; i < 2*n_aug_+1; i++){
     VectorXd x_diff = Xsig_pred_.col(i) - x_;
     while(x_diff(3) > M_PI) x_diff(3) -= 2.*M_PI;
     while(x_diff(3) < -M_PI) x_diff(3) += 2.*M_PI;

     VectorXd z_diff = Zsig.col(i) - z_pred_;
     while(z_diff(1) > M_PI) z_diff(1) -= 2.*M_PI;
     while(z_diff(1) < -M_PI) z_diff(1) += 2.*M_PI;

     Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
   }

   // calculate Kalman gain K
   MatrixXd K = Tc * S.inverse();

   // update state mean and covariance
   // residual
   VectorXd z_diff = z_ - z_pred_;
   while(z_diff(1) > M_PI) z_diff(1) -= 2.*M_PI;
   while(z_diff(1) < -M_PI) z_diff(1) += 2.*M_PI;

   x_ = x_ + K*z_diff;

   P_ = P_ - K*S*K.transpose();

}