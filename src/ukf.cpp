#include <fstream>
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
  std_a_ = 3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 2.5;
  
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

  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = 7;

  // predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  Xsig_pred_.fill(0.0);

  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // Weights of sigma points
  weights_ = VectorXd(2 * n_aug_ + 1);
  double w0 = lambda_ / (lambda_ + n_aug_);
  double w = 1 / (2 * (lambda_ + n_aug_));
  weights_.fill(w);
  weights_(0) = w0;

  // time when the state is true, in us
  time_us_= 0;

  // Set NIS
  NIS_radar_ = 0.0;
  NIS_laser_ = 0.0;

  R_laser_ = MatrixXd(2, 2);
  R_laser_ << pow(std_laspx_, 2), 0,
            0, pow(std_laspy_, 2);

  H_laser_ = MatrixXd(2, n_x_);
  H_laser_ << 1, 0, 0, 0, 0,
            0, 1, 0, 0, 0;

  R_radar_ = MatrixXd(3, 3);
  R_radar_ << pow(std_radr_, 2), 0, 0,
            0, pow(std_radphi_, 2), 0,
            0, 0, pow(std_radrd_, 2);

  is_initialized_ = false;
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
   if (!is_initialized_)
   {
       if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
       {
           double rho = meas_package.raw_measurements_[0];
           double phi = meas_package.raw_measurements_[1];
           double rhod = meas_package.raw_measurements_[2];

           // calculate position x, y
           double x = rho * cos(phi);
           double y = rho * sin(phi);

           // calculate velocity
           double vx = rhod * cos(phi);
           double vy = rhod * sin(phi);
           double v = sqrt(vx * vx + vy * vy);

           // initialize the state vector
           x_ << x, y, v, rho, rhod;

           // initialize the state covariance matric
           P_ <<   1, 0, 0, 0, 0,
                   0, 1, 0, 0, 0,
                   0, 0, 200, 0, 0,
                   0, 0, 0, 100, 0,
                   0, 0, 0, 0, 1;
       }
       else if (meas_package.sensor_type_ == MeasurementPackage::LASER)
       {
           // initialize the state vector
           x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0.0, 0.0, 0.0;

           // initialize the state covariance matric
           P_ <<   std_laspx_*std_laspx_, 0, 0, 0, 0,
                   0, std_laspy_*std_laspy_, 0, 0, 0,
                   0, 0, 200, 0, 0,
                   0, 0, 0, 100, 0,
                   0, 0, 0, 0, 1;
       }

       time_us_ = meas_package.timestamp_;
       is_initialized_ = true;
   }
   else
   {
       double dt = (meas_package.timestamp_ - time_us_) / 1e6;
       time_us_ = meas_package.timestamp_;

       Prediction(dt);

       if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
           UpdateRadar(meas_package);
       else if (meas_package.sensor_type_ == MeasurementPackage::LASER)
           UpdateLidar(meas_package);
   }
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
    MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);;
    GenerateSigmaPoints(&Xsig_aug);
    Xsig_pred_ = SigmaPointPrediction(Xsig_aug, delta_t);
    PredictMeanAndCovariance(&x_, &P_);
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
    // residual
    VectorXd z = meas_package.raw_measurements_;
    VectorXd z_pred = H_laser_ * x_;
    VectorXd y = z - z_pred;

    MatrixXd Ht = H_laser_.transpose();
    MatrixXd S = H_laser_ * P_ * Ht + R_laser_;
    MatrixXd Sinv = S.inverse();
    MatrixXd PHt = P_ * Ht;
    MatrixXd K = PHt * Sinv;
    MatrixXd I = MatrixXd::Identity(n_x_, n_x_);

    // update state mean and covariance matrix
    x_ = x_ + (K * y);
    P_ = (I - K * H_laser_) * P_;

    NIS_laser_ = y.transpose() * Sinv * y;
    std::ofstream NIS_laser_file_;
    if (!NIS_laser_file_.is_open())
        NIS_laser_file_.open("../NIS/NIS_Lidar.csv", std::fstream::out | std::fstream::app);
    NIS_laser_file_ << NIS_laser_ << std::endl;

}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
//    int n_z = 3;
//    VectorXd z_pred = VectorXd(n_z);
//    MatrixXd S = MatrixXd(n_z,n_z);
//    PredictRadarMeasurement(&z_pred, &S);
    // set measurement dimension, radar can measure r, phi, and r_dot
    int n_z = 3;

    // create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

    // mean predicted measurement
    VectorXd z_pred = VectorXd(n_z);
    z_pred.fill(0.0);

    // measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z,n_z);
    S.fill(0.0);

    // transform sigma points into measurement space
    for (int i = 0; i < 2 * n_aug_ + 1; ++i)
    {
        // 2n+1 simga points
        // extract values for better readability
        double p_x = Xsig_pred_(0,i);
        double p_y = Xsig_pred_(1,i);
        double v  = Xsig_pred_(2,i);
        double yaw = Xsig_pred_(3,i);

        double v1 = cos(yaw)*v;
        double v2 = sin(yaw)*v;

        // measurement model
        Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                       // r
        Zsig(1,i) = atan2(p_y, p_x);                                // phi
        Zsig(2,i) = (p_x*v1 + p_y*v2) / sqrt(p_x*p_x + p_y*p_y);   // r_dot
    }

    // mean predicted measurement
    for (int i=0; i < 2 * n_aug_ + 1; ++i)
    {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }

    // innovation covariance matrix S
    for (int i = 0; i < 2 * n_aug_ + 1; ++i)
    {
        // 2n+1 simga points
        // residual
        VectorXd z_diff = Zsig.col(i) - z_pred;

        // angle normalization
        while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
        while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

        S = S + weights_(i) * z_diff * z_diff.transpose();
    }

    // add measurement noise covariance matrix
    S = S + R_radar_;

    /*Update State*/
    //create example vector for incoming radar measurement
    VectorXd z = meas_package.raw_measurements_;

    // create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd(n_x_, n_z);
    Tc.fill(0.0);

    // calculate cross correlation matrix
    for (int i = 0; i < 2 * n_aug_ + 1; ++i)
    {
        // 2n+1 simga points
        // residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        // angle normalization
        while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
        while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        // angle normalization
        while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
        while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    // Kalman gain K;
    MatrixXd K = Tc * S.inverse();

    // residual
    VectorXd z_diff = z - z_pred;

    // angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    // update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K*S*K.transpose();

    //calculate NIS
    NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
    std::ofstream NIS_radar_file_;
    if (!NIS_radar_file_.is_open())
        NIS_radar_file_.open("../NIS/NIS_Radar.csv", std::fstream::out | std::fstream::app);
    NIS_radar_file_ << NIS_radar_ << std::endl;
}

void UKF::GenerateSigmaPoints(MatrixXd* Xsig_out) {

    // set state
    VectorXd x_aug = VectorXd(n_aug_);
    x_aug.fill(0.0);
    x_aug.head(n_x_) = x_;

    // set covariance matrix
    MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
    P_aug.fill(0.0);
    P_aug.topLeftCorner(n_x_, n_x_) = P_;
    P_aug(n_x_, n_x_) = pow(std_a_, 2);
    P_aug(n_x_ + 1, n_x_ + 1) = pow(std_yawdd_, 2);

    // calculate square root of P
    MatrixXd A_aug = P_aug.llt().matrixL();

    // create sigma point matrix
    MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

    // calculate sigma points ...
    Xsig_aug.col(0) = x_aug;
    MatrixXd term = sqrt(lambda_ + n_aug_) * A_aug;
    for(int i=0; i < n_aug_; ++i)
    {
        Xsig_aug.col(i + 1) = x_aug + term.col(i);
        Xsig_aug.col(i + n_aug_ + 1) = x_aug - term.col(i);
    }

    // print result
    // std::cout << "Xsig = " << std::endl << Xsig << std::endl;

    // write result
    *Xsig_out = Xsig_aug;
}

MatrixXd  UKF::SigmaPointPrediction(MatrixXd& Xsig_aug, double delta_t) {

    // create matrix with predicted sigma points as columns
    MatrixXd Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);

    for (int i = 0; i< 2*n_aug_+1; ++i)
    {
        // extract values for better readability
        double p_x = Xsig_aug(0,i);
        double p_y = Xsig_aug(1,i);
        double v = Xsig_aug(2,i);
        double yaw = Xsig_aug(3,i);
        double yawd = Xsig_aug(4,i);
        double nu_a = Xsig_aug(5,i);
        double nu_yawdd = Xsig_aug(6,i);

        // predicted state values
        double px_p, py_p;

        // avoid division by zero
        if (fabs(yawd) > 0.001) {
            px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
            py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
        } else {
            px_p = p_x + v*delta_t*cos(yaw);
            py_p = p_y + v*delta_t*sin(yaw);
        }

        double v_p = v;
        double yaw_p = yaw + yawd*delta_t;
        double yawd_p = yawd;

        // add noise
        px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
        py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
        v_p = v_p + nu_a*delta_t;

        yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
        yawd_p = yawd_p + nu_yawdd*delta_t;

        // write predicted sigma point into right column
        Xsig_pred(0,i) = px_p;
        Xsig_pred(1,i) = py_p;
        Xsig_pred(2,i) = v_p;
        Xsig_pred(3,i) = yaw_p;
        Xsig_pred(4,i) = yawd_p;
    }

    // return result
    return Xsig_pred;
}

void UKF::PredictMeanAndCovariance(VectorXd* x_out, MatrixXd* P_out) {

    // create vector for predicted state
    VectorXd x = VectorXd(n_x_);
    x.fill(0.0);

    // create covariance matrix for prediction
    MatrixXd P = MatrixXd(n_x_, n_x_);
    P.fill(0.0);

    // predict state mean
    for (int i = 0; i < 2 * n_aug_ + 1; ++i)
    {
        // iterate over sigma points
        x = x + weights_(i) * Xsig_pred_.col(i);
    }

    // predict state covariance matrix
    for (int i = 0; i < 2 * n_aug_ + 1; ++i)
    {
        // iterate over sigma points
        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x;
        // angle normalization
        while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
        while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

        P = P + weights_(i) * x_diff * x_diff.transpose() ;
    }

    // write result
    *x_out = x;
    *P_out = P;
}

void UKF::PredictRadarMeasurement(VectorXd* z_out, MatrixXd* S_out) {

    // set measurement dimension, radar can measure r, phi, and r_dot
    int n_z = 3;

    // create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

    // mean predicted measurement
    VectorXd z_pred = VectorXd(n_z);
    z_pred.fill(0.0);

    // measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z,n_z);
    S.fill(0.0);

    // transform sigma points into measurement space
    for (int i = 0; i < 2 * n_aug_ + 1; ++i)
    {
        // 2n+1 simga points
        // extract values for better readability
        double p_x = Xsig_pred_(0,i);
        double p_y = Xsig_pred_(1,i);
        double v  = Xsig_pred_(2,i);
        double yaw = Xsig_pred_(3,i);

        double v1 = cos(yaw)*v;
        double v2 = sin(yaw)*v;

        // measurement model
        Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                       // r
        Zsig(1,i) = atan2(p_y, p_x);                                // phi
        Zsig(2,i) = (p_x*v1 + p_y*v2) / sqrt(p_x*p_x + p_y*p_y);   // r_dot
    }

    // mean predicted measurement
    for (int i=0; i < 2 * n_aug_ + 1; ++i)
    {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }

    // innovation covariance matrix S
    for (int i = 0; i < 2 * n_aug_ + 1; ++i)
    {
        // 2n+1 simga points
        // residual
        VectorXd z_diff = Zsig.col(i) - z_pred;

        // angle normalization
        while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
        while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

        S = S + weights_(i) * z_diff * z_diff.transpose();
    }

    // add measurement noise covariance matrix
    MatrixXd R = MatrixXd(n_z,n_z);
    R <<  std_radr_*std_radr_, 0, 0,
            0, std_radphi_*std_radphi_, 0,
            0, 0,std_radrd_*std_radrd_;
    S = S + R;

    // write result
    *z_out = z_pred;
    *S_out = S;
}
