///home/krang/Desktop/Test/KinectPCLDemo/ransac/src

#include <pcl/io/openni_grabber.h>
#include <pcl/visualization/cloud_viewer.h>
#include <iostream>
#include <pcl/console/parse.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/integral_image_normal.h>
#include <boost/thread/thread.hpp>
#include <pcl/console/parse.h>
// #include <pcl/normal_3d.h>
#include <limits>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/organized_multi_plane_segmentation.h>

class SimpleOpenNIViewer
{
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr finalcloud;// = new pcl::PointCloud<pcl::PointXYZRGB>;
	// pcl::PointCloud<pcl::PointXYZRGB>::Ptr finalcloud (new pcl::PointCloud<pcl::PointXYZRGB>);

public:
	SimpleOpenNIViewer () : viewer ("PCL OpenNI Viewer") {}



	template <typename PointT> void customCopyPointCloud (const pcl::PointCloud<PointT> &cloud_in, const std::vector<int> &indices, pcl::PointCloud<PointT> &cloud_out){
	  // Do we want to copy everything?
	  if (indices.size () == cloud_in.points.size ())
	  {
	  cloud_out = cloud_in;
	  return;
	  }
	  // Allocate enough space and copy the basics
	  cloud_out.points.resize (indices.size ());
	  cloud_out.header = cloud_in.header;
	  // cloud_out.width = static_cast<uint32_t>(indices.size ());
	  // cloud_out.height = 1;
		cloud_out.width = cloud_in.width;
		cloud_out.height = cloud_in.height;

	  cloud_out.is_dense = cloud_in.is_dense;
	  cloud_out.sensor_orientation_ = cloud_in.sensor_orientation_;
	  cloud_out.sensor_origin_ = cloud_in.sensor_origin_;
	  // Iterate over each point
	  for (size_t i = 0; i < indices.size (); ++i)
	  cloud_out.points[i] = cloud_in.points[indices[i]];

		//remove NAN points from the cloud
		// std::vector<int> m;
		// pcl::removeNaNFromPointCloud(cloud_out,cloud_out, m);
	 }



	void cloud_cb_ (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr &cloud){

		bool shouldRansac = true;
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr finalcloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
		pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
		// pcl::PointCloud<pcl::PointXYZ>::Ptr tempCloud (new pcl::PointCloud<pcl::PointXYZ>);

		if(shouldRansac){

			// std::vector<int> inliers;
			// // created RandomSampleConsensus object and compute the appropriated model
			// pcl::SampleConsensusModelPlane<pcl::PointXYZRGBA>::Ptr model_p (new pcl::SampleConsensusModelPlane<pcl::PointXYZRGBA> (cloud));
			// //Apply
			// pcl::RandomSampleConsensus<pcl::PointXYZRGBA> ransac (model_p);
			// ransac.setDistanceThreshold (.2);
			// ransac.computeModel();
			// ransac.getInliers(inliers);
			//



			// pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
			// pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
			// // Create the segmentation object
			// pcl::SACSegmentation<pcl::PointXYZRGBA> seg;
			// // Optional
			// seg.setOptimizeCoefficients (true);
			// // Mandatory
			// seg.setModelType (pcl::SACMODEL_PLANE);
			// seg.setMethodType (pcl::SAC_RANSAC);
			// seg.setDistanceThreshold (0.16);
			// seg.setMaxIterations (100);
			//
			// seg.setInputCloud (cloud);
			// seg.segment (*inliers, *coefficients);
			//
			//
			// // Create the filtering object
			// pcl::ExtractIndices<pcl::PointXYZRGBA> extract;
			// // Extract the inliers
			// extract.setInputCloud (cloud);
			// extract.setIndices (inliers);
			// extract.setNegative (false);
			// extract.filter (*finalcloud);
			//
			// std::cout << "Model coefficients: " << coefficients->values[0] << " " << coefficients->values[1] << " "<< coefficients->values[2] << " "<< coefficients->values[3] << std::endl;



			// //All surface normals
			// // Object for normal estimation.
			// pcl::IntegralImageNormalEstimation<pcl::PointXYZRGBA, pcl::Normal> normalEstimation;
			// normalEstimation.setInputCloud(cloud);
			// normalEstimation.setRadiusSearch(0.03);
			// pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZRGBA>);
			// normalEstimation.setSearchMethod(kdtree);
			// normalEstimation.compute(*normals);
			// //ORganised Multi plane segmentation
			// pcl::OrganizedMultiPlaneSegmentation<pcl::PointT, pcl::Normal, pcl::Label> mps;
			// mps.setMinInliers(1000);
			// mps.setAngularThreshold(0.017453*2.0); //2 degrees
			// mps.setDistanceThreshold(0.2);
			// mps.setInputNormals(normals);
			// mps.setInputCloud(cloud);
			// std::vector<pcl::PlanarRegion<PointT>> regions;
			// mps.segmentAndRegine(regions);
			//
			// for (size_t i=0; i<regions.size(); i++){
			// 	Eigen::Vector4f coeff = regions[i].getCoefficients();
			// 	cout << coeff[0] <<" " <<  coeff[1] << " " << coeff[2] << " " << coeff[3];
			// }

		}


		if (!viewer.wasStopped()){
			if(shouldRansac){

				//Show cloud
				viewer.showCloud (finalcloud);
			}
			else{
				viewer.showCloud (cloud);
			}
		}

	}

	void run (){
		pcl::Grabber* interface = new pcl::OpenNIGrabber();
		boost::function<void (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr&)> f = boost::bind (&SimpleOpenNIViewer::cloud_cb_, this, _1);

		interface->registerCallback (f);
		interface->start ();

		while (!viewer.wasStopped()){
			boost::this_thread::sleep (boost::posix_time::seconds (1));
		}

		interface->stop ();
}

pcl::visualization::CloudViewer viewer;
};

int main ()
{
	SimpleOpenNIViewer v;
	v.run ();
	return 0;
}
