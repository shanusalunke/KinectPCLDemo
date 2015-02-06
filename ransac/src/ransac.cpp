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



class SimpleOpenNIViewer
{
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr finalcloud;// = new pcl::PointCloud<pcl::PointXYZRGB>;
	// pcl::PointCloud<pcl::PointXYZRGB>::Ptr finalcloud (new pcl::PointCloud<pcl::PointXYZRGB>);

public:
	SimpleOpenNIViewer () : viewer ("PCL OpenNI Viewer") {}

	void cloud_cb_ (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr &cloud){

		bool shouldRansac = true;
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr finalcloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
		pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr tempCloud (new pcl::PointCloud<pcl::PointXYZ>);

		if(shouldRansac){

			std::vector<int> inliers;

			// created RandomSampleConsensus object and compute the appropriated model
			pcl::SampleConsensusModelPlane<pcl::PointXYZRGBA>::Ptr model_p (new pcl::SampleConsensusModelPlane<pcl::PointXYZRGBA> (cloud));


			//Apply
			pcl::RandomSampleConsensus<pcl::PointXYZRGBA> ransac (model_p);
			ransac.setDistanceThreshold (.1);
			ransac.computeModel();
			ransac.getInliers(inliers);

			//Copies all inliers of the model computed to another PointCloud
			pcl::copyPointCloud<pcl::PointXYZRGBA>(*cloud, inliers, *finalcloud);

			// estimate normal
			//get temp cloud
			pcl::copyPointCloud<pcl::PointXYZ>(*tempCloud,*finalcloud);
			//
      pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
      ne.setNormalEstimationMethod (ne.AVERAGE_3D_GRADIENT);
      ne.setMaxDepthChangeFactor(0.02f);
      ne.setNormalSmoothingSize(10.0f);
      ne.setInputCloud(tempCloud);
      ne.compute(*normals);

		}

		cout << "S";

		if (!viewer.wasStopped()){
			if(shouldRansac){

				//Show cloud
				viewer.showCloud (finalcloud);

				//Print normals
				cout << normals;

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
