#include <iostream>
#include <pcl/console/parse.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>
#include <pcl/io/openni_grabber.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/console/parse.h>

using namespace std;
using namespace pcl;
 
//PointCloud<PointXYZRGB>::Ptr cloud(new PointCloud<PointXYZRGB>); // A cloud that will store color info.
PointCloud<PointXYZRGB>::Ptr cloud(new PointCloud<PointXYZRGB>);    // A fallback cloud with just depth data.
PointCloud<PointXYZRGB>::Ptr final(new PointCloud<PointXYZRGB>);    // A fallback cloud with just depth data.
//boost::shared_ptr<visualization::CloudViewer> viewer;                 // Point cloud viewer object.
boost::shared_ptr<visualization::PCLVisualizer> viewer;                 // Point cloud viewer object.
Grabber* openniGrabber;                                               // OpenNI grabber that takes data from the device.

// This function is called every time the device has new data.
bool grabberRecieved = false;
void grabberCallback(const PointCloud<PointXYZRGB>::ConstPtr& cloud)
{
	if(! ::grabberRecieved){
		cout << "grabberCallback" << " "<< cloud->points.size() <<"\n";
		::grabberRecieved = true;
	}
	
	// copies all inliers of the model computed to another PointCloud
	copyPointCloud<PointXYZRGB>(*cloud, *::cloud);

	//if (! viewer->wasStopped())
		//viewer->showCloud(cloud);
}


//Returns a viewer object
boost::shared_ptr<visualization::PCLVisualizer> simpleVis (PointCloud<PointXYZRGB>::ConstPtr cloud)
{
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  boost::shared_ptr<visualization::PCLVisualizer> viewer (new visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addPointCloud<PointXYZRGB> (cloud, "sample cloud");
  viewer->setPointCloudRenderingProperties (visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  //viewer->addCoordinateSystem (1.0, "global");
  viewer->initCameraParameters ();
  return (viewer);
}


//RGB Visualizer
boost::shared_ptr<visualization::PCLVisualizer> rgbVis (PointCloud<PointXYZRGB>::ConstPtr cloud)
{
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  boost::shared_ptr<visualization::PCLVisualizer> viewer (new visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  visualization::PointCloudColorHandlerRGBField<PointXYZRGB> rgb(cloud);
  viewer->addPointCloud<PointXYZRGB> (cloud, rgb, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  return (viewer);
}

// Main function
int main(int argc, char** argv)
{
	bool justVisualize(false);
 
	
	// Second mode, start fetching and displaying frames from the device.	
	openniGrabber = new OpenNIGrabber();
	if (openniGrabber == 0)
		return -1;
	boost::function<void (const PointCloud<PointXYZRGB>::ConstPtr&)> f = boost::bind(&grabberCallback, _1);
	openniGrabber->registerCallback(f);
	
 
 	//Start grabber
	openniGrabber->start();
 
 	//Wait to recieve grabbber callback at least once
 	while(! ::grabberRecieved){
 		boost::this_thread::sleep(boost::posix_time::seconds(1));
 	}
 	
 	cout << "Cloud: "<< cloud->points.size() <<"\n";
 	
 	vector<int> inliers;
 	// created RandomSampleConsensus object and compute the appropriated model
	SampleConsensusModelSphere<PointXYZRGB>::Ptr model_s(new SampleConsensusModelSphere<PointXYZRGB> (cloud));
	SampleConsensusModelPlane<PointXYZRGB>::Ptr model_p (new SampleConsensusModelPlane<PointXYZRGB> (cloud));
	if(console::find_argument (argc, argv, "-f") >= 0)
	{
		RandomSampleConsensus<PointXYZRGB> ransac (model_p);
		ransac.setDistanceThreshold (.01);
		ransac.computeModel();
		ransac.getInliers(inliers);
	}
	else if (console::find_argument (argc, argv, "-sf") >= 0 )
	{
		RandomSampleConsensus<PointXYZRGB> ransac (model_s);
		ransac.setDistanceThreshold (.01);
		ransac.computeModel();
		ransac.getInliers(inliers);
	}
	
	// copies all inliers of the model computed to another PointCloud
	copyPointCloud<PointXYZRGB>(*cloud, inliers, *final);  
  	
	// creates the visualization object and adds either our orignial cloud or all of the inliers
	// depending on the command line arguments specified.
	boost::shared_ptr<visualization::PCLVisualizer> viewer;
	if (console::find_argument (argc, argv, "-f") >= 0 || console::find_argument (argc, argv, "-sf") >= 0){
		viewer = rgbVis(final);
		//viewer = simpleVis(final);
	}
	else{
		viewer = rgbVis(cloud);
		//viewer = simpleVis(cloud);
	}
 
	// Main loop.
	while (! viewer->wasStopped()){
		viewer->spinOnce (100);
		boost::this_thread::sleep(boost::posix_time::seconds(1));
	}
 
	if (! justVisualize)
		openniGrabber->stop();
}

