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
#include <pcl/common/projection_matrix.h>

#include <pcl/surface/convex_hull.h>

#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/png_io.h>

// #include <direct.h>
#include <json/json.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>


int folder_number = 1;

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


	void cloud_cb_ (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr &ipcloud){


		//
		// SAVE (create directory)
		//
		//std::string str("");
		char* str;
		sprintf(str, "%d", ::folder_number);
		mkdir(str,0);



		bool shouldRansac = true;
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBA>(*ipcloud));
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr finalcloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
		pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr tempcloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr hullcloud (new pcl::PointCloud<pcl::PointXYZ>);

		// pcl::copyPointCloud(*cloud, *ipcloud);

		if(shouldRansac){




			//
			// SEGMENTATION
			//
			pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
			pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
			// Create the segmentation object
			pcl::SACSegmentation<pcl::PointXYZRGBA> seg;
			// Optional
			seg.setOptimizeCoefficients (true);
			// Mandatory
			seg.setModelType (pcl::SACMODEL_PLANE);
			seg.setMethodType (pcl::SAC_RANSAC);
			seg.setDistanceThreshold (0.16);
			seg.setMaxIterations (100);
			// Do in loop
			// seg.setInputCloud (cloud);
			// seg.segment (*inliers, *coefficients);


			// Create the filtering object
			pcl::ExtractIndices<pcl::PointXYZRGBA> extract;
			// Extract the inliers in loop later
			// extract.setInputCloud (cloud);
			// extract.setIndices (inliers);
			// extract.setNegative (false);
			// extract.filter (*finalcloud);





			int i = 1, nr_points = (int) cloud->points.size ();
			bool shouldSegmentMore = true;
			pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> single_color1 (finalcloud, 0, 255, 0);
				// While 30% of the original cloud is still there
				while (shouldSegmentMore)
				{
					// std::cout << "Number of points: " << cloud->points.size()<<"\n";
					std::cout << "================= SEGMENT " << folder_number <<" =================\n";
					// std::cout << "Because " << cloud->points.size()<< " > " << (0.3 * nr_points) <<"\n";

					// Segment the largest planar component from the remaining cloud
					seg.setInputCloud (cloud);
					seg.segment (*inliers, *coefficients);

					//IF We find a segment
					if(inliers->indices.size () > 0)
					{
						// Extract the inliers
						extract.setInputCloud (cloud);
						extract.setIndices (inliers);
						extract.setNegative (false);
						extract.setKeepOrganized(true);
						extract.filter (*finalcloud);

						// Create the filtering object
						extract.setNegative (true);
						extract.filter (*tempcloud);
						// pcl::copyPointCloud(*cloud, *tempcloud);
						cloud.swap (tempcloud);

						// std::cout << "Number of points: \n cloud " << cloud->points.size() << "|" << nr_points << "\n final: "<< finalcloud->points.size() << "\n temp: "<< tempcloud->points.size() << " inliers: " << inliers->indices.size () <<"\n";
						// std::cout << "Model coefficients " << i << ":" << coefficients->values[0] << " " << coefficients->values[1] << " "<< coefficients->values[2] << " "<< coefficients->values[3] << std::endl;


						// viewer.showCloud (finalcloud);
						// viewer.showCloud (finalcloud);
						viewer.showCloud (finalcloud);


						//CONVEX HULL
						pcl::copyPointCloud(*finalcloud, *hullcloud);
						std::vector<int> hullindices;
						pcl::removeNaNFromPointCloud(*hullcloud, *hullcloud, hullindices);
						std::cout << "Hullcloud: " << hullcloud->points.size() <<"\n";
						pcl::ConvexHull<pcl::PointXYZ> cHull;
						pcl::PointCloud<pcl::PointXYZ> cHull_points;
						cHull.setInputCloud(hullcloud);
						cHull.setComputeAreaVolume(true);
						cHull.reconstruct (cHull_points);
						// std::cout << "Hullcloud recinstruct: " << cHull_points.points.size() <<"\n";
						// cout << "CONVEX HULL: " << cHull.getTotalArea() <<"\n";
						// cout << "CONVEX HULL: " << cHull.getTotalVolume() <<"\n";



						//TEXTURE: Eignfaces
						Eigen::Vector4f xyz_centroid;
						Eigen::Matrix3f covariance_matrix;
						pcl::compute3DCentroid(*hullcloud, xyz_centroid);
						pcl::computeCovarianceMatrix (*hullcloud, xyz_centroid, covariance_matrix);

						// cout << "COVARIANCEMATRIX: " << covariance_matrix << "\n";




						//
						// SAVE
						//
						//std::string name2("");
						//std::string name1("");
						char* name1;
						char* name2;
						sprintf(name1, "%d", i);
						sprintf(name2, "%d", ::folder_number);
						std::string name = std::string(name1) + std::string(name2) + ".png";

						// Save the image (cloud must be organized).
						pcl::io::savePNGFile(name, *finalcloud, "rgb");

						//Json
						json_object* jObj	= json_object_new_object();

						json_object* jNormals						= json_object_new_object();
						json_object* jArea							= json_object_new_object();
						json_object* jVolume						= json_object_new_object();
						json_object* jCovarianceMatrix	= json_object_new_object();


						//Normals
						json_object_object_add(jNormals,"x",json_object_new_int(coefficients->values[0]));
						json_object_object_add(jNormals,"y",json_object_new_int(coefficients->values[1]));
						json_object_object_add(jNormals,"z",json_object_new_int(coefficients->values[2]));
						json_object_object_add(jNormals,"d",json_object_new_int(coefficients->values[3]));


						json_object_object_add(jObj,"normals",jNormals);
						json_object_object_add(jObj,"surfaceArea",json_object_new_int(cHull.getTotalArea()));
						json_object_object_add(jObj,"volume",json_object_new_int(cHull.getTotalVolume()));
						// json_object_object_add(jObj,"covarianceMatrix",jn1);

						//Write to file
						cout << jObj;

					}

					i++;
					if (cloud->points.size () > 0.3 * nr_points || inliers->indices.size () == 0){
						cout << "-------------------------------------------------------------------------------------------------------------------------" <<"\n";
						cout << "Cloud points: " << (cloud->points.size () > 0.3 * nr_points) <<"\n";
						cout << "Inliers indices: " << (inliers->indices.size () == 0) <<"\n";
						shouldSegmentMore = false;
					}

				}


		}//end of if


	}

	void run (){
		pcl::Grabber* interface = new pcl::OpenNIGrabber();
		boost::function<void (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr&)> f = boost::bind (&SimpleOpenNIViewer::cloud_cb_, this, _1);

		interface->registerCallback (f);
		interface->start ();

		while (!viewer.wasStopped()){
			boost::this_thread::sleep (boost::posix_time::seconds (5));
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
