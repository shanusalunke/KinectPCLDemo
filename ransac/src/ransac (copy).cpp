///home/krang/Desktop/Test/KinectPCLDemo/ransac/src
#include <typeinfo>

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
#include <fstream>
#include <pcl/io/png_io.h>
#include <string>
#include <sstream> 

#include <pcl/io/pcd_io.h>
#include <pcl/point_types_conversion.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/intensity_gradient.h>
#include <pcl/features/rift.h>
 
typedef pcl::Histogram<32> RIFT32;
using namespace std;

int folder_number = 0;

class SimpleOpenNIViewer
{
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr finalcloud;// = new pcl::PointCloud<pcl::PointXYZRGB>;
	// pcl::PointCloud<pcl::PointXYZRGB>::Ptr finalcloud (new pcl::PointCloud<pcl::PointXYZRGB>);

public:
	SimpleOpenNIViewer () : viewer ("PCL OpenNI Viewer") {}

	void cloud_cb_ (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr &ipcloud){

		bool shouldRansac = true;
		bool shouldSave = false;
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBA>(*ipcloud));
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr finalcloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr tempcloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr hullcloud (new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::IntensityGradient>::Ptr gradients(new pcl::PointCloud<pcl::IntensityGradient>);
		pcl::PointCloud<RIFT32>::Ptr descriptors(new pcl::PointCloud<RIFT32>());
		pcl::PointCloud<pcl::PointXYZI>::Ptr cloudIntensity(new pcl::PointCloud<pcl::PointXYZI>);
		pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgbcloud (new pcl::PointCloud<pcl::PointXYZRGB>);
		
		string foldername = "chair1/";
		if (shouldSave){
			::folder_number++;
			std::ostringstream ostr; //output string stream
			ostr << foldername;
			string str = ostr.str();
			cout << "------------------------------------------------------" << str <<"\n";
			mkdir(str.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
		}

		if(shouldRansac){

			//
			// SEGMENTATION
			//
			pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
			pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
			pcl::SACSegmentation<pcl::PointXYZRGBA> seg;
			seg.setOptimizeCoefficients (true); //Optional
			seg.setModelType (pcl::SACMODEL_PLANE); //Mandatory
			seg.setMethodType (pcl::SAC_RANSAC); //Mandatory
			seg.setDistanceThreshold (0.16); //Mandatory
			seg.setMaxIterations (100); //Mandatory


			// Create the filtering object
			pcl::ExtractIndices<pcl::PointXYZRGBA> extract;
			// Extract the inliers in loop later
			// extract.setInputCloud (cloud);
			// extract.setIndices (inliers);
			// extract.setNegative (false);
			// extract.filter (*finalcloud);

			int i = 0, nr_points = (int) cloud->points.size ();
			bool shouldSegmentMore = true;
			pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> single_color1 (finalcloud, 0, 255, 0);
				// While 30% of the original cloud is still there
				while (shouldSegmentMore)
				{
					// std::cout << "Number of points: " << cloud->points.size()<<"\n";
					std::cout << "================= SEGMENT " << i+1 <<" =================\n";
					//std::cout << "Because " << cloud->points.size()<< " > " << (0.3 * nr_points) <<"\n";

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
						cloud.swap (tempcloud);

						//std::cout << "Number of points: \n cloud " << cloud->points.size() << "|" << nr_points << "\n final: "<< finalcloud->points.size() << "\n temp: "<< tempcloud->points.size() << " inliers: " << inliers->indices.size () <<"\n";
						std::cout << "Normals: " << i << ":" << coefficients->values[0] << " " << coefficients->values[1] << " "<< coefficients->values[2] << " "<< coefficients->values[3] << std::endl;

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
						std::cout << "Hullcloud recinstruct: " << cHull_points.points.size() <<"\n";
						cout << "Area: " << cHull.getTotalArea() <<"\n";
						cout << "Volume: " << cHull.getTotalVolume() <<"\n";

						//TEXTURE: Eignfaces
						Eigen::Vector4f xyz_centroid;
						Eigen::Matrix3f covariance_matrix;
						pcl::compute3DCentroid(*hullcloud, xyz_centroid);
						pcl::computeCovarianceMatrix (*hullcloud, xyz_centroid, covariance_matrix);
						cout << "Covariance Matrix: " << covariance_matrix << "\n";
						
						
						//
						//RIFT
						//
						// Convert the RGB to intensity.
						pcl::copyPointCloud(*finalcloud, *rgbcloud);
						//Remove NAN
						std::vector<int> normalindices;
						pcl::removeNaNFromPointCloud(*rgbcloud, *rgbcloud, hullindices);
						pcl::PointCloudXYZRGBtoXYZI(*rgbcloud, *cloudIntensity);
						
						// Estimate the normals.
						pcl::NormalEstimation<pcl::PointXYZI, pcl::Normal> normalEstimation;
						normalEstimation.setInputCloud(cloudIntensity);
						normalEstimation.setRadiusSearch(0.03);
						pcl::search::KdTree<pcl::PointXYZI>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZI>);
						normalEstimation.setSearchMethod(kdtree);
						normalEstimation.compute(*normals);
						
						// Compute the intensity gradients.
						pcl::IntensityGradientEstimation < pcl::PointXYZI, pcl::Normal, pcl::IntensityGradient,
						pcl::common::IntensityFieldAccessor<pcl::PointXYZI> > ge;
						ge.setInputCloud(cloudIntensity);
						ge.setInputNormals(normals);
						ge.setRadiusSearch(0.03);
						ge.compute(*gradients);
						
						// RIFT estimation object.
						pcl::RIFTEstimation<pcl::PointXYZI, pcl::IntensityGradient, RIFT32> rift;
						rift.setInputCloud(cloudIntensity);
						rift.setSearchMethod(kdtree);
						// Set the intensity gradients to use.
						rift.setInputGradient(gradients);
						// Radius, to get all neighbors within.
						rift.setRadiusSearch(0.02);
						// Set the number of bins to use in the distance dimension.
						rift.setNrDistanceBins(4);
						// Set the number of bins to use in the gradient orientation dimension.
						rift.setNrGradientBins(8);
						// Note: you must change the output histogram size to reflect the previous values.
						rift.compute(*descriptors);
						

						viewer.showCloud (finalcloud);
						
						cout <<"i="<<i<<"\n";
						//std::cout << typeid(gradients).name() << '\n';
						cout << "Normals2=" << normals->at(i).normal_x<< normals->at(i).normal_y<< normals->at(i).normal_z <<"\n";
						cout <<"\nGriadients" << gradients->gradient[0]<< gradients->gradient[1];
						//cout << "\nDescriptors" << descriptors;
						
						if(shouldSave){
						
							std::ostringstream ostr1; //output string stream
							ostr1 << foldername << ::folder_number << i << ".png";
							string imgname = ostr1.str();
						
							std::ostringstream ostr2; //output string stream
							ostr2 << foldername << ::folder_number << i << ".json";
							string filename = ostr2.str();

							pcl::io::savePNGFile(imgname, *finalcloud, "rgb");
							ofstream myfile;
							myfile.open(filename.c_str());
							myfile << "{";
							//Normals
							myfile << "\"normals\":{ \"x\":"<<coefficients->values[0] <<",\"y\":"<< coefficients->values[1] <<",\"z\":"<<coefficients->values[2]<<",\"d\":"<<coefficients->values[3] << "},";
							//Surface Area
							myfile << "\"area\":"<< cHull.getTotalArea() << ",";
							//Volume
							myfile << "\"volume\":"<< cHull.getTotalVolume() << ",";
							//Covariance
							myfile << "\"covariance\":"<< covariance_matrix << ",";
						
							myfile << "}";
							myfile.close();
						}
					}

					i++;
					if (cloud->points.size () < 0.3 * nr_points || inliers->indices.size () == 0){
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
