///home/krang/Desktop/Test/KinectPCLDemo/ransac/src
#include <typeinfo>
#include <termios.h>
#include <stdio.h>
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
#include <pcl/features/pfh.h>
#include <pcl/features/fpfh.h>
#include <pcl/filters/voxel_grid.h>

typedef pcl::Histogram<32> RIFT32;
using namespace std;

int folder_number = 0;
int save_name = 0;
char ch = 's';

class SimpleOpenNIViewer
{
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr finalcloud;// = new pcl::PointCloud<pcl::PointXYZRGB>;
	// pcl::PointCloud<pcl::PointXYZRGB>::Ptr finalcloud (new pcl::PointCloud<pcl::PointXYZRGB>);

public:
	SimpleOpenNIViewer () : viewer ("PCL OpenNI Viewer") {}
	void cloud_cb_ (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr &ipcloud){

		// ::ch = getchar();

		bool shouldRansac = true;
		bool shouldSave = true;
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBA>(*ipcloud));
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr finalcloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr tempcloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr hullcloud (new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr smallcloud (new pcl::PointCloud<pcl::PointXYZ>);
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


            //Point Feature Histogram
						//Remove NAN
						// std::vector<int> normalindices;
						// cout << hullcloud->points.size() <<"\t";
						// pcl::removeNaNFromPointCloud(*hullcloud, *hullcloud, hullindices);
						// cout << hullcloud->points.size() <<"\n";

						//Downsample
						// Create the filtering object
						pcl::VoxelGrid<pcl::PointXYZ> sor;
						sor.setInputCloud (hullcloud);
						sor.setLeafSize (0.01f, 0.01f, 0.01f);
						sor.filter (*smallcloud);

						// Estimate the normals.
						pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
						normalEstimation.setInputCloud(smallcloud);
						normalEstimation.setRadiusSearch(0.03);
						pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
						normalEstimation.setSearchMethod(kdtree);
						normalEstimation.compute(*normals);
						cout << normals->points[0] <<"\n";

						cout << "\nDONE NORMALS\n";

						// Compute the intensity gradients.
						pcl::IntensityGradientEstimation < pcl::PointXYZI, pcl::Normal, pcl::IntensityGradient,
						pcl::common::IntensityFieldAccessor<pcl::PointXYZI> > ge;
						ge.setInputCloud(smallcloud);
						ge.setInputNormals(normals);
						ge.setRadiusSearch(0.03);
						ge.compute(*gradients);

            // Create the PFH estimation class, and pass the input dataset+normals to it
            // pcl::PFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::PFHSignature125> pfh;
            // pfh.setInputCloud (smallcloud);
            // pfh.setInputNormals (normals);
            // pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
            // pfh.setSearchMethod (tree);
            // pcl::PointCloud<pcl::PFHSignature125>::Ptr pfhs (new pcl::PointCloud<pcl::PFHSignature125> ());
            // pfh.setRadiusSearch (0.5);
            // pfh.compute (*pfhs);
            // cout << pfhs->points[0] <<"\n";


						// Create the FPFH estimation class, and pass the input dataset+normals to it
						pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
						fpfh.setInputCloud (smallcloud);
						fpfh.setInputNormals (normals);
						pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
						fpfh.setSearchMethod (tree);
						pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs (new pcl::PointCloud<pcl::FPFHSignature33> ());
						fpfh.setRadiusSearch (0.05);
						fpfh.compute (*fpfhs);
						cout << fpfhs->points[0] <<"\n";

						viewer.showCloud (finalcloud);

						cout <<"i="<<i<<"\n";

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

							myfile << "\"normals\": \"("<<coefficients->values[0] <<","<< coefficients->values[1] <<","<<coefficients->values[2]<<","<<coefficients->values[3] << ")\",";
							//Surface Area
							myfile << "\"area\":"<< cHull.getTotalArea() << ",";
							//Volume
							myfile << "\"volume\":"<< cHull.getTotalVolume() << ",";
							//Covariance
							// myfile << "\"covariance\":"<< covariance_matrix << "\",";
							myfile << "\"gradient\":\""<<gradients->at(0).gradient_x <<","<< gradients->at(0).gradient_y <<","<<gradients->at(0).gradient_z<<"\",";
							myfile << "\"normals2\":\""<< normals->points[0] << "\",";
							myfile << "\"fpfhs\":\""<< fpfhs->points[0] << "\"}";
							// myfile << "}";
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
		interface->start();

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
