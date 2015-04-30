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


    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBA>(*ipcloud));

    viewer.showCloud (cloud);
    ::ch = getchar();

    string foldername = "pcd/";
    if (true){
      ::folder_number++;
      std::ostringstream ostr; //output string stream
      ostr << foldername;
      string str = ostr.str();
      cout << "------------------------------------------------------" << str <<"\n";
      mkdir(str.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }

    //Save pcd
    std::ostringstream ostr1; //output string stream
    ostr1 << foldername << ::save_name << ".pcd";
    string filename = ostr1.str();
    // pcl::io::savePCDFile (filename, ipcloud);
    pcl::io::savePCDFileASCII (filename, *cloud);
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
