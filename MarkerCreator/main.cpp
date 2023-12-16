#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

int main(int argc, char** argv) {
	
	//take int and marker format
	int markerId = atoi(argv[1]);
	int markerSize = atoi(argv[2]);
	std::string markerName = argv[3];

	cv::Mat markerImage;
	cv::aruco::Dictionary dictionary;
	switch (markerSize) {
	case 4:
		dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
		break;
	case 5:
		dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_50);
		break;
	case 6:
		dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_50);
		break;
	case 7:
		dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_7X7_50);
		break;
	default:
		std::cout << "Invalid marker size" << std::endl;
		return -1;
	}
	cv::aruco::generateImageMarker(dictionary, markerId, 200, markerImage, 1);
	cv::imwrite(markerName, markerImage);
	return 0;
}