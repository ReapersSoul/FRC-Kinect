#include <libfreenect.hpp>
#include <libfreenect.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <pthread.h>
#include <algorithm>
#include <thread>
#include <fstream>

#include <GL/freeglut.h>
#include <GL/gl.h>
#include <GL/glu.h>

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <glm/glm.hpp>

#ifdef GraphicCard
#include <CL/opencl.hpp>
#endif

namespace FRC_Kinect
{
	float _lerp(float a, float b, float t)
	{
		return a + (b - a) * t;
	}

	float _map(float value, float istart, float istop, float ostart, float ostop)
	{
		return ostart + (ostop - ostart) * ((value - istart) / (istop - istart));
	}

	struct boundingBox
	{
		glm::vec3 topLeft;
		glm::vec3 topRight;
		glm::vec3 bottomLeft;
		glm::vec3 bottomRight;
		glm::vec3 center;

		boundingBox(glm::vec3 topLeft, glm::vec3 topRight, glm::vec3 bottomLeft, glm::vec3 bottomRight)
		{
			this->topLeft = topLeft;
			this->topRight = topRight;
			this->bottomLeft = bottomLeft;
			this->bottomRight = bottomRight;
			this->center = glm::vec3((topLeft.x + topRight.x + bottomLeft.x + bottomRight.x) / 4, (topLeft.y + topRight.y + bottomLeft.y + bottomRight.y) / 4, (topLeft.z + topRight.z + bottomLeft.z + bottomRight.z) / 4);
		}

		boundingBox()
		{
			this->topLeft = glm::vec3(0, 0, 0);
			this->topRight = glm::vec3(0, 0, 0);
			this->bottomLeft = glm::vec3(0, 0, 0);
			this->bottomRight = glm::vec3(0, 0, 0);
			this->center = glm::vec3(0, 0, 0);
		}

		static boundingBox fromRect(cv::Rect rect)
		{
			return boundingBox(glm::vec3(rect.x, rect.y, 0), glm::vec3(rect.x + rect.width, rect.y, 0), glm::vec3(rect.x, rect.y + rect.height, 0), glm::vec3(rect.x + rect.width, rect.y + rect.height, 0));
		}

		static boundingBox fromCornerPoints(std::vector<cv::Point2f> points)
		{
			return boundingBox(glm::vec3(points[0].x, points[0].y, 0), glm::vec3(points[1].x, points[1].y, 0), glm::vec3(points[2].x, points[2].y, 0), glm::vec3(points[3].x, points[3].y, 0));
		}
	};

	struct Marker : public boundingBox
	{
		int id;

		Marker(glm::vec3 topLeft, glm::vec3 topRight, glm::vec3 bottomLeft, glm::vec3 bottomRight, int id) : boundingBox(topLeft, topRight, bottomLeft, bottomRight)
		{
			this->id = id;
		}

		Marker() : boundingBox()
		{
			this->id = -1;
		}

		static Marker fromCornerPoints(std::vector<cv::Point2f> points, int id, int width, int height)
		{
			return Marker(glm::vec3(points[0].x / (float)width, points[0].y / (float)height, 0), glm::vec3(points[1].x / (float)width, points[1].y / (float)height, 0), glm::vec3(points[2].x / (float)width, points[2].y / (float)height, 0), glm::vec3(points[3].x / (float)width, points[3].y / (float)height, 0), id);
		}

		static Marker fromRect(cv::Rect rect, int id, int width, int height)
		{
			return Marker(glm::vec3(rect.x / (float)width, rect.y / (float)height, 0), glm::vec3((rect.x + rect.width) / (float)width, rect.y / (float)height, 0), glm::vec3(rect.x / (float)width, (rect.y + rect.height) / (float)height, 0), glm::vec3((rect.x + rect.width) / (float)width, (rect.y + rect.height) / (float)height, 0), id);
		}

		void DetrmineDepth(std::vector<uint16_t> depth, int width, int height)
		{
			// get the corners and center from normalized coordinates
			glm::vec3 topLeft = glm::vec3(_map(this->topLeft.x, 0, 1, 0, width), _map(this->topLeft.y, 0, 1, 0, height), 0);
			glm::vec3 topRight = glm::vec3(_map(this->topRight.x, 0, 1, 0, width), _map(this->topRight.y, 0, 1, 0, height), 0);
			glm::vec3 bottomLeft = glm::vec3(_map(this->bottomLeft.x, 0, 1, 0, width), _map(this->bottomLeft.y, 0, 1, 0, height), 0);
			glm::vec3 bottomRight = glm::vec3(_map(this->bottomRight.x, 0, 1, 0, width), _map(this->bottomRight.y, 0, 1, 0, height), 0);
			glm::vec3 center = glm::vec3((topLeft.x + topRight.x + bottomLeft.x + bottomRight.x) / 4, (topLeft.y + topRight.y + bottomLeft.y + bottomRight.y) / 4, 0);

			// get the depth data
			float depthTopLeft = _map(depth[(int)topLeft.y * width + (int)topLeft.x], 0, 2048, 0, 1);
			float depthTopRight = _map(depth[(int)topRight.y * width + (int)topRight.x], 0, 2048, 0, 1);
			float depthBottomLeft = _map(depth[(int)bottomLeft.y * width + (int)bottomLeft.x], 0, 2048, 0, 1);
			float depthBottomRight = _map(depth[(int)bottomRight.y * width + (int)bottomRight.x], 0, 2048, 0, 1);
			float depthCenter = _map(depth[(int)center.y * width + (int)center.x], 0, 2048, 0, 1);

			// set the depth data
			this->topLeft.z = depthTopLeft;
			this->topRight.z = depthTopRight;
			this->bottomLeft.z = depthBottomLeft;
			this->bottomRight.z = depthBottomRight;
			this->center.z = depthCenter;
		}
	};

	// use opencv to find apriltags 36h11
	std::vector<Marker> findApriltags(cv::Mat image, std::vector<uint16_t> deapthData, std::vector<cv::aruco::PredefinedDictionaryType> dictionaryType = {cv::aruco::DICT_6X6_250})
	{
		std::vector<Marker> boxes;
		cv::Mat gray;
		cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
		for (int i = 0; i < dictionaryType.size(); i++)
		{
			std::vector<int> markerIds;
			std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
			cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(dictionaryType[i]);
			cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
			detectorParams.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
			detectorParams.cornerRefinementWinSize = 5;
			detectorParams.cornerRefinementMaxIterations = 30;
			detectorParams.cornerRefinementMinAccuracy = .5;
			cv::aruco::ArucoDetector detector(dictionary, detectorParams);
			detector.detectMarkers(gray, markerCorners, markerIds, rejectedCandidates);
			for (int i = 0; i < markerIds.size(); i++)
			{
				boxes.push_back(Marker::fromCornerPoints(markerCorners[i], markerIds[i], image.cols, image.rows));
				boxes.back().DetrmineDepth(deapthData, image.cols, image.rows);
			}
		}
		return boxes;
	}

	struct Color
	{
		float r;
		float g;
		float b;
		Color(float r, float g, float b)
		{
			this->r = r;
			this->g = g;
			this->b = b;
		}

		// hex
		Color(int hex)
		{
			this->r = ((hex >> 16) & 0xFF) / 255.0;
			this->g = ((hex >> 8) & 0xFF) / 255.0;
			this->b = ((hex) & 0xFF) / 255.0;
		}

		Color()
		{
			this->r = 0;
			this->g = 0;
			this->b = 0;
		}

		static Color Lerp(Color a, Color b, float t)
		{
			return Color(_lerp(a.r, b.r, t), _lerp(a.g, b.g, t), _lerp(a.b, b.b, t));
		}

		static Color Gradient(float value, std::vector<Color> colors)
		{
			if (colors.size() == 0)
			{
				return Color();
			}
			if (colors.size() == 1)
			{
				return colors[0];
			}
			float t = _map(value, 0, 1, 0, colors.size() - 1);
			int i = (int)t;
			t -= i;
			return Lerp(colors[i], colors[i + 1], t);
		}
	};

	// define MyFreenectDevice and Mutex class
	class Mutex
	{
	public:
		Mutex()
		{
			pthread_mutex_init(&m_mutex, NULL);
		}
		void lock()
		{
			pthread_mutex_lock(&m_mutex);
		}
		void unlock()
		{
			pthread_mutex_unlock(&m_mutex);
		}

	private:
		pthread_mutex_t m_mutex;
	};

	class Kinect : public Freenect::FreenectDevice
	{
	private:
		int DeapthDataSize = 640 * 480;
		std::vector<uint16_t> DeapthData;
		Mutex DeapthMutex;
		bool NewDeapthFrame;

		std::vector<Color> colors;
		float colorClipDistanceFront = 0;
		float colorClipDistanceBack = 0;
		float colorClipDistanceFront_off = 1.67;
		float colorClipDistanceBack_off = -1.78;

		int ImagePixelSize = 640 * 480;
		int ImageDataSize = ImagePixelSize * 3;
		std::vector<uint8_t> ImageData;
		Mutex ImageMutex;
		bool NewImageFrame;

#ifdef GraphicCard
		// opencl
		cl::Context context;
		cl::CommandQueue queue;
		cl::Program program;
		cl::Kernel kernel;
#endif

	public:
		float getColorClipDistanceFront_off()
		{
			return colorClipDistanceFront_off;
		}

		float getColorClipDistanceBack_off()
		{
			return colorClipDistanceBack_off;
		}

		float getColorClipDistanceFront()
		{
			return colorClipDistanceFront;
		}

		void setColorClipDistanceFront(float colorClipDistance)
		{
			this->colorClipDistanceFront = colorClipDistance;
		}

		float getColorClipDistanceBack()
		{
			return colorClipDistanceBack;
		}

		void setColorClipDistanceBack(float colorClipDistance)
		{
			this->colorClipDistanceBack = colorClipDistance;
		}

		void setColors(std::vector<Color> colors, bool reverse = false)
		{
			if (reverse)
			{
				std::reverse(colors.begin(), colors.end());
			}
			this->colors = colors;
		}

		std::vector<Color> getColors()
		{
			return colors;
		}

		std::vector<uint8_t> DeapthToColor()
		{
			std::vector<uint8_t> data(DeapthDataSize * 3);
#ifdef GraphicCard
			// opencl
			cl::Buffer bufferDeapth(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, DeapthDataSize * sizeof(uint16_t), &DeapthData[0]);
			cl::Buffer bufferColors(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, colors.size() * sizeof(Color), &colors[0]);
			cl::Buffer bufferColor(context, CL_MEM_WRITE_ONLY, DeapthDataSize * sizeof(uint8_t) * 3);
			kernel.setArg(0, bufferDeapth);
			kernel.setArg(1, DeapthDataSize);
			kernel.setArg(2, bufferColors);
			kernel.setArg(3, colors.size());
			kernel.setArg(4, colorClipDistanceFront_off);
			kernel.setArg(5, colorClipDistanceFront);
			kernel.setArg(6, colorClipDistanceBack_off);
			kernel.setArg(7, colorClipDistanceBack);
			kernel.setArg(8, bufferColor);
			queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(DeapthDataSize));
			queue.enqueueReadBuffer(bufferColor, CL_TRUE, 0, DeapthDataSize * sizeof(uint8_t) * 3, &data[0]);
#else
			// cpu
			int threadcount = 15;
			std::vector<std::thread> threads;

			for (int i = 0; i < threadcount; i++)
			{
				std::vector<Color> colors = this->colors;
				threads.push_back(std::thread([&, i, colors]()
											  {
					for (int j = i * DeapthDataSize / threadcount; j < (i + 1) * DeapthDataSize / threadcount; j++)
					{
						// get the depth data
						float depth = _map(DeapthData[j], 0, 2048, 0, 1);
						Color color = Color::Gradient(_map(depth, 0, 1, colorClipDistanceFront_off - colorClipDistanceFront, colorClipDistanceBack_off - colorClipDistanceBack), colors);
						// set the color
						data[j * 3] = color.r * 255;
						data[j * 3 + 1] = color.g * 255;
						data[j * 3 + 2] = color.b * 255;
					} }));
			}

			for (int i = 0; i < threadcount; i++)
			{
				threads[i].join();
			}
#endif
			return data;
		}

		Kinect(freenect_context *_ctx, int _index) : Freenect::FreenectDevice(_ctx, _index)
		{
			DeapthData.resize(DeapthDataSize);
			NewDeapthFrame = false;
			ImageData.resize(ImageDataSize);
			NewImageFrame = false;

#ifdef GraphicCard
			// opencl
			std::vector<cl::Platform> platforms;
			cl::Platform::get(&platforms);
			cl::Platform platform = platforms[0];
			std::vector<cl::Device> devices;
			platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
			cl::Device device = devices[0];
			context = cl::Context(device);
			queue = cl::CommandQueue(context, device);
			std::ifstream file("DepthToColor.cl");
			std::string prog(std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));
			program = cl::Program(context, prog);
			program.build(devices);
			kernel = cl::Kernel(program, "depthToColor");
#endif
		}

		void VideoCallback(void *_rgb, uint32_t timestamp)
		{
			ImageMutex.lock();
			uint8_t *rgb = static_cast<uint8_t *>(_rgb);
			// copy to imagedata
			if (ImageData.size() != getVideoBufferSize())
			{
				ImageData.resize(getVideoBufferSize());
			}
			copy(rgb, rgb + getVideoBufferSize(), ImageData.begin());
			NewImageFrame = true;
			ImageMutex.unlock();
		};

		void DepthCallback(void *_depth, uint32_t timestamp)
		{
			DeapthMutex.lock();
			uint16_t *depth = static_cast<uint16_t *>(_depth);
			// copy to deapthdata
			if (DeapthData.size() != getDepthBufferSize())
			{
				DeapthData.resize(getDepthBufferSize());
			}
			copy(depth, depth + getDepthBufferSize(), DeapthData.begin());
			NewDeapthFrame = true;
			DeapthMutex.unlock();
		}

		bool getRGB(cv::Mat &output)
		{
			ImageMutex.lock();
			if (NewImageFrame)
			{
				// copy to image mat
				if (output.rows != 480 || output.cols != 640)
				{
					output = cv::Mat(480, 640, CV_8UC3, &ImageData[0]);
				}
				else
				{
					output.data = &ImageData[0];
				}
				NewImageFrame = false;
				ImageMutex.unlock();
				return true;
			}
			else
			{
				ImageMutex.unlock();
				return false;
			}
		}

		bool getDepth(cv::Mat &output)
		{
			DeapthMutex.lock();
			if (NewDeapthFrame)
			{
				// create deapth mat using deapthtoColor
				if (output.rows != 480 || output.cols != 640)
				{
					output = cv::Mat(480, 640, CV_8UC3, &DeapthToColor()[0]);
				}
				else
				{
					memccpy(output.data, &DeapthToColor()[0], 0, DeapthToColor().size());
				}
				NewDeapthFrame = false;
				DeapthMutex.unlock();
				return true;
			}
			else
			{
				DeapthMutex.unlock();
				return false;
			}
		}

		std::vector<Marker> GetMarkers(std::vector<cv::aruco::PredefinedDictionaryType> dictionaryType = {cv::aruco::DICT_6X6_250})
		{
			std::vector<Marker> markers;
			ImageMutex.lock();
			markers = findApriltags(cv::Mat(480, 640, CV_8UC3, &ImageData[0]), DeapthData, dictionaryType);
			ImageMutex.unlock();
			return markers;
		}
	};

	static Kinect *GetDevice(int id)
	{
		static Freenect::Freenect freenect;
		return &freenect.createDevice<Kinect>(id);
	}

} // namespace FRC - Kinect

// define OpenGL variables
GLuint gl_depth_tex;
GLuint gl_rgb_tex;
int g_argc;
char **g_argv;
int got_frames(0);
int window(0);

// define libfreenect variables
Freenect::Freenect freenect;
FRC_Kinect::Kinect *device;
double freenect_angle(0);
freenect_video_format requested_format(FREENECT_VIDEO_RGB);

// define Kinect Device control elements
// glutKeyboardFunc Handler
void keyPressed(unsigned char key, int x, int y)
{
	if (key == 27)
	{
		device->setLed(LED_OFF);
		freenect_angle = 0;
		glutDestroyWindow(window);
	}
	if (key == '1')
	{
		device->setLed(LED_GREEN);
	}
	if (key == '2')
	{
		device->setLed(LED_RED);
	}
	if (key == '3')
	{
		device->setLed(LED_YELLOW);
	}
	if (key == '4')
	{
		device->setLed(LED_BLINK_GREEN);
	}
	if (key == '5')
	{
		// 5 is the same as 4
		device->setLed(LED_BLINK_GREEN);
	}
	if (key == '6')
	{
		device->setLed(LED_BLINK_RED_YELLOW);
	}
	if (key == '0')
	{
		device->setLed(LED_OFF);
	}
	if (key == 'f')
	{
		if (requested_format == FREENECT_VIDEO_IR_8BIT)
		{
			requested_format = FREENECT_VIDEO_RGB;
		}
		else if (requested_format == FREENECT_VIDEO_RGB)
		{
			requested_format = FREENECT_VIDEO_YUV_RGB;
		}
		else
		{
			requested_format = FREENECT_VIDEO_IR_8BIT;
		}
		device->setVideoFormat(requested_format);
	}

	if (key == 'w')
	{
		freenect_angle++;
		if (freenect_angle > 30)
		{
			freenect_angle = 30;
		}
	}
	if (key == 's' || key == 'd')
	{
		freenect_angle = 0;
	}
	if (key == 'x')
	{
		freenect_angle--;
		if (freenect_angle < -30)
		{
			freenect_angle = -30;
		}
	}
	if (key == 'e')
	{
		freenect_angle = 10;
	}
	if (key == 'c')
	{
		freenect_angle = -10;
	}
	device->setTiltDegrees(freenect_angle);

	if (key == 'p')
	{
		device->setColorClipDistanceFront(device->getColorClipDistanceFront() + 0.01);
	}
	if (key == 'o')
	{
		device->setColorClipDistanceFront(device->getColorClipDistanceFront() - 0.01);
	}
	if (key == 'l')
	{
		device->setColorClipDistanceBack(device->getColorClipDistanceBack() + 0.01);
	}
	if (key == 'k')
	{
		device->setColorClipDistanceBack(device->getColorClipDistanceBack() - 0.01);
	}
}

void DrawCircle(float cx, float cy, float r, int num_segments, bool normalized = true, int width = 640, int height = 480)
{
	if (normalized)
	{
		cx = cx * width;
		cy = cy * height;
		r = r;
	}
	glBegin(GL_TRIANGLE_FAN);
	for (int i = 0; i < num_segments; i++)
	{
		float theta = 2.0f * 3.1415926f * float(i) / float(num_segments);
		float x = r * cosf(theta);
		float y = r * sinf(theta);
		glVertex2f(x + cx, y + cy);
	}
	glEnd();
}

void DrawBox(FRC_Kinect::boundingBox box, bool normalized = true, int width = 640, int height = 480)
{
	if (normalized)
	{
		box.topLeft.x = box.topLeft.x * width;
		box.topLeft.y = box.topLeft.y * height;
		box.topRight.x = box.topRight.x * width;
		box.topRight.y = box.topRight.y * height;
		box.bottomLeft.x = box.bottomLeft.x * width;
		box.bottomLeft.y = box.bottomLeft.y * height;
		box.bottomRight.x = box.bottomRight.x * width;
		box.bottomRight.y = box.bottomRight.y * height;
		box.center.x = box.center.x * width;
		box.center.y = box.center.y * height;
	}
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f);
	glVertex2f(box.topLeft.x, box.topLeft.y);
	glTexCoord2f(1.0f, 0.0f);
	glVertex2f(box.topRight.x, box.topRight.y);
	glTexCoord2f(1.0f, 1.0f);
	glVertex2f(box.bottomRight.x, box.bottomRight.y);
	glTexCoord2f(0.0f, 1.0f);
	glVertex2f(box.bottomLeft.x, box.bottomLeft.y);
	glEnd();
}

void DrawText(float x, float y, char *string, bool normalized = true, int width = 640, int height = 480)
{
	if (normalized)
	{
		x = x * width;
		y = y * height;
	}
	glRasterPos2f(x, y);
	for (int i = 0; i < strlen(string); i++)
	{
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, string[i]);
	}
}

// define OpenGL functions
void DrawGLScene()
{
	cv::Mat depth;
	cv::Mat rgb;

	// using getTiltDegs() in a closed loop is unstable
	/*if(device->getState().m_code == TILT_STATUS_STOPPED){
	  freenect_angle = device->getState().getTiltDegs();
	}*/
	device->updateState();
	printf("\r demanded tilt angle: %+4.2f device tilt angle: %+4.2f\n", freenect_angle, device->getState().getTiltDegs());
	printf("\r color clip distance front: %+4.2f color clip distance back: %+4.2f\n", device->getColorClipDistanceFront(), device->getColorClipDistanceBack());
	printf("\r applied clip distance front: %+4.2f applied clip distance back: %+4.2f\n", device->getColorClipDistanceFront_off() - device->getColorClipDistanceFront(), device->getColorClipDistanceBack_off() - device->getColorClipDistanceBack());

	device->getDepth(depth);
	device->getRGB(rgb);

	got_frames = 0;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();

	glEnable(GL_TEXTURE_2D);

	glBindTexture(GL_TEXTURE_2D, gl_depth_tex);
	glTexImage2D(GL_TEXTURE_2D, 0, 4, 640, 480, 0, GL_RGB, GL_UNSIGNED_BYTE, &depth.data[0]);

	DrawBox(FRC_Kinect::boundingBox(glm::vec3(0, 0, 0), glm::vec3(640, 0, 0), glm::vec3(0, 480, 0), glm::vec3(640, 480, 0)),false);

	glBindTexture(GL_TEXTURE_2D, gl_rgb_tex);
	glTexImage2D(GL_TEXTURE_2D, 0, 4, 640, 480, 0, GL_RGB, GL_UNSIGNED_BYTE, &rgb.data[0]);

	DrawBox(FRC_Kinect::boundingBox(glm::vec3(640, 0, 0), glm::vec3(1280, 0, 0), glm::vec3(640, 480, 0), glm::vec3(1280, 480, 0)),false);

	glDisable(GL_TEXTURE_2D);
	// find apriltags
	std::vector<FRC_Kinect::Marker> boxes = device->GetMarkers({cv::aruco::DICT_4X4_250, cv::aruco::DICT_5X5_250, cv::aruco::DICT_6X6_250, cv::aruco::DICT_7X7_250});
	for (int i = 0; i < boxes.size(); i++)
	{
		FRC_Kinect::Marker box = boxes[i];
		// shift render position to right side
		glTranslatef(640, 0, 0);

		glColor3f(1.0f, 0.0f, 0.0f);
		DrawBox(box);

		// draw center point
		glColor3f(0.0f, 1.0f, 0.0f);
		//TOOD: fix scale by depth
		float scale = FRC_Kinect::_map(1-box.center.z, 0, 1, 0, 50);
		DrawCircle(box.center.x, box.center.y, scale, 10);

		// draw id text over box
		char idText[10];
		sprintf(idText, "ID:%d Depth:%2.3f", box.id, box.center.z);
		glColor4f(0.0f, 0.0f, 0.0f, 255.0f);
		DrawText(box.center.x - 5, box.center.y - 15, idText);

		glTranslatef(-640, 0, 0);
	}
	glColor3f(1.0f, 1.0f, 1.0f);
	// set console cursor to 0,0
	printf("\033[2J\033[1;1H");
	fflush(stdout);
	glutSwapBuffers();
}

void InitGL()
{
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClearDepth(1.0);
	glDepthFunc(GL_LESS);
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glShadeModel(GL_SMOOTH);
	glGenTextures(1, &gl_depth_tex);
	glBindTexture(GL_TEXTURE_2D, gl_depth_tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glGenTextures(1, &gl_rgb_tex);
	glBindTexture(GL_TEXTURE_2D, gl_rgb_tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, 640 * 2, 480, 0, 0.0f, 1.0f);
	glMatrixMode(GL_MODELVIEW);
}

void displayKinectData(FRC_Kinect::Kinect *device)
{
	glutInit(&g_argc, g_argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH);
	glutInitWindowSize(640 * 2, 480);
	glutInitWindowPosition(0, 0);
	window = glutCreateWindow("c++ wrapper example");
	glutDisplayFunc(&DrawGLScene);
	glutIdleFunc(&DrawGLScene);
	glutKeyboardFunc(&keyPressed);
	InitGL();
	glutMainLoop();
}
// define main function
int main(int argc, char **argv)
{
	// Get Kinect Device
	device = &freenect.createDevice<FRC_Kinect::Kinect>(0);

	// Set Kinect Device Colors
	std::vector<FRC_Kinect::Color> colors;
	colors.push_back(FRC_Kinect::Color(0x03001e));
	colors.push_back(FRC_Kinect::Color(0x7303c0));
	colors.push_back(FRC_Kinect::Color(0xec38bc));
	colors.push_back(FRC_Kinect::Color(0xfdeff9));
	device->setColors(colors);

	// Start Kinect Device
	device->setTiltDegrees(0);
	device->startVideo();
	device->startDepth();
	// handle Kinect Device Data
	device->setLed(LED_BLINK_GREEN);
	displayKinectData(device);
	// Stop Kinect Device
	device->stopVideo();
	device->stopDepth();
	device->setLed(LED_OFF);
	return 0;
}