#include <libfreenect.hpp>
#include <libfreenect.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <pthread.h>
#include <algorithm>
#include <thread>

#include <GL/freeglut.h>
#include <GL/gl.h>
#include <GL/glu.h>

using namespace std;

float _lerp(float a, float b, float t){
  return a + (b - a) * t;
}

float _map(float value, float istart, float istop, float ostart, float ostop){
  return ostart + (ostop - ostart) * ((value - istart) / (istop - istart));
}

struct Color{
  float r;
  float g;
  float b;
  Color(float r, float g, float b){
    this->r = r;
    this->g = g;
    this->b = b;
  }

  //hex
  Color(int hex){
    this->r = ((hex >> 16) & 0xFF) / 255.0;
    this->g = ((hex >> 8) & 0xFF) / 255.0;
    this->b = ((hex) & 0xFF) / 255.0;
  }

  Color(){
    this->r = 0;
    this->g = 0;
    this->b = 0;
  }
  
  static Color Lerp(Color a, Color b, float t){
    return Color(_lerp(a.r, b.r, t), _lerp(a.g, b.g, t), _lerp(a.b, b.b, t));
  }
  
  static Color Gradient(float value, std::vector<Color> colors){
    if(colors.size() == 0){
      return Color();
    }
    if(colors.size() == 1){
      return colors[0];
    }
    float t = _map(value, 0, 1, 0, colors.size() - 1);
    int i = (int)t;
    t -= i;
    return Lerp(colors[i], colors[i + 1], t);
  }

  static void copyToPixel(Color c, void* pixel){
    uint8_t* p = (uint8_t*)pixel;
    p[0] = (uint8_t)(c.r * 255);
    p[1] = (uint8_t)(c.g * 255);
    p[2] = (uint8_t)(c.b * 255);
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

namespace FRC_Kinect
{
#define FREENECT_FRAME_PIX 640 * 480
#define FREENECT_VIDEO_RGB_SIZE FREENECT_FRAME_PIX * 3

  class Kinect : public Freenect::FreenectDevice
  {
  private:
    vector<uint8_t> m_buffer_depth;
    vector<uint8_t> m_buffer_video;
    Mutex m_rgb_mutex;
    Mutex m_depth_mutex;
    bool m_new_rgb_frame;
    bool m_new_depth_frame;
    std::vector<Color> colors;
    float colorClipDistanceFront = 0;
    float colorClipDistanceBack = 0;
    float colorClipDistanceFront_off = 2;
    float colorClipDistanceBack_off = -.39;

  public:

    float getColorClipDistanceFront_off(){
      return colorClipDistanceFront_off;
    }

    float getColorClipDistanceBack_off(){
      return colorClipDistanceBack_off;
    }

    float getColorClipDistanceFront(){
      return colorClipDistanceFront;
    }

    void setColorClipDistanceFront(float colorClipDistance){
      this->colorClipDistanceFront = colorClipDistance;
    }

    float getColorClipDistanceBack(){
      return colorClipDistanceBack;
    }

    void setColorClipDistanceBack(float colorClipDistance){
      this->colorClipDistanceBack = colorClipDistance;
    }

    void setColors(std::vector<Color> colors, bool reverse = false){
      if(reverse){
        std::reverse(colors.begin(), colors.end());
      }
      this->colors = colors;
    }

    Kinect(freenect_context *_ctx, int _index) : Freenect::FreenectDevice(_ctx, _index), m_buffer_depth(FREENECT_VIDEO_RGB_SIZE), m_buffer_video(FREENECT_VIDEO_RGB_SIZE), m_new_rgb_frame(false), m_new_depth_frame(false)
    {
    }

    void VideoCallback(void *_rgb, uint32_t timestamp)
    {
      m_rgb_mutex.lock();
      uint8_t *rgb = static_cast<uint8_t *>(_rgb);
      copy(rgb, rgb + getVideoBufferSize(), m_buffer_video.begin());
      m_new_rgb_frame = true;
      m_rgb_mutex.unlock();
    };

    void DepthCallback(void *_depth, uint32_t timestamp)
    {
      m_depth_mutex.lock();
      uint16_t *depth = static_cast<uint16_t *>(_depth);

      int num_threads = 15;
      std::vector<std::thread> threads;

      for (unsigned int i = 0; i < num_threads; i++)
      {
        threads.push_back(std::thread([&, i]() {
          for (unsigned int j = i * FREENECT_FRAME_PIX / num_threads; j < (i + 1) * FREENECT_FRAME_PIX / num_threads; j++)
          {
            float d = depth[j] / 2048.0;
            float v= _map(d, 0, 1, colorClipDistanceBack_off-colorClipDistanceBack, colorClipDistanceFront_off-colorClipDistanceFront);
            Color::copyToPixel(Color::Gradient(v, colors), &m_buffer_depth[j * 3]);
          }
        }));
      }
      
      for (auto &thread : threads)
      {
        thread.join();
      }

      m_new_depth_frame = true;
      m_depth_mutex.unlock();
    }
    
    bool getRGB(vector<uint8_t> &buffer)
    {
      m_rgb_mutex.lock();
      if (m_new_rgb_frame)
      {
        buffer.swap(m_buffer_video);
        m_new_rgb_frame = false;
        m_rgb_mutex.unlock();
        return true;
      }
      else
      {
        m_rgb_mutex.unlock();
        return false;
      }
    }
    
    bool getDepth(vector<uint8_t> &buffer)
    {
      m_depth_mutex.lock();
      if (m_new_depth_frame)
      {
        buffer.swap(m_buffer_depth);
        m_new_depth_frame = false;
        m_depth_mutex.unlock();
        return true;
      }
      else
      {
        m_depth_mutex.unlock();
        return false;
      }
    }
  };


  static Kinect* GetDevice(int id){
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
// define OpenGL functions
void DrawGLScene()
{
  static std::vector<uint8_t> depth(640 * 480 * 4);
  static std::vector<uint8_t> rgb(640 * 480 * 4);

  // using getTiltDegs() in a closed loop is unstable
  /*if(device->getState().m_code == TILT_STATUS_STOPPED){
    freenect_angle = device->getState().getTiltDegs();
  }*/
  device->updateState();
  printf("\r demanded tilt angle: %+4.2f device tilt angle: %+4.2f\n", freenect_angle, device->getState().getTiltDegs());
  printf("\r color clip distance front: %+4.2f color clip distance back: %+4.2f\n", device->getColorClipDistanceFront(), device->getColorClipDistanceBack());
  printf("\r applied clip distance front: %+4.2f applied clip distance back: %+4.2f\n", device->getColorClipDistanceFront_off()-device->getColorClipDistanceFront(), device->getColorClipDistanceBack_off()-device->getColorClipDistanceBack());
  fflush(stdout);

  device->getDepth(depth);
  device->getRGB(rgb);

  got_frames = 0;

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glLoadIdentity();

  glEnable(GL_TEXTURE_2D);

  glBindTexture(GL_TEXTURE_2D, gl_depth_tex);
  glTexImage2D(GL_TEXTURE_2D, 0, 4, 640, 480, 0, GL_RGB, GL_UNSIGNED_BYTE, &depth[0]);

  glBegin(GL_TRIANGLE_FAN);
  glColor4f(255.0f, 255.0f, 255.0f, 255.0f);
  glTexCoord2f(0, 0);
  glVertex3f(0, 0, 0);
  glTexCoord2f(1, 0);
  glVertex3f(640, 0, 0);
  glTexCoord2f(1, 1);
  glVertex3f(640, 480, 0);
  glTexCoord2f(0, 1);
  glVertex3f(0, 480, 0);
  glEnd();

  glBindTexture(GL_TEXTURE_2D, gl_rgb_tex);
  glTexImage2D(GL_TEXTURE_2D, 0, 4, 640, 480, 0, GL_RGB, GL_UNSIGNED_BYTE, &rgb[0]);

  glBegin(GL_TRIANGLE_FAN);
  glColor4f(255.0f, 255.0f, 255.0f, 255.0f);
  glTexCoord2f(0, 0);
  glVertex3f(640, 0, 0);
  glTexCoord2f(1, 0);
  glVertex3f(640*2, 0, 0);
  glTexCoord2f(1, 1);
  glVertex3f(640*2, 480, 0);
  glTexCoord2f(0, 1);
  glVertex3f(640, 480, 0);
  glEnd();

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
  glOrtho(0, 640*2, 480, 0, 0.0f, 1.0f);
  glMatrixMode(GL_MODELVIEW);
}

void displayKinectData(FRC_Kinect::Kinect *device)
{
  glutInit(&g_argc, g_argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH);
  glutInitWindowSize(640*2, 480);
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
  std::vector<Color> colors;
  colors.push_back(Color(0x03001e));
  colors.push_back(Color(0x7303c0));
  colors.push_back(Color(0xec38bc));
  colors.push_back(Color(0xfdeff9));
  device->setColors(colors, true);

  // Start Kinect Device
  device->setTiltDegrees(0);
  device->startVideo();
  device->startDepth();
  // handle Kinect Device Data
  device->setLed(LED_RED);
  displayKinectData(device);
  // Stop Kinect Device
  device->stopVideo();
  device->stopDepth();
  device->setLed(LED_OFF);
  return 0;
}
