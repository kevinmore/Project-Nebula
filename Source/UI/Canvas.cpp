#include "Canvas.h"

#include <QtCore/QCoreApplication>
#include <QtGui/QKeyEvent>
#include <QtGui/QOpenGLContext>
#include <QtCore/QTimer>

Canvas::Canvas(QScreen *screen)
	: QWindow(screen),
	  m_context(new QOpenGLContext),
	  m_scene(new Scene(this)),
	  m_rightButtonPressed(false),
	  m_middleButtonPressed(false),
	  m_cameraSpeed(5000.0f),
	  m_cameraSensitivity(0.2f)
{
	// It defines the type of the rendering area, in our case it is an OpenGL area
	setSurfaceType(QSurface::OpenGLSurface);

	// Defined the properties of the rendering area
	QSurfaceFormat format;

	format.setDepthBufferSize(24);
	format.setMajorVersion(4);
	format.setMinorVersion(3);
	format.setSamples(16); // Multisampling x16
	format.setProfile(QSurfaceFormat::CoreProfile); // Functionality deprecated in OpenGL version 3.0 is not available.
	format.setOption(QSurfaceFormat::DebugContext); // Used to request a debug context with extra debugging information.
	format.setRenderableType(QSurfaceFormat::OpenGL); // Desktop OpenGL rendering
	format.setSwapBehavior(QSurfaceFormat::TripleBuffer); // Decrease the risk of skipping a frame when the rendering rate is just barely keeping up with the screen refresh rate.

	// Create the window
	setFormat(format);
	create();
	resize(800, 600);
	setTitle("Canvas");

	// Apply the format
	m_context->setFormat(format);
	m_context->create();
	m_context->makeCurrent(this);

	// Define the scene
	m_scene->setContext(m_context.data());
	m_scene->setCanvas(this);

	qDebug() << endl <<  "- OpenGL version :" << reinterpret_cast<const char*>(glGetString(GL_VERSION));
	qDebug() << "- GLSL version :" << reinterpret_cast<const char*>(glGetString(GL_SHADING_LANGUAGE_VERSION));
	qDebug() << "- Vendor :" << reinterpret_cast<const char*>(glGetString(GL_VENDOR));
	qDebug() << "- Renderer (GPU) :" << reinterpret_cast<const char*>(glGetString(GL_RENDERER)) << endl;

	m_updateTimer.start(); // Timer for updating the scene (camera ... etc)

	initializeGL();

	connect(this, SIGNAL(widthChanged(int)), this, SLOT(resizeGL()));
	connect(this, SIGNAL(heightChanged(int)), this, SLOT(resizeGL()));

	resizeGL();

	// Create a timer for updating the rendering area of 60Hz
	QTimer* timer = new QTimer(this);
	connect(timer, SIGNAL(timeout()), this, SLOT(updateScene()));
	timer->start(0.016); // f = 1 / 16.10e-3 = 60Hz

}

Canvas::~Canvas() 
{
	SAFE_DELETE(m_scene);
}

/**
 * @brief Initializing the rendering area
 */
void Canvas::initializeGL()
{
    m_context->makeCurrent(this);
    m_scene->initialize();
}

/**
 * @brief Updating the rendering area
 */
void Canvas::paintGL()
{
    if(isExposed())
    {
        m_context->makeCurrent(this);
        m_context->swapBuffers(this);

        emit updateFramerate();
    }
}

/**
 * @brief Resize the rendering area
 */
void Canvas::resizeGL()
{
    m_context->makeCurrent(this);
    m_scene->resize(width(), height());
}

/**
 * @brief Update the scene
 */
void Canvas::updateScene()
{
    m_scene->update(static_cast<float>(m_updateTimer.elapsed())/1000.0f);
    paintGL();
}



Scene* Canvas::getScene()
{
	return ( static_cast<Scene*>(m_scene) );
}

void Canvas::keyPressEvent(QKeyEvent* e)
{
	Camera* camera = getScene()->getCamera();

	switch (e->key())
	{
	case Qt::Key_Escape:

		QCoreApplication::instance()->quit();
		break;

	case Qt::Key_Right:
		camera->setSideSpeed(static_cast<float>(m_cameraSpeed));
		break;

	case Qt::Key_Left:
		camera->setSideSpeed(static_cast<float>(-m_cameraSpeed));
		break;

	case Qt::Key_Up:
		camera->setForwardSpeed(static_cast<float>(m_cameraSpeed));
		break;

	case Qt::Key_Down:
		camera->setForwardSpeed(static_cast<float>(-m_cameraSpeed));
		break;

	case Qt::Key_R:
		camera->setVerticalSpeed(static_cast<float>(m_cameraSpeed));
		break;

	case Qt::Key_F:
		camera->setVerticalSpeed(static_cast<float>(-m_cameraSpeed));
		break;

	case Qt::Key_Control:
		camera->setViewCenterFixed(true);
		break;

	default:
		QWindow::keyPressEvent(e);
	}

}

void Canvas::keyReleaseEvent(QKeyEvent* e)
{
	Camera* camera = getScene()->getCamera();

	switch (e->key())
	{
	case Qt::Key_Right:
	case Qt::Key_Left:
		camera->setSideSpeed(0.0f);
		break;

	case Qt::Key_Up:
	case Qt::Key_Down:
		camera->setForwardSpeed(0.0f);
		break;

	case Qt::Key_R:
	case Qt::Key_F:
		camera->setVerticalSpeed(0.0f);
		break;

	case Qt::Key_Control:
		camera->setViewCenterFixed(false);
		break;

	default:
		QWindow::keyReleaseEvent(e);
	}

}

void Canvas::wheelEvent( QWheelEvent *e )
{
	int delta = e->delta();
	Camera* camera = getScene()->getCamera();

	Camera::CameraTranslationOption option = camera->isViewCenterFixed()
		? Camera::DontTranslateViewCenter
		: Camera::TranslateViewCenter;

	if (e->orientation() == Qt::Vertical) 
	{
		camera->translate(vec3(0, 0, 0.5 * delta), option);
	}

	e->accept();
}

void Canvas::mousePressEvent(QMouseEvent* e)
{
	m_pos = m_prevPos = e->pos();

	if(e->button() == Qt::RightButton)
	{
		m_rightButtonPressed = true;
	}
	else if (e->button() == Qt::MiddleButton)
	{
		m_middleButtonPressed = true;
		setCursor(QCursor(Qt::ClosedHandCursor));
	}

	QWindow::mousePressEvent(e);
}

void Canvas::mouseReleaseEvent(QMouseEvent* e)
{
	if(e->button() == Qt::RightButton)
	{
		m_rightButtonPressed = false;
	}
	else if (e->button() == Qt::MiddleButton)
	{
		Camera* camera = getScene()->getCamera();
		camera->setSideSpeed(0.0f);
		camera->setVerticalSpeed(0.0f);
		m_middleButtonPressed = false;
		setCursor(QCursor(Qt::ArrowCursor));
		if(camera->isFollowingTarget()) camera->setViewCenterFixed(true);
	}
	QWindow::mouseReleaseEvent(e);
}

void Canvas::mouseMoveEvent(QMouseEvent* e)
{
	m_pos = e->pos();

	float dx = static_cast<float>(m_cameraSensitivity) * (static_cast<float>(m_pos.x()) - static_cast<float>(m_prevPos.x()));
	float dy = static_cast<float>(-m_cameraSensitivity) * (static_cast<float>(m_pos.y()) - static_cast<float>(m_prevPos.y()));

	m_prevPos = m_pos;

	Camera* camera = getScene()->getCamera();

	if(m_rightButtonPressed)
	{
		camera->setPanAngle(dx);
		camera->setTiltAngle(dy);
	}
	else if (m_middleButtonPressed)
	{
		if (camera->isFollowingTarget())
		{
			camera->setViewCenterFixed(false);
			camera->setVerticalSpeed(dy * m_cameraSpeed);
		}
		else
		{
			camera->setSideSpeed(dx * m_cameraSpeed);
			camera->setVerticalSpeed(dy * m_cameraSpeed);
		}
	}

	QWindow::mouseMoveEvent(e);
}

void Canvas::setCameraSpeed(double speed)
{
	m_cameraSpeed = speed;
}

void Canvas::setCameraSensitivity(double sensitivity)
{
	m_cameraSensitivity = sensitivity;
}
