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
	  m_cameraSpeed(getScene()->getCamera()->speed()),
	  m_cameraSensitivity(getScene()->getCamera()->sensitivity())
{
	// It defines the type of the rendering area, in our case it is an OpenGL area
	setSurfaceType(QSurface::OpenGLSurface);

	// Defined the properties of the rendering area
	QSurfaceFormat format;

	format.setDepthBufferSize(24);
	format.setMajorVersion(4);
	format.setMinorVersion(3);
	format.setSamples(16); // Multi-sampling x16
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
	// only renders the exposed canvas
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
	case Qt::Key_D:
		camera->setSideSpeed(static_cast<float>(m_cameraSpeed));
		break;

	case Qt::Key_A:
		camera->setSideSpeed(static_cast<float>(-m_cameraSpeed));
		break;

	case Qt::Key_W:
		camera->setForwardSpeed(static_cast<float>(m_cameraSpeed));
		break;

	case Qt::Key_S:
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
	case Qt::Key_D:
	case Qt::Key_A:
		camera->setSideSpeed(0.0f);
		break;

	case Qt::Key_W:
	case Qt::Key_S:
		camera->setForwardSpeed(0.0f);
		break;

	case Qt::Key_R:
	case Qt::Key_F:
		camera->setVerticalSpeed(0.0f);
		break;

	case Qt::Key_Control:
		camera->setViewCenterFixed(false);
		break;

	case Qt::Key_Delete:
		emit deleteObject();
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
		camera->translate(vec3(0, 0, m_cameraSpeed * delta * 0.001), option);
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
		// clear the debug mode first
		//getScene()->toggleDebugMode(false);
		// emit an empty game object
		//emit objectPicked(GameObjectPtr());

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
	else if (e->button() == Qt::LeftButton)
	{
		// clear the debug mode first
		getScene()->toggleDebugMode(false);

		// do a ray casting to pick game objects
		GameObjectPtr selectedObject = pickObject(e->pos());
		if (selectedObject)
		{
			// show the bounding box
			ModelPtr model = selectedObject->getComponent("Model").dynamicCast<IModel>();
			model->showBoundingVolume();
		}
		// emit the signal
		emit objectPicked(selectedObject);
	}
	QWindow::mouseReleaseEvent(e);
}

void Canvas::mouseMoveEvent(QMouseEvent* e)
{
	m_pos = e->pos();

	if(m_rightButtonPressed)
	{
		float dx = static_cast<float>(m_cameraSensitivity) * (static_cast<float>(m_pos.x()) - static_cast<float>(m_prevPos.x()));
		float dy = static_cast<float>(-m_cameraSensitivity) * (static_cast<float>(m_pos.y()) - static_cast<float>(m_prevPos.y()));

		m_prevPos = m_pos;

		Camera* camera = getScene()->getCamera();

		camera->setPanAngle(dx);
		camera->setTiltAngle(dy);
	}
	else if (m_middleButtonPressed)
	{
		float dx = static_cast<float>(m_cameraSensitivity) * (static_cast<float>(m_pos.x()) - static_cast<float>(m_prevPos.x()));
		float dy = static_cast<float>(-m_cameraSensitivity) * (static_cast<float>(m_pos.y()) - static_cast<float>(m_prevPos.y()));

		m_prevPos = m_pos;

		Camera* camera = getScene()->getCamera();

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
	getScene()->getCamera()->setSpeed(speed);
}

void Canvas::setCameraSensitivity(double sensitivity)
{
	m_cameraSensitivity = sensitivity;
	getScene()->getCamera()->setSensitivity(sensitivity);
}

bool Canvas::testRaySpehreIntersection( const vec3& rayDirection, const float radius, const GameObjectPtr target, float& intersectionDistance )
{
	// work out components of quadratic
	vec3 rayOrigin = getScene()->getCamera()->position();
	vec3 dist_to_sphere = rayOrigin - target->position();
	float b = vec3::dotProduct(rayDirection, dist_to_sphere);
	float c = dist_to_sphere.lengthSquared() - radius * radius;
	float b_squared_minus_c = b * b - c;
	// check for "imaginary" answer. == ray completely misses sphere
	if (b_squared_minus_c < 0.0f) {
		return false;
	}
	// check for ray hitting twice (in and out of the sphere)
	if (b_squared_minus_c > 0.0f) {
		// get the 2 intersection distances along ray
		float t_a = -b + sqrt (b_squared_minus_c);
		float t_b = -b - sqrt (b_squared_minus_c);
		intersectionDistance = t_b;
		// if behind viewer, throw one or both away
		if (t_a < 0.0) {
			if (t_b < 0.0) {
				return false;
			}
		} else if (t_b < 0.0) {
			intersectionDistance = t_a;
		}

		return true;
	}
	// check for ray hitting once (skimming the surface)
	if (0.0f == b_squared_minus_c) {
		// if behind viewer, throw away
		float t = -b + sqrt (b_squared_minus_c);
		if (t < 0.0f) {
			return false;
		}
		intersectionDistance = t;
		return true;
	}
	// note: could also check if ray origin is inside sphere radius
	return false;
}

bool Canvas::testRayOBBIntersection( const vec3& rayDirection, const GameObjectPtr target, float& intersectionDistance )
{
	// Retrieve the bounding box
	ModelPtr model = target->getComponent("Model").dynamicCast<IModel>();
	if(!model) return false;
	BoxColliderPtr box = model->getBoundingBox();
	vec3 center = box->getCenter();
	vec3 halfExtents = box->getHalfExtents();

	// model space aabb boundaries
	vec3 aabbMin = center - halfExtents;
	vec3 aabbMax = center + halfExtents;

	// apply a necessary scale to the extents
// 	vec3 scale = target->scale();
// 	aabbMin.setX(aabbMin.x() * scale.x());
// 	aabbMin.setY(aabbMin.y() * scale.y());
// 	aabbMin.setZ(aabbMin.z() * scale.z());
// 	aabbMax.setX(aabbMax.x() * scale.x());
// 	aabbMax.setY(aabbMax.y() * scale.y());
// 	aabbMax.setZ(aabbMax.z() * scale.z());

	mat4 modelMatrix = target->getTransformMatrix();

	// Intersection method from Real-Time Rendering and Essential Mathematics for Games
	float tMin = 0.0f;
	float tMax = 100000.0f;
	vec3 rayOrigin = getScene()->getCamera()->position();
	vec3 targetPos = target->position();
	vec3 delta = targetPos - rayOrigin;
	quat rotation = Math::Quaternion::computeQuaternion(target->rotation());

	// Test intersection with the 2 planes perpendicular to the OBB's X axis
	{
		vec3 xaxis = rotation.rotatedVector(Math::Vector3::UNIT_X);
		float e = vec3::dotProduct(xaxis, delta);
		float f = vec3::dotProduct(rayDirection, xaxis);

		if ( fabs(f) > 0.001f ){ // Standard case

			float t1 = (e+aabbMin.x())/f; // Intersection with the "left" plane
			float t2 = (e+aabbMax.x())/f; // Intersection with the "right" plane
			// t1 and t2 now contain distances betwen ray origin and ray-plane intersections

			// We want t1 to represent the nearest intersection, 
			// so if it's not the case, invert t1 and t2
			if (t1>t2){
				float w=t1;t1=t2;t2=w; // swap t1 and t2
			}

			// tMax is the nearest "far" intersection (amongst the X,Y and Z planes pairs)
			if ( t2 < tMax )
				tMax = t2;
			// tMin is the farthest "near" intersection (amongst the X,Y and Z planes pairs)
			if ( t1 > tMin )
				tMin = t1;

			// And here's the trick :
			// If "far" is closer than "near", then there is NO intersection.
			// See the images in the tutorials for the visual explanation.
			if (tMax < tMin )
				return false;

		}else{ // Rare case : the ray is almost parallel to the planes, so they don't have any "intersection"
			if(-e+aabbMin.x() > 0.0f || -e+aabbMax.x() < 0.0f)
				return false;
		}
	}

	// Test intersection with the 2 planes perpendicular to the OBB's Y axis
	// Exactly the same thing than above.
	{
		vec3 yaxis = rotation.rotatedVector(Math::Vector3::UNIT_Y);
		float e = vec3::dotProduct(yaxis, delta);
		float f = vec3::dotProduct(rayDirection, yaxis);

		if ( fabs(f) > 0.001f ){

			float t1 = (e+aabbMin.y())/f;
			float t2 = (e+aabbMax.y())/f;

			if (t1>t2){float w=t1;t1=t2;t2=w;}

			if ( t2 < tMax )
				tMax = t2;
			if ( t1 > tMin )
				tMin = t1;
			if (tMin > tMax)
				return false;

		}else{
			if(-e+aabbMin.y() > 0.0f || -e+aabbMax.y() < 0.0f)
				return false;
		}
	}

	// Test intersection with the 2 planes perpendicular to the OBB's Z axis
	// Exactly the same thing than above.
	{
		vec3 zaxis = rotation.rotatedVector(Math::Vector3::UNIT_Z);
		float e = vec3::dotProduct(zaxis, delta);
		float f = vec3::dotProduct(rayDirection, zaxis);

		if ( fabs(f) > 0.001f ){

			float t1 = (e+aabbMin.z())/f;
			float t2 = (e+aabbMax.z())/f;

			if (t1>t2){float w=t1;t1=t2;t2=w;}

			if ( t2 < tMax )
				tMax = t2;
			if ( t1 > tMin )
				tMin = t1;
			if (tMin > tMax)
				return false;

		}else{
			if(-e+aabbMin.z() > 0.0f || -e+aabbMax.z() < 0.0f)
				return false;
		}
	}

	intersectionDistance = tMin;
	return true;
}

vec3 Canvas::getRayFromMouse( const QPoint& mousePos)
{
	// The ray Start and End positions, in Normalized Device Coordinates (Have you read Tutorial 4 ?)
	vec4 lRayStart_NDC(
		((float)mousePos.x()/(float)width()  - 0.5f) * 2.0f,
		(0.5f - (float)mousePos.y()/(float)height()) * 2.0f,
		-1.0, // The near plane maps to Z=-1 in Normalized Device Coordinates
		1.0f
		);
	vec4 lRayEnd_NDC(
		((float)mousePos.x()/(float)width()  - 0.5f) * 2.0f,
		(0.5f - (float)mousePos.y()/(float)height()) * 2.0f,
		0.0,
		1.0f
		);

	Camera* camera = getScene()->getCamera();

	mat4 M = (camera->projectionMatrix() * camera->viewMatrix()).inverted();

	vec4 lRayEnd_world   = M * lRayEnd_NDC  ; lRayEnd_world  /=lRayEnd_world.w();

	vec3 lRayDir_world(lRayEnd_world - camera->position());
	return lRayDir_world.normalized();
}

GameObjectPtr Canvas::pickObject(const QPoint& mousePos)
{
	// get a ray from the mouse position
	vec3 direction = getRayFromMouse(mousePos);

	// prepare the container (a ray can hit several object)
	QMap<float, GameObjectPtr> distanceMap;
	foreach(GameObjectPtr go, getScene()->objectManager()->m_gameObjectMap.values())
	{
		float distance = 0.0f;
		if(testRayOBBIntersection(direction, go, distance))
		{
			distanceMap[distance] = go;
		}
	}

	// get the nearest object
	float minDist = 10000.0f;
	foreach(float val, distanceMap.keys())
	{
		minDist = qMin(minDist, val);
	}

	// NOTE: the returned object can be NULL
	return distanceMap[minDist];
}

void Canvas::showGPUInfo()
{
	QString info =  "OpenGL version - ";
	info += reinterpret_cast<const char*>(glGetString(GL_VERSION));
	info += "\n\nGLSL version - ";
	info += reinterpret_cast<const char*>(glGetString(GL_SHADING_LANGUAGE_VERSION));
	info += "\n\nVendor - ";
	info += reinterpret_cast<const char*>(glGetString(GL_VENDOR));
	info += "\n\nRenderer (GPU) - ";
	info += reinterpret_cast<const char*>(glGetString(GL_RENDERER));

	QMessageBox::about(0, tr("GPU Information"), info);
}
